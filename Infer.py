import base64
import io
import json
import multiprocessing
import traceback
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

class RVCWorker:
    """RVC model loading, streaming conversion and file conversion."""

    def initialize(self):
        runtime, model_path, index_path = sys.argv[1:4]
        os.chdir(runtime)
        sys.path.insert(0, runtime)
        os.environ["rmvpe_root"] = os.path.join(runtime, "assets", "rmvpe")
        # Some upstream dependencies parse argv. They must not see our paths.
        sys.argv = [sys.argv[0]]
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torchaudio
        import faiss
        import soundfile as sf
        from types import SimpleNamespace
        from infer import rtrvc as rt

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        faiss.omp_set_num_threads(min(4, os.cpu_count() or 1))
        rt.printt = lambda *args: None
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        config = SimpleNamespace(device=device, is_half=device.startswith("cuda"),
                                 use_jit=False, dml=False, x_pad=1, x_query=6, x_center=38, x_max=41)
        rvc = rt.RVC(0, 0, model_path, index_path, 0, config)
        if getattr(rvc, "net_g", None) is None or not hasattr(rvc, "tgt_sr"):
            raise RuntimeError("RVC 模型加载失败，请检查模型及运行环境。")
        speaker = 0
        to16 = torchaudio.transforms.Resample(48000, 16000).to(device)
        to48 = (torchaudio.transforms.Resample(rvc.tgt_sr, 48000).to(device)
                if rvc.tgt_sr != 48000 else None)
        state = None
        file_pipeline = None

        def phase_vocoder(a, b, fade_out, fade_in):
            window = torch.sqrt(fade_out * fade_in)
            fa = torch.fft.rfft(a * window)
            fb = torch.fft.rfft(b * window)
            absab = torch.abs(fa) + torch.abs(fb)
            n = a.shape[0]
            if n % 2 == 0:
                absab[1:-1] *= 2
            else:
                absab[1:] *= 2
            phia = torch.angle(fa)
            phib = torch.angle(fb)
            deltaphase = phib - phia
            deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
            w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
            t = torch.arange(n).unsqueeze(-1).to(a) / n
            result = (
                a * (fade_out**2)
                + b * (fade_in**2)
                + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
            )
            return result

        def configure(o):
            nonlocal speaker
            rate = float(o["indexRate"])
            if not 0 <= rate <= 1:
                raise ValueError("索引率必须在 0～1 之间。")
            speaker = int(o["speakerId"])
            if not 0 <= speaker < rvc.net_g.emb_g.num_embeddings:
                raise ValueError("角色 ID 不在模型支持的范围内。")
            if rate and not os.path.isfile(index_path):
                raise ValueError("此模型没有索引文件，请把索引率设为 0。")
            if rate and not hasattr(rvc, "index"):
                rvc.index = faiss.deserialize_index(np.fromfile(index_path, dtype=np.uint8))
                rvc.big_npy = rvc.index.reconstruct_n(0, rvc.index.ntotal)
            rvc.index_rate = rate
            rvc.change_key(int(o["pitch"]))

        def stream(samples, o, reset):
            nonlocal state
            block = int(o["blockMilliseconds"]) * 48
            overlap = int(o["crossfadeMilliseconds"]) * 48
            extra = int(o["contextMilliseconds"]) * 48
            search = 480
            key = (block, overlap, extra)
            if samples.size != block:
                raise ValueError("音频分块大小不匹配。")
            if reset or state is None or state[0] != key:
                history = torch.zeros(extra + overlap + search + block, device=device)
                tail = torch.zeros(overlap, device=device)
                fade = torch.sin(torch.linspace(0, torch.pi / 2, overlap, device=device)) ** 2
                state = (key, history, tail, fade)
                rvc.cache_pitch.zero_()
                rvc.cache_pitchf.zero_()
            _, history, tail, fade = state
            history[:-block] = history[block:].clone()
            history[-block:] = torch.from_numpy(samples.copy()).to(device)
            wav16 = to16(history)
            with torch.no_grad():
                converted = rvc.infer(wav16, block // 3, extra // 480,
                    (block + overlap + search) // 480, o["f0Method"],
                    protect=float(o["protect"]), speaker_id=speaker,
                    filter_radius=o["realtimeHarvestFilterRadius"])
                if to48 is not None:
                    converted = to48(converted)
                # Envelope matching uses local 20 ms windows and preserves transitions.
                mix = float(o["rmsMixRate"])
                if mix < 1:
                    def envelope(wav):
                        return F.avg_pool1d(F.pad(wav.square()[None, None], (480, 480), mode="replicate"),
                                            960, stride=480).sqrt()
                    src = envelope(history[extra:])
                    dst = envelope(converted)
                    ratio = F.interpolate(src, size=converted.numel(), mode="linear", align_corners=False)[0, 0]
                    denom = F.interpolate(dst, size=converted.numel(), mode="linear", align_corners=False)[0, 0]
                    converted *= (ratio / denom.clamp_min(1e-3)).clamp_max(10).pow(1 - mix)
                if converted.numel() < block + overlap + search:
                    raise RuntimeError("RVC 返回的音频长度不足。")
                # Normalized cross correlation aligns this chunk with the retained tail.
                probe = converted[:overlap + search].view(1, 1, -1)
                correlation = F.conv1d(probe, tail.view(1, 1, -1))[0, 0]
                energy = F.conv1d(probe.square(), torch.ones(1, 1, overlap, device=device))[0, 0].sqrt()
                offset = int((correlation / energy.clamp_min(1e-8)).argmax().item())
                converted = converted[offset:]
                if o["usePhaseVocoder"]:
                    converted[:overlap] = phase_vocoder(tail, converted[:overlap], 1 - fade, fade)
                else:
                    converted[:overlap] = converted[:overlap] * fade + tail * (1 - fade)
                tail[:] = converted[block:block + overlap]
                output = converted[:block].clamp(-1, 1).cpu().numpy().astype("<f4")
            if not np.isfinite(output).all():
                raise RuntimeError("模型输出包含无效数值，请检查模型与 RVC 运行环境。")
            return output

        def process(request):
            nonlocal state, file_pipeline
            o = request["options"]
            configure(o)
            start = time.perf_counter()
            if request["command"] == "stream":
                samples = np.frombuffer(base64.b64decode(request["pcm"], validate=True), dtype="<f4")
                output = stream(samples, o, request.get("reset", False))
                return {"ok": True, "pcm": base64.b64encode(output.tobytes()).decode("ascii"),
                       "milliseconds": (time.perf_counter() - start) * 1000}
            elif request["command"] == "file":
                output_rate = o.get("outputSampleRate", 0)
                if output_rate not in (0, 16000, 22050, 24000, 32000, 44100, 48000):
                    raise ValueError("不支持的文件输出采样率。")
                from scipy.signal import resample_poly
                from math import gcd
                from infer.vc.pipeline import Pipeline
                samples, sr = sf.read(request["path"], dtype="float32", always_2d=True)
                samples = samples.mean(axis=1)
                if len(samples) < sr // 10 or not np.isfinite(samples).all():
                    raise ValueError("请输入至少 0.1 秒的有效音频。")
                factor = gcd(sr, 16000)
                samples = resample_poly(samples, 16000 // factor, sr // factor)
                peak = np.abs(samples).max() / .95
                if peak > 1:
                    samples /= peak
                if file_pipeline is None:
                    file_pipeline = Pipeline(rvc.tgt_sr, config)
                if hasattr(rvc, "model_rmvpe"):
                    file_pipeline.model_rmvpe = rvc.model_rmvpe
                if hasattr(rvc, "model_fcpe"):
                    file_pipeline.model_fcpe = rvc.model_fcpe
                audio = file_pipeline.pipeline(rvc.model, rvc.net_g, speaker, samples,
                    [0, 0, 0], o["pitch"], o["f0Method"], index_path, o["indexRate"], rvc.if_f0,
                    rvc.tgt_sr, output_rate, o["rmsMixRate"], rvc.version, o["protect"],
                    filter_radius=o["filterRadius"])
                buffer = io.BytesIO()
                sf.write(buffer, audio, output_rate or rvc.tgt_sr, format="WAV", subtype="PCM_16")
                state = None
                return {"ok": True, "wav": base64.b64encode(buffer.getvalue()).decode("ascii"),
                       "milliseconds": (time.perf_counter() - start) * 1000}
            else:
                raise ValueError("不支持的操作。")

        self._process = process
        return {"device": device, "sampleRate": rvc.tgt_sr}

    def process(self, request):
        return self._process(request)


    @classmethod
    def run(cls):
        multiprocessing.freeze_support()
        protocol = sys.stdout
        sys.stdout = sys.stderr

        def reply(value):
            protocol.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            protocol.flush()

        try:
            worker = cls()
            metadata = worker.initialize()
            reply({**metadata, "ok": True, "protocolVersion": 1})
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("Request must be a JSON object")
                    reply(worker.process(request))
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    reply({"ok": False, "error": str(exc)})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "error": str(exc)})
            raise SystemExit(1)
        finally:
            sys.stdout = protocol


if __name__ == "__main__":
    RVCWorker.run()
