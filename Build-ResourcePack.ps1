#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Version,
    [string]$OutputDirectory,
    [string]$Python,
    [ValidateRange(0, 9)][int]$CompressionLevel = 9,
    [switch]$Plan,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot '..\..\Build-ResourcePacks.ps1'
$options = @{ Model = 'RVC'; CompressionLevel = $CompressionLevel; Plan = $Plan; Force = $Force }
if ($Version) { $options.Version = $Version }
if ($OutputDirectory) { $options.OutputDirectory = $OutputDirectory }
if ($Python) { $options.Python = $Python }
& $entry @options
