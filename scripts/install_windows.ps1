param(
    [switch]$Plan,
    [switch]$SkipNukeRegistration,
    [string]$RuntimeDirectory,
    [string]$NukeDirectory,
    [string]$UvExecutable
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RuntimeDirectory) { $RuntimeDirectory = Join-Path $projectRoot '.runtime' }
if (-not $NukeDirectory) { $NukeDirectory = Join-Path $env:USERPROFILE '.nuke' }
$runtime = [IO.Path]::GetFullPath($RuntimeDirectory)
$seaRoot = Join-Path $runtime 'SEA-RAFT'
$venvRoot = Join-Path $runtime 'venv'
$python = Join-Path $venvRoot 'Scripts\python.exe'
$revision = '9137517ba24e628442aec097d3afe71d03503b75'
$configPath = Join-Path $projectRoot 'smartvector\install_config.json'

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    Write-Host ("Running: " + $Program + ' ' + ($Arguments -join ' '))
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

function Find-Uv {
    if ($UvExecutable) {
        if (-not (Test-Path -LiteralPath $UvExecutable -PathType Leaf)) {
            throw "uv executable not found: $UvExecutable"
        }
        return (Resolve-Path -LiteralPath $UvExecutable).Path
    }
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    $candidates = @(
        (Join-Path $runtime 'tools\uv.exe'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\uv.exe'),
        (Join-Path $env:USERPROFILE '.local\bin\uv.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

if ($env:OS -ne 'Windows_NT') { throw 'This installer is for native Windows.' }
if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows installation is required.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first.' }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'NVIDIA driver / nvidia-smi not found. CUDA generation requires an NVIDIA GPU.'
}
$driverText = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
if ([version]$driverText -lt [version]'527.41') {
    throw "NVIDIA driver $driverText is too old for CUDA 12.1 wheels (minimum 527.41)."
}

Write-Host "Project: $projectRoot"
Write-Host "Runtime: $runtime"
Write-Host "SEA-RAFT revision: $revision"
Write-Host "NVIDIA driver: $driverText"
Write-Host 'Python: 3.10.13; PyTorch: 2.2.0 CUDA 12.1; OpenEXR: 3.3.3'
if ($Plan) {
    Write-Host 'Plan only: no downloads, package installation, or Nuke configuration changes.'
    return
}

New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:HF_HOME = Join-Path $runtime 'huggingface'
$env:TORCH_HOME = Join-Path $runtime 'torch-cache'
$env:TEMP = Join-Path $runtime 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$uv = Find-Uv
if (-not $uv) {
    $installer = Join-Path $runtime 'install-uv.ps1'
    Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile $installer
    $env:UV_INSTALL_DIR = Join-Path $runtime 'tools'
    & $installer
    $uv = Find-Uv
    if (-not $uv) { throw 'The official uv installer finished without creating uv.exe.' }
}

if (-not (Test-Path -LiteralPath $seaRoot)) {
    Invoke-Checked 'git' @('clone', '--filter=blob:none', 'https://github.com/princeton-vl/SEA-RAFT.git', $seaRoot)
}
if (-not (Test-Path -LiteralPath (Join-Path $seaRoot '.git'))) {
    throw "SEA-RAFT directory is not a Git checkout: $seaRoot"
}
Invoke-Checked 'git' @('-C', $seaRoot, 'fetch', '--depth=1', 'origin', $revision)
Invoke-Checked 'git' @('-C', $seaRoot, 'checkout', '--detach', $revision)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Invoke-Checked $uv @('venv', '--python', '3.10.13', $venvRoot)
}
Invoke-Checked $uv @('pip', 'install', '--python', $python,
    'torch==2.2.0', 'torchvision==0.17.0', '--index-url', 'https://download.pytorch.org/whl/cu121')
Invoke-Checked $uv @('pip', 'install', '--python', $python,
    'numpy==1.26.4', 'scipy>=1.10,<2', 'huggingface_hub>=0.23,<1',
    'safetensors>=0.4.3,<1', 'OpenEXR==3.3.3')

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $projectRoot + [IO.Path]::PathSeparator + $previousPythonPath
    Invoke-Checked $python @('-m', 'smartvector.verify_runtime', '--sea-raft-root', $seaRoot)
} finally {
    $env:PYTHONPATH = $previousPythonPath
}

$settings = @{
    python_path = $python.Replace('\', '/')
    sea_raft_root = $seaRoot.Replace('\', '/')
    hf_home = $env:HF_HOME.Replace('\', '/')
    torch_home = $env:TORCH_HOME.Replace('\', '/')
} | ConvertTo-Json
$tempConfig = "$configPath.partial"
[IO.File]::WriteAllText($tempConfig, $settings + [Environment]::NewLine)
Move-Item -LiteralPath $tempConfig -Destination $configPath -Force

if (-not $SkipNukeRegistration) {
    New-Item -ItemType Directory -Force -Path $NukeDirectory | Out-Null
    $initPath = Join-Path $NukeDirectory 'init.py'
    $pluginPath = (Join-Path $projectRoot 'nuke_plugin').Replace('\', '/')
    $line = "nuke.pluginAddPath(r'$pluginPath')"
    $existing = if (Test-Path -LiteralPath $initPath) { [IO.File]::ReadAllText($initPath) } else { '' }
    if (-not $existing.Contains($line)) {
        $prefix = if ($existing.Length -gt 0 -and -not $existing.EndsWith("`n")) { "`r`n" } else { '' }
        [IO.File]::AppendAllText($initPath, "$prefix# SEA-RAFT SmartVector`r`nimport nuke`r`n$line`r`n")
    }
    Write-Host "Nuke plugin registered in $initPath"
}
Write-Host "Installed. Restart Nuke. Runtime Python: $python"
