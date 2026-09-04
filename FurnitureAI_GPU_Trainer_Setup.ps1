[CmdletBinding()]
param(
    [string]$InstallRoot = "$env:ProgramData\FurnitureAI\LocalTrainer",
    [string]$TaskName = "FurnitureAI-LocalTrainer"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$RepoUrl = "https://github.com/lil-fahad/furniture-ai-system.git"
$RepoRoot = Join-Path $InstallRoot "repo"
$VenvRoot = Join-Path $InstallRoot "venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"
$NetworkServiceSid = New-Object Security.Principal.SecurityIdentifier("S-1-5-20")
$WorkerAccount = $NetworkServiceSid.Translate([Security.Principal.NTAccount]).Value
$WorkerAclSid = "*S-1-5-20"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedSelf {
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $PSCommandPath),
        "-InstallRoot", ('"{0}"' -f $InstallRoot),
        "-TaskName", ('"{0}"' -f $TaskName)
    )
    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments | Out-Null
}

function Refresh-ProcessPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-WithWinget([string]$Id) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Missing required software and winget is unavailable. Install Git and Python 3.11+ once, then rerun this file."
    }
    & $winget.Source install --id $Id --exact --silent --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed while installing $Id (exit $LASTEXITCODE)."
    }
    Refresh-ProcessPath
}

function Resolve-PythonLauncher {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3.12 -c "import sys; assert sys.version_info >= (3,11)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @($py.Source, "-3.12") }
        & $py.Source -3.11 -c "import sys; assert sys.version_info >= (3,11)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @($py.Source, "-3.11") }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import sys; assert sys.version_info >= (3,11)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @($python.Source) }
    }
    return $null
}

if (-not (Test-Administrator)) {
    Write-Host "Requesting Administrator permission for one-time installation..."
    Invoke-ElevatedSelf
    exit 0
}

Write-Host "=== FurnitureAI GPU Local Trainer Setup ==="
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Git..."
    Install-WithWinget "Git.Git"
}
$GitExe = (Get-Command git.exe -ErrorAction Stop).Source

$launcher = Resolve-PythonLauncher
if (-not $launcher) {
    Write-Host "Installing Python 3.12..."
    Install-WithWinget "Python.Python.3.12"
    $launcher = Resolve-PythonLauncher
}
if (-not $launcher) {
    throw "Python 3.11+ could not be located after installation."
}

Write-Host "Checking NVIDIA GPU and driver..."
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw "NVIDIA GPU driver was not detected (nvidia-smi.exe missing). Install/update the NVIDIA driver, then rerun this file."
}
& $nvidiaSmi.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi could not access the GPU. Fix the NVIDIA driver before training."
}

# Running this setup is the explicit administrative approval point for code
# installation/update. The persistent worker does not self-update by default.
if (Test-Path (Join-Path $RepoRoot ".git")) {
    Write-Host "Updating the explicitly installed FurnitureAI checkout..."
    & $GitExe -C $RepoRoot fetch origin main
    if ($LASTEXITCODE -ne 0) { throw "GitHub fetch failed." }
    & $GitExe -C $RepoRoot checkout main
    if ($LASTEXITCODE -ne 0) { throw "Could not checkout main." }
    & $GitExe -C $RepoRoot pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) { throw "Could not fast-forward the local main branch." }
}
else {
    if (Test-Path $RepoRoot) { Remove-Item -Recurse -Force $RepoRoot }
    Write-Host "Downloading FurnitureAI training code from GitHub..."
    & $GitExe clone --branch main --single-branch $RepoUrl $RepoRoot
    if ($LASTEXITCODE -ne 0) { throw "Git clone failed." }
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating isolated training environment..."
    if ($launcher.Count -eq 2) {
        & $launcher[0] $launcher[1] -m venv $VenvRoot
    }
    else {
        & $launcher[0] -m venv $VenvRoot
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create Python virtual environment." }
}

Write-Host "Installing/updating FurnitureAI training dependencies..."
& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $PythonExe -m pip install -e "$RepoRoot[training]"
if ($LASTEXITCODE -ne 0) { throw "FurnitureAI training dependency installation failed." }

Write-Host "Verifying PyTorch CUDA access..."
$gpuCheck = @'
import json
import sys
import torch
ok = bool(torch.cuda.is_available())
payload = {
    "cuda_available": ok,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if ok else None,
    "gpu_count": torch.cuda.device_count() if ok else 0,
}
print(json.dumps(payload))
sys.exit(0 if ok else 3)
'@
$gpuJson = & $PythonExe -c $gpuCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host $gpuJson
    throw "PyTorch cannot use CUDA on this computer. Update the NVIDIA driver or install a CUDA-enabled PyTorch build, then rerun this setup."
}
Write-Host "GPU READY for the installation account: $gpuJson"

# The persistent task runs under Network Service, never SYSTEM/Highest. Give
# that restricted account modify access only to the dedicated trainer tree.
& icacls.exe $InstallRoot /grant "${WorkerAclSid}:(OI)(CI)M" /T /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Could not grant the restricted training account access to $InstallRoot."
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument ('-m training.local_worker --repo "{0}"' -f $RepoRoot) `
    -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal `
    -UserId $WorkerAccount `
    -LogonType ServiceAccount `
    -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
$Task = New-ScheduledTask `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "FurnitureAI GPU-only local model training worker (restricted account)"
Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null

try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch {}
Start-ScheduledTask -TaskName $TaskName

Write-Host ""
Write-Host "=== READY ==="
Write-Host "GPU training worker installed and started."
Write-Host "Worker account: $WorkerAccount (restricted service account)"
Write-Host "Repository: $RepoRoot"
Write-Host "Datasets expected under: $RepoRoot\data\"
Write-Host "Worker state: $RepoRoot\.furnitureai-local\state.json"
Write-Host "Training logs: $RepoRoot\.furnitureai-local\logs\"
Write-Host "Checkpoints: $RepoRoot\.furnitureai-local\checkpoints\"
Write-Host "The worker starts automatically on every Windows boot and resumes from the latest valid checkpoint."
Write-Host "It only runs allow-listed FurnitureAI model-training tasks."
Write-Host "Automatic GitHub code synchronization is disabled by default. Rerun this setup to explicitly update code."
Write-Host "If a Windows service session cannot access your NVIDIA GPU, do not switch to SYSTEM; use a dedicated restricted training account or the Linux systemd installation."
