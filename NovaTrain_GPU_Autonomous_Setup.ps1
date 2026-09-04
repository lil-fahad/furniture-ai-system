#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\FurnitureAI",
    [string]$Repository = "https://github.com/lil-fahad/furniture-ai-system.git",
    [string]$Branch = "main",
    [switch]$NoAutoStart
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version Latest

function Step([string]$m) { Write-Host "[NovaTrain] $m" -ForegroundColor Cyan }
function Cmd([string]$name) { Get-Command $name -ErrorAction SilentlyContinue }
function Checked([string]$exe, [string[]]$args, [string]$cwd) {
    $p = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $cwd -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) { throw "$exe failed with exit code $($p.ExitCode): $($args -join ' ')" }
}
function Ensure([string]$name, [string]$wingetId) {
    $c = Cmd $name; if ($c) { return $c.Source }
    if (-not (Cmd winget)) { throw "$name is required and winget is unavailable." }
    Step "Installing $name"
    & winget install --id $wingetId --exact --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Could not install $name." }
    $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
    $c = Cmd $name; if (-not $c) { throw "$name installed but is not visible in PATH. Sign out/in and rerun." }
    return $c.Source
}
function HasNvidia() {
    $n = Cmd nvidia-smi; if (-not $n) { return $false }
    & $n.Source --query-gpu=name --format=csv,noheader 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Step "Preparing directories"
$Repo = Join-Path $InstallRoot "furniture-ai-system"
$State = Join-Path $InstallRoot "novatrain-state"
$Inbox = Join-Path $State "inbox"
$Outbox = Join-Path $State "outbox"
$Logs = Join-Path $State "logs"
$Venv = Join-Path $InstallRoot ".venv"
New-Item -ItemType Directory -Force -Path $InstallRoot,$State,$Inbox,$Outbox,$Logs | Out-Null

$Git = Ensure "git" "Git.Git"
$Python = $null
if (Cmd py) {
    & (Cmd py).Source -3.12 -c "import sys; assert sys.version_info >= (3,11)" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = @((Cmd py).Source,"-3.12") }
}
if (-not $Python -and (Cmd python)) {
    & (Cmd python).Source -c "import sys; assert sys.version_info >= (3,11)" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = @((Cmd python).Source) }
}
if (-not $Python) {
    $p = Ensure "python" "Python.Python.3.12"
    $Python = @($p)
}

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    Step "Cloning FurnitureAI"
    Checked $Git @("clone","--branch",$Branch,"--single-branch",$Repository,$Repo) $InstallRoot
} else {
    Step "Fast-forwarding FurnitureAI (--ff-only)"
    Checked $Git @("-C",$Repo,"fetch","origin",$Branch) $InstallRoot
    Checked $Git @("-C",$Repo,"checkout",$Branch) $InstallRoot
    Checked $Git @("-C",$Repo,"pull","--ff-only","origin",$Branch) $InstallRoot
}

$Vpy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Vpy)) {
    Step "Creating isolated Python environment"
    $args = @(); if ($Python.Count -gt 1) { $args += $Python[1..($Python.Count-1)] }; $args += @("-m","venv",$Venv)
    Checked $Python[0] $args $InstallRoot
}
Checked $Vpy @("-m","pip","install","--upgrade","pip","setuptools","wheel") $InstallRoot

$Nvidia = HasNvidia
if ($Nvidia) {
    Step "NVIDIA GPU detected; installing a CUDA-enabled PyTorch build"
    $installed = $false
    foreach ($flavor in @("cu130","cu128","cu126")) {
        $p = Start-Process -FilePath $Vpy -ArgumentList @("-m","pip","install","--upgrade","torch","torchvision","--index-url","https://download.pytorch.org/whl/$flavor") -Wait -PassThru -NoNewWindow
        if ($p.ExitCode -eq 0) {
            & $Vpy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 9)"
            if ($LASTEXITCODE -eq 0) { $installed = $true; break }
        }
    }
    if (-not $installed) { throw "NVIDIA is present but no tested CUDA PyTorch wheel became usable. Refusing silent CPU fallback." }
} else {
    Step "No NVIDIA GPU detected; installing portable PyTorch"
    Checked $Vpy @("-m","pip","install","--upgrade","torch","torchvision") $InstallRoot
}

Step "Installing FurnitureAI training dependencies"
$p = Start-Process -FilePath $Vpy -ArgumentList @("-m","pip","install","-e",".[training]") -WorkingDirectory $Repo -Wait -PassThru -NoNewWindow
if ($p.ExitCode -ne 0) {
    Checked $Vpy @("-m","pip","install","-e",".") $Repo
    Checked $Vpy @("-m","pip","install","timm","scikit-learn","pillow","numpy","psutil") $Repo
}

Step "Verifying accelerator visibility"
$env:NOVATRAIN_EXPECT_NVIDIA = if ($Nvidia) { "1" } else { "0" }
& $Vpy -c "import os,torch,sys,json; d={'torch':torch.__version__,'cuda':torch.cuda.is_available(),'cuda_version':torch.version.cuda,'gpu_count':torch.cuda.device_count() if torch.cuda.is_available() else 0,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}; print(json.dumps(d,indent=2)); sys.exit(3 if os.getenv('NOVATRAIN_EXPECT_NVIDIA')=='1' and not torch.cuda.is_available() else 0)"
if ($LASTEXITCODE -ne 0) { throw "GPU verification failed." }

$TrackedConfig = Join-Path $Repo "training\local_training_jobs.json"
$LocalConfig = Join-Path $State "local_training_jobs.json"
if (-not (Test-Path $TrackedConfig)) { throw "Missing training/local_training_jobs.json" }
$cfg = Get-Content $TrackedConfig -Raw | ConvertFrom-Json
$cfg.sync.enabled = $true
$cfg.sync.remote = "origin"
$cfg.sync.branch = $Branch
$cfg.sync.interval_seconds = 300
$cfg.shutdown_grace_seconds = 60
$cfg | ConvertTo-Json -Depth 30 | Set-Content -Path $LocalConfig -Encoding UTF8

Step "Validating trusted training job manifest"
& $Vpy -c "from pathlib import Path; from training.local_worker import load_config; load_config(Path(r'$LocalConfig')); print('worker_config_ok')"
if ($LASTEXITCODE -ne 0) { throw "Training worker configuration is invalid." }

$Runner = Join-Path $State "run_novatrain_worker.ps1"
$allowCpu = if ($Nvidia) { '$false' } else { '$true' }
$runnerText = @'
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$repo = "__REPO__"
$python = "__PYTHON__"
$config = "__CONFIG__"
$log = "__LOG__"
while ($true) {
  & $python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() or __ALLOW_CPU__) else 9)"
  if ($LASTEXITCODE -ne 0) { Add-Content $log "GPU unavailable; retry in 60s"; Start-Sleep 60; continue }
  & $python -m training.local_worker --repo $repo --config $config *>> $log
  $code = $LASTEXITCODE
  if ($code -eq 76) { Start-Sleep 2 } elseif ($code -eq 0) { Start-Sleep 10 } else { Add-Content $log "worker exit=$code; retry in 30s"; Start-Sleep 30 }
}
'@
$runnerText = $runnerText.Replace('__REPO__',$Repo.Replace('"','`"')).Replace('__PYTHON__',$Vpy.Replace('"','`"')).Replace('__CONFIG__',$LocalConfig.Replace('"','`"')).Replace('__LOG__',(Join-Path $Logs 'worker.log').Replace('"','`"')).Replace('__ALLOW_CPU__',$allowCpu)
Set-Content -Path $Runner -Value $runnerText -Encoding UTF8

# Receive channel: GitHub main supplies the allowlisted training manifest/config and code via ff-only pulls.
# Local/AegisData training material can be placed in the inbox; heavy datasets/checkpoints are never auto-pushed to Git.
$status = [ordered]@{ installed_at=(Get-Date).ToUniversalTime().ToString('o'); computer=$env:COMPUTERNAME; repo=$Repo; state=$State; inbox=$Inbox; outbox=$Outbox; nvidia=$Nvidia; branch=$Branch }
$status | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $Outbox "machine_status.json") -Encoding UTF8

$Nova = Join-Path $Repo "tools\novatrain\novatrain_ai.py"
if (Test-Path $Nova) {
    Step "Running Nova Doctor"
    & $Vpy $Nova doctor
    if ($LASTEXITCODE -eq 2) { throw "Nova Doctor reported FAIL." }
}

$Task = "NovaTrain-FurnitureAI-AutonomousTrainer"
if (-not $NoAutoStart) {
    Step "Registering automatic startup"
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Runner`"" -WorkingDirectory $Repo
    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    try { Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction SilentlyContinue } catch {}
    Register-ScheduledTask -TaskName $Task -Action $Action -Trigger $Trigger -Settings $Settings -Description "FurnitureAI NovaTrain autonomous training; checkpoint resume and GitHub receive" | Out-Null
    Start-ScheduledTask -TaskName $Task
}

Write-Host ""
Write-Host "NovaTrain setup complete." -ForegroundColor Green
Write-Host "Repository : $Repo"
Write-Host "Data Inbox : $Inbox"
Write-Host "Outbox     : $Outbox"
Write-Host "Logs       : $Logs"
Write-Host "Task       : $Task"
Write-Host "The worker starts automatically at Windows logon and resumes from worker-managed checkpoints." -ForegroundColor Green
