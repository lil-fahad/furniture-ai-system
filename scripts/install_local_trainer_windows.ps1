[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$TaskName = "FurnitureAI-LocalTrainer"
)

$ErrorActionPreference = "Stop"
$NetworkServiceSid = New-Object Security.Principal.SecurityIdentifier("S-1-5-20")
$WorkerAccount = $NetworkServiceSid.Translate([Security.Principal.NTAccount]).Value
$WorkerAclSid = "*S-1-5-20"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$admin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    throw "Run this installer once from PowerShell as Administrator."
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
$Venv = Join-Path $RepoRoot ".trainer-venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $Venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $Venv
    }
    else {
        throw "Python 3.11+ is required. Install Python, then rerun this installer."
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -e "$RepoRoot[training]"

# The persistent worker must not run as SYSTEM/Administrator. Code and the
# virtual environment are read/execute only; only runtime/data/output trees are writable.
$WritableRoots = @(
    (Join-Path $RepoRoot ".furnitureai-local"),
    (Join-Path $RepoRoot "models"),
    (Join-Path $RepoRoot "data")
)
foreach ($path in $WritableRoots) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}
& icacls.exe $RepoRoot /grant:r "${WorkerAclSid}:(OI)(CI)RX" /T /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Could not grant the restricted training account read access to $RepoRoot."
}
foreach ($path in $WritableRoots) {
    & icacls.exe $path /grant:r "${WorkerAclSid}:(OI)(CI)M" /T /C | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not grant the restricted training account write access to $path."
    }
}

$Arguments = "-m training.local_worker --repo `"$RepoRoot`""
$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument $Arguments `
    -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -AtStartup
$TaskPrincipal = New-ScheduledTaskPrincipal `
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
    -Principal $TaskPrincipal `
    -Settings $Settings `
    -Description "FurnitureAI autonomous local model training worker (restricted account)"

Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Host "Installed and started $TaskName"
Write-Host "Worker account: $WorkerAccount (least-privilege service account)"
Write-Host "Worker state: $RepoRoot\.furnitureai-local\state.json"
Write-Host "Training logs: $RepoRoot\.furnitureai-local\logs\"
Write-Host "It will start automatically on every Windows boot."
Write-Host "Automatic GitHub code synchronization is disabled by the committed queue."
