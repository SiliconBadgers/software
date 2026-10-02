# Windows entry point. Creates .venv on first use, then runs the core runner.
#   windows\run.ps1 doctor
#   windows\run.ps1 profile [--smoke] [--dry-run] ...
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'

if (-not (Test-Path $VenvPy)) {
    $Py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
    & $Py -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 or newer is required (set PYTHON to choose one)' }
    & $Py -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv' }
}

$Requirements = Join-Path $Root 'core\requirements.txt'
$Stamp = Join-Path $Venv '.sb-requirements.sha256'
$Want = (Get-FileHash $Requirements -Algorithm SHA256).Hash
$Have = if (Test-Path $Stamp) { (Get-Content $Stamp -Raw).Trim() } else { '' }
if ($Want -ne $Have) {
    & $VenvPy -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
    Set-Content -Path $Stamp -Value $Want -NoNewline
}

& $VenvPy (Join-Path $Root 'core\runner\cli.py') --platform-dir (Join-Path $Root 'windows') @args
exit $LASTEXITCODE
