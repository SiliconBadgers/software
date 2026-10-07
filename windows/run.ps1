# Windows entry point. Creates .venv on first use, then runs the core runner.
#   windows\run.ps1 doctor
#   windows\run.ps1 profile [--smoke] [--dry-run] ...
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'

# Ninja + MSVC needs the developer environment. Import it only when the
# historical standalone LLVM toolchain is absent, and only into this process.
if (-not (Test-Path 'C:\Program Files\LLVM\bin\clang++.exe')) {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $VsWhere) {
        $Vs = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $DevCmd = if ($Vs) { Join-Path $Vs 'Common7\Tools\VsDevCmd.bat' } else { $null }
        if ($DevCmd -and (Test-Path $DevCmd)) {
            cmd /s /c "`"$DevCmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
                $Name, $Value = $_ -split '=', 2
                if ($Name) { Set-Item -Path "env:$Name" -Value $Value }
            }
        }
    }
}

# A toolkit installed while Codex/PowerShell is already running is not present
# in this process's PATH. Add only its bin directory, without changing the
# user's persistent environment.
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    $CudaBase = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $CudaBase) {
        $CudaBin = Get-ChildItem -LiteralPath $CudaBase -Directory |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName 'bin' } |
            Where-Object { Test-Path (Join-Path $_ 'nvcc.exe') } |
            Select-Object -First 1
        if ($CudaBin) {
            $CudaRuntime = Join-Path $CudaBin 'x64'
            $CudaPaths = if (Test-Path $CudaRuntime) { "$CudaRuntime;$CudaBin" } else { $CudaBin }
            $env:PATH = "$CudaPaths;$env:PATH"
        }
    }
}

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

& $VenvPy (Join-Path $Root 'core\runner\cli.py') --platform-dir windows @args
exit $LASTEXITCODE
