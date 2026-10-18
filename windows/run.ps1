# Windows entry point. uv manages .venv from the committed lockfile.
#   windows\run.ps1 doctor
#   windows\run.ps1 profile [--smoke] [--dry-run] ...
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Uv = if ($env:UV) { $env:UV } else { 'uv' }
if (-not (Get-Command $Uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required: https://docs.astral.sh/uv/getting-started/installation/'
}

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

# --project preserves the caller's working directory and relative CLI arguments.
$Py = if ($env:PYTHON) { $env:PYTHON } else { '3.12' }
& $Uv run --locked --project $Root --group profiling --python $Py python `
    (Join-Path $Root 'core\runner\cli.py') --platform-dir (Join-Path $Root 'windows') @args
exit $LASTEXITCODE
