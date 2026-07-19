$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Could not find executable Python at: $PythonExe`nActivate or create your venv first."
    exit 1
}

Write-Host "Starting dashboard..."
& $PythonExe -m streamlit run "src/dashboard/dashboard.py" @args
