# Launch API + review UI on Windows
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
}
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
