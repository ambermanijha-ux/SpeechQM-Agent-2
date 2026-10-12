# One-time setup on Windows: Python 3.11 environment (.venv) with all packages, and Ollama.
# Usage (PowerShell):  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager) ..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Creating .venv (Python 3.11) and installing packages ..."
uv venv -p 3.11 --allow-existing .venv
if ($LASTEXITCODE -ne 0) { exit 1 }
# auto: the CUDA build of torch on machines with an NVIDIA GPU, CPU build otherwise
uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and
    -not (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Ollama ..."
        winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    } else {
        Write-Host "Install Ollama from https://ollama.com/download, start it, then run run.bat"
        exit 1
    }
}
Write-Host ""
Write-Host "Setup done. Run the checks with:  run.bat C:\path\to\audio\folder"
