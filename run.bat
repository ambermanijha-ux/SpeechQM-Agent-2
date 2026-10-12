@echo off
rem Usage: run.bat C:\path\to\audio\folder [--language Hindi] [--prompt "..."]   (see: run.bat --help)
if not exist "%~dp0.venv\Scripts\python.exe" (echo Run setup.ps1 first & exit /b 1)
set PYTHONUTF8=1
set PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%
"%~dp0.venv\Scripts\python.exe" "%~dp0run_qc.py" %*
