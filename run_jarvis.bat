@echo off
cd /d "%~dp0"
if not exist "%~dp0jarvis\data\logs" mkdir "%~dp0jarvis\data\logs"
"%~dp0venv\Scripts\python.exe" -m jarvis --voice >> "%~dp0jarvis\data\logs\jarvis.log" 2>&1
