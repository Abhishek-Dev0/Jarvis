@echo off
rem Start JARVIS.bat — double-click this for an interactive session: voice
rem mode, output visible in this window. (run_jarvis.bat is a DIFFERENT
rem file, used only by Launch JARVIS.vbs for a silent background start with
rem output going to a log file instead — that one looks like nothing is
rem happening if you run it directly, on purpose, because it has no console
rem window to print to when launched that way.)
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -m jarvis --voice
echo.
echo JARVIS has stopped. Press any key to close this window.
pause >nul
