@echo off
cd /d "%~dp0"
if not exist "app\settings.json" (
    python configure_windows.py
    if errorlevel 1 goto end
)
python launch.py start
:end
pause
