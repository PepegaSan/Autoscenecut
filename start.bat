@echo off
setlocal
REM Starts Scenecut without a console window. Videos dropped onto this file are added right away.

set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
    echo Die Python-Umgebung fehlt noch - bitte zuerst install.bat ausfuehren.
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0scenecut_app.py" %*
endlocal
