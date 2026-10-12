@echo off
setlocal EnableDelayedExpansion
title Scenecut - Einrichtung

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"

echo.
echo ========================================
echo  Scenecut - Python-Umgebung einrichten
echo ========================================
echo.

REM --- Python 3.12 suchen (DaVinci Resolve erwartet fuer das Scripting genau diese Version) ---
set "PY_CMD="
py -3.12 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.12"
if not defined PY_CMD (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1 && set "PY_CMD=python"
)

if not defined PY_CMD (
    echo Python 3.12 wurde nicht gefunden.
    echo Es wird versucht, Python 3.12 per winget zu installieren.
    echo Dafuer kann ein Bestaetigungsfenster erscheinen - bitte zulassen.
    echo.
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 (
        echo.
        echo Python 3.12 wurde installiert. Bitte dieses Fenster schliessen
        echo und install.bat noch einmal starten, damit der neue PATH gilt.
        echo.
        pause
        exit /b 0
    )
    REM winget ging nicht - notfalls ein anderes Python nehmen
    where python >nul 2>&1 && set "PY_CMD=python"
    if not defined PY_CMD (
        echo.
        echo Automatische Installation fehlgeschlagen.
        echo Bitte Python 3.12 installieren: https://www.python.org/downloads/
        echo Beim Setup "Add python.exe to PATH" ankreuzen, dann install.bat erneut starten.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo HINWEIS: Nutze ein anderes Python als 3.12. Die App laeuft damit,
    echo "Direkt in DaVinci laden" funktioniert aber evtl. nicht.
    echo.
)

echo Python: %PY_CMD%
echo.

REM --- Virtuelle Umgebung (.venv) ---
if exist "%VENV_PY%" (
    echo Virtuelle Umgebung existiert bereits: %VENV%
) else (
    echo Erstelle virtuelle Umgebung in .venv ...
    %PY_CMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo venv konnte nicht erstellt werden.
        pause
        exit /b 1
    )
)

echo.
echo Installiere Pakete (NumPy, OpenCV, ONNX Runtime, tkinterdnd2) ...
echo.
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo.
    echo Paket-Installation fehlgeschlagen.
    pause
    exit /b 1
)

REM --- FFmpeg pruefen ---
echo.
set "MISSING="
for %%T in (ffmpeg ffprobe ffplay) do (
    where %%T >nul 2>&1 || set "MISSING=!MISSING! %%T"
)
if defined MISSING (
    echo WARNUNG: nicht im PATH gefunden:%MISSING%
    echo Scenecut braucht FFmpeg ^(ffplay fuer die Vorschau^). Installieren z. B. mit:
    echo   winget install -e --id Gyan.FFmpeg
    echo Danach ein neues Fenster oeffnen, damit der PATH gilt.
) else (
    echo FFmpeg gefunden.
)

echo.
echo ========================================
echo  Fertig - starten mit start.bat
echo ========================================
echo.
pause
endlocal
