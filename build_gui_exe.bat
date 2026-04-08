@echo off
setlocal
cd /d "%~dp0"

where pyinstaller >nul 2>&1
if errorlevel 1 (
  echo Install build tools first:
  echo   pip install -r requirements-build.txt
  exit /b 1
)

python -c "import tkinterdnd2" 2>nul
if errorlevel 1 (
  echo ERROR: tkinterdnd2 must be installed in the SAME Python as PyInstaller:
  echo   python -m pip install tkinterdnd2
  exit /b 1
)

pyinstaller --noconfirm --clean scenecut_gui.spec
if errorlevel 1 exit /b 1

echo.
echo Done. Executable:
echo   %~dp0dist\ScenecutNVIDIA.exe
echo.
echo Copy ScenecutNVIDIA.exe to any folder. First run creates config_nvidia.ini and output\ next to the exe.
endlocal
