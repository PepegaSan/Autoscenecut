@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem If you use Conda (e.g. autocut_env), use build_gui_exe_conda.bat instead —
rem otherwise "python" may be a different interpreter without torch.

rem UTF-8 so error text from PyInstaller is readable in this window
chcp 65001 >nul 2>&1

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: "python" not found in PATH.
  echo Install Python 3.10+ or run this from "Developer Command Prompt" / adjust PATH.
  echo You can also run manually:  py -3.12 -m pip install ...
  goto :fail
)

echo === Build EXE (same interpreter: python -m PyInstaller^) ===
python --version
echo.

for %%M in (tkinterdnd2 torch onnxruntime faster_whisper ctranslate2) do (
  python -c "import %%M" 2>nul
  if errorlevel 1 (
    echo ERROR: Python module missing: %%M
    echo   python -m pip install -r requirements.txt
    echo If you use multiple Pythons, use the SAME one for pip and this script, e.g.:
    echo   py -3.12 -m pip install -r requirements.txt
    echo   py -3.12 -m PyInstaller --noconfirm --clean scenecut_gui.spec
    goto :fail
  )
)

python -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo ERROR: PyInstaller not installed for this Python.
  echo   python -m pip install -r requirements-build.txt
  goto :fail
)

echo All checks OK. Building... (often 5-15+ minutes, one-file + torch is large^)
echo.

set PYTHONUNBUFFERED=1
python -u -m PyInstaller --noconfirm --clean scenecut_gui.spec
if errorlevel 1 (
  echo.
  echo BUILD FAILED — read the lines above ^(often: missing DLL / antivirus / disk space^).
  goto :fail
)

echo.
echo OK:  %~dp0dist\ScenecutNVIDIA.exe
echo.
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
