@echo off
setlocal EnableExtensions
cd /d "%~dp0"

chcp 65001 >nul 2>&1

set "CONDA_ENV=autocut_env"

where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: conda not found in PATH. Open Anaconda Prompt or add condabin to PATH.
  goto :fail
)

echo === Build EXE using conda env: %CONDA_ENV% ===
echo NOTE: Use CALL conda run in .bat files — without CALL, the batch stops after the first conda.
echo.

call conda run -n %CONDA_ENV% python --version
if errorlevel 1 (
  echo ERROR: conda env "%CONDA_ENV%" not found or broken.
  echo   conda create -n %CONDA_ENV% python=3.11
  echo   conda activate %CONDA_ENV%
  echo   pip install -r requirements.txt
  echo   pip install -r requirements-build.txt
  goto :fail
)

echo.
echo Checking imports...
call conda run -n %CONDA_ENV% python "%~dp0build_check_deps.py"
if errorlevel 1 (
  echo ERROR: dependency check failed.
  goto :fail
)

echo.
echo Starting PyInstaller ^(often 5–15+ minutes^)...
echo Live log: conda run --no-capture-output streams lines here ^(otherwise the window can look frozen^).
echo.

call conda run --no-capture-output -n %CONDA_ENV% python -u -m PyInstaller --noconfirm --clean scenecut_gui.spec
if errorlevel 1 (
  echo.
  echo BUILD FAILED — read messages above.
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
