@echo off
REM ============================================================
REM  Vision Studio - one-time setup
REM  Creates a private Python environment, installs the
REM  dependencies and pre-downloads the detection models so the
REM  app works fully offline afterwards.
REM ============================================================
setlocal
cd /d "%~dp0"

echo [1/4] Creating virtual environment...
py -3.13 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo Could not create the environment. Install Python 3.10 - 3.13 from
    echo https://www.python.org/downloads/ and run setup.bat again.
    pause
    exit /b 1
)
set PY=.venv\Scripts\python.exe

echo [2/4] Upgrading pip...
"%PY%" -m pip install --upgrade pip

echo [3/4] Installing PyTorch (CPU) + Ultralytics + PySide6...
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed - see the messages above.
    pause
    exit /b 1
)

echo [4/4] Downloading detection models for offline use...
"%PY%" -c "from detector import Detector, HardwareProfile; p = HardwareProfile.detect(); [Detector(k, p) for k in ('nano', 'small')]; print('Models ready:', p.summary)"

echo.
echo Setup complete!  Double-click run.bat to start Vision Studio.
pause
endlocal
