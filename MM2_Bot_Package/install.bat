@echo off
echo ========================================
echo MM2 Bot - Installation (Python 3.14)
echo ========================================
echo.

echo [1/3] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.14 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [2/3] Checking Python version...
python --version | findstr /C:"Python 3.14"
if %errorlevel% neq 0 (
    echo WARNING: Python 3.14 not detected!
    echo Detected: %errorlevel%
    echo This bot requires Python 3.14 for optimal performance
    echo Continue? (y/n)
    set /p continue=
    if /i not "%continue%"=="y" exit /b 1
)

echo.
echo [3/3] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo.
echo [4/3] Verifying installation...
python -c "import sys; assert sys.version_info >= (3, 14), 'Python 3.14+ required'; import ultralytics; import cv2; import win32api; print('OK')"
if %errorlevel% neq 0 (
    echo ERROR: Verification failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure you have a model in weights/ folder
echo 2. Run MM2_Bot_Launcher.exe
echo.
pause

