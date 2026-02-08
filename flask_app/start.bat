# Batch file to start Flask application (Windows)

@echo off
echo 🚀 Starting Kidney Stone Detection Flask Application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 📦 Checking dependencies...
pip install -q -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "uploads" mkdir uploads
if not exist "static\results" mkdir "static\results"
if not exist "models" mkdir models

REM Check if model exists
if not exist "models\best.pt" (
    echo ⚠️  Model file not found in models\best.pt
    echo 💡 The app will look for the model in the parent directory
)

REM Start Flask application
echo 🌐 Starting Flask application...
echo 📱 Open your browser: http://localhost:5000
echo.
python app.py

pause

