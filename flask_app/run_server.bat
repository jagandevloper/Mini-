@echo off
REM Kidney Stone Detection Flask Server
REM This batch file runs the Flask app in a detached process

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo Starting Kidney Stone Detection Web Application...
echo Port: 5000
echo Press Ctrl+C in the spawned window to stop the server

REM Run Flask app in a new window that stays open
start "Kidney Stone Detection Server" python app.py

echo Flask server started in a new window.
timeout /t 3 /nobreak
start http://localhost:5000
