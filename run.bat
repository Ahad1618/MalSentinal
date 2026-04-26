@echo off
REM MalSentinel Startup Script
REM This script sets up and runs the Streamlit application

echo ========================================
echo  MalSentinel - Malware Detection System
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [*] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [*] Starting Streamlit application...
echo [*] Open your browser to: http://localhost:8501
echo.

REM Run the Streamlit app
streamlit run main.py

pause
