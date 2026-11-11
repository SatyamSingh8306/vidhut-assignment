@echo off
REM Startup script for Vision-Based Elderly Monitoring System (Windows)
REM Runs both FastAPI backend and Streamlit frontend

echo 🏥 Starting Vision-Based Elderly Monitoring System

REM Check for .env file
if not exist .env (
    echo ⚠️  .env file not found. Copying from .env.example
    copy .env.example .env
    echo 📝 Please edit .env with your configuration before running again.
    pause
    exit /b 1
)

REM Load environment variables from .env
for /f "usebackq delims==" %%a in (.env) do (
    if not "%%a"=="" if not "%%a"=="#" set "%%a"
)

REM Start FastAPI backend in background
echo 🚀 Starting FastAPI backend on http://127.0.0.1:8000
start /b python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

REM Wait for FastAPI to start
timeout /t 3 /nobreak >nul

REM Check if FastAPI is running
curl -s http://127.0.0.1:8000/api/health >nul
if errorlevel 1 (
    echo ❌ FastAPI failed to start
    pause
    exit /b 1
)
echo ✅ FastAPI backend running

REM Start Streamlit frontend
echo 🌐 Starting Streamlit frontend on http://127.0.0.1:8501
start /b streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1

echo.
echo ✨ System started successfully!
echo    📡 FastAPI API:     http://127.0.0.1:8000
echo    📚 API Docs:        http://127.0.0.1:8000/docs
echo    🖥️  Streamlit UI:   http://127.0.0.1:8501
echo.
echo Press Ctrl+C to stop all services

REM Keep window open
pause