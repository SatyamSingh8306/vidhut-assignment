#!/usr/bin/env bash
# Startup script for Vision-Based Elderly Monitoring System
# Runs both FastAPI backend and Streamlit frontend

set -e

echo "🏥 Starting Vision-Based Elderly Monitoring System"

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example"
    cp .env.example .env
    echo "📝 Please edit .env with your configuration before running again."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start FastAPI backend
echo "🚀 Starting FastAPI backend on http://127.0.0.1:8000"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 3

# Check if FastAPI is running
if ! curl -s http://127.0.0.1:8000/api/health > /dev/null; then
    echo "❌ FastAPI failed to start"
    kill $FASTAPI_PID 2>/dev/null
    exit 1
fi
echo "✅ FastAPI backend running"

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend on http://127.0.0.1:8501"
streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1 &
STREAMLIT_PID=$!

echo ""
echo "✨ System started successfully!"
echo "   📡 FastAPI API:     http://127.0.0.1:8000"
echo "   📚 API Docs:        http://127.0.0.1:8000/docs"
echo "   🖥️  Streamlit UI:   http://127.0.0.1:8501"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID