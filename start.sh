#!/bin/bash
set -e

echo "🚀 Starting Restaurant Chatbot..."

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install --production

# Build frontend
echo "🔨 Building frontend..."
npm run build

# Start frontend in background
echo "🌐 Starting frontend..."
npm start &
FRONTEND_PID=$!

# Start backend
echo "🧠 Starting backend..."
cd ../backend
python services/backend_api.py &
BACKEND_PID=$!

echo "✅ Both services started!"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend PID: $BACKEND_PID"

# Wait for both processes
wait $FRONTEND_PID $BACKEND_PID