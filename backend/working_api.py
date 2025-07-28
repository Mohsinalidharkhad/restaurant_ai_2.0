#!/usr/bin/env python3
"""
Working API for Railway deployment
This script starts a FastAPI server that serves actual content and can add full functionality
"""

import os
import sys
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

print("🚀 Starting working Restaurant Assistant API...")
print(f"🔧 Python version: {sys.version}")
print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Create FastAPI app
app = FastAPI(
    title="Restaurant Assistant API",
    description="AI-powered restaurant assistant with menu knowledge and reservation management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for initialization
system_initialized = False
initialization_in_progress = False

@app.get("/")
async def root():
    """Main page - serves actual content"""
    return {
        "message": "Restaurant Assistant API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "chat": "/api/chat (coming soon)",
            "docs": "/docs"
        },
        "timestamp": time.time()
    }

@app.get("/health")
async def health():
    """Health check for Railway"""
    return {
        "status": "ok",
        "service": "restaurant-assistant-api",
        "initialized": system_initialized,
        "timestamp": time.time()
    }

@app.get("/api/status")
async def status():
    """API status endpoint"""
    return {
        "status": "running",
        "message": "Restaurant Assistant API is ready",
        "initialized": system_initialized,
        "initialization_in_progress": initialization_in_progress,
        "timestamp": time.time()
    }

@app.get("/api/chat")
async def chat():
    """Chat endpoint - placeholder for now"""
    return {
        "message": "Chat functionality coming soon!",
        "status": "development",
        "note": "Full restaurant assistant will be available once initialization completes"
    }

@app.get("/docs")
async def docs():
    """API documentation"""
    return {
        "title": "Restaurant Assistant API",
        "version": "1.0.0",
        "description": "AI-powered restaurant assistant with menu knowledge and reservation management",
        "endpoints": {
            "GET /": "Main API information",
            "GET /health": "Health check for Railway",
            "GET /api/status": "API status and initialization progress",
            "GET /api/chat": "Chat functionality (coming soon)",
            "GET /docs": "This documentation"
        }
    }

def initialize_background():
    """Background initialization function"""
    global system_initialized, initialization_in_progress
    
    try:
        print("🔄 Starting background initialization...")
        initialization_in_progress = True
        
        # Simulate initialization time
        time.sleep(10)
        
        print("✅ Background initialization completed!")
        system_initialized = True
        initialization_in_progress = False
        
    except Exception as e:
        print(f"❌ Background initialization failed: {e}")
        initialization_in_progress = False

# Start background initialization
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"🔧 Using port: {port}")
    print("🔧 Starting uvicorn server...")
    
    # Start background initialization
    init_thread = threading.Thread(target=initialize_background, daemon=True)
    init_thread.start()
    
    try:
        uvicorn.run(
            "working_api:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 