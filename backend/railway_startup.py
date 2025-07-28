#!/usr/bin/env python3
"""
Railway-specific startup script
This script handles Railway's port configuration and environment properly
"""

import os
import sys
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

print("🚀 Starting Railway-specific API...")
print(f"🔧 Python version: {sys.version}")
print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Create FastAPI app
app = FastAPI(title="Railway API", version="1.0.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    print(f"📥 Request: {request.method} {request.url}")
    print(f"📥 Client: {request.client}")
    print(f"📥 Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    print(f"📤 Response: {response.status_code} in {process_time:.3f}s")
    
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    print("✅ Root endpoint called")
    return {
        "message": "Railway API is running!",
        "status": "ok",
        "timestamp": time.time(),
        "port": os.getenv("PORT"),
        "environment": os.getenv("RAILWAY_ENVIRONMENT"),
        "project_id": os.getenv("RAILWAY_PROJECT_ID"),
        "service_id": os.getenv("RAILWAY_SERVICE_ID")
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    print("✅ Health endpoint called")
    return {
        "status": "ok",
        "service": "railway-api",
        "timestamp": time.time(),
        "port": os.getenv("PORT")
    }

@app.get("/test")
async def test():
    """Test endpoint"""
    print("✅ Test endpoint called")
    return {
        "message": "Test endpoint working!",
        "timestamp": time.time()
    }

@app.get("/debug")
async def debug():
    """Debug endpoint with full environment info"""
    print("✅ Debug endpoint called")
    return {
        "message": "Debug information",
        "environment": {
            "PORT": os.getenv("PORT"),
            "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT"),
            "RAILWAY_PROJECT_ID": os.getenv("RAILWAY_PROJECT_ID"),
            "RAILWAY_SERVICE_ID": os.getenv("RAILWAY_SERVICE_ID"),
            "PYTHON_VERSION": sys.version,
            "WORKING_DIRECTORY": os.getcwd(),
            "HOSTNAME": os.getenv("HOSTNAME"),
            "PWD": os.getenv("PWD"),
        },
        "timestamp": time.time()
    }

@app.get("/api/status")
async def status():
    """API status endpoint"""
    print("✅ Status endpoint called")
    return {
        "status": "running",
        "message": "Railway API is ready",
        "timestamp": time.time(),
        "port": os.getenv("PORT")
    }

if __name__ == "__main__":
    # Get port from Railway environment
    port = int(os.getenv("PORT", "8000"))
    print(f"🔧 Using port: {port}")
    print(f"🔧 Starting uvicorn server on 0.0.0.0:{port}...")
    
    # Log all environment variables for debugging
    print("🔧 Environment variables:")
    for key, value in os.environ.items():
        if key.startswith(('RAILWAY_', 'PORT', 'HOSTNAME', 'PWD')):
            print(f"   {key}: {value}")
    
    try:
        uvicorn.run(
            "railway_startup:app",
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