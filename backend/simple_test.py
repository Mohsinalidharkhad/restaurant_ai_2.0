#!/usr/bin/env python3
"""
Simple test script for Railway deployment
This script logs all requests and ensures proper port binding
"""

import os
import sys
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

print("🚀 Starting simple test API for Railway...")
print(f"🔧 Python version: {sys.version}")
print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Create FastAPI app
app = FastAPI(title="Simple Test API", version="1.0.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    print(f"📥 Request: {request.method} {request.url}")
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
        "message": "Simple Test API is running!",
        "status": "ok",
        "timestamp": time.time(),
        "port": os.getenv("PORT", "8000"),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "development")
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    print("✅ Health endpoint called")
    return {
        "status": "ok",
        "service": "simple-test-api",
        "timestamp": time.time()
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
    """Debug endpoint with environment info"""
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
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("PORT", "8000"))
    print(f"🔧 Using port: {port}")
    print(f"🔧 Starting uvicorn server on 0.0.0.0:{port}...")
    
    try:
        uvicorn.run(
            "simple_test:app",
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