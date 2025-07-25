#!/usr/bin/env python3
"""
Minimal FastAPI startup for Railway healthcheck testing
This script starts a basic FastAPI server without any heavy imports
"""

import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Create a minimal FastAPI app
app = FastAPI(title="Restaurant Assistant API", version="1.0.0")

@app.get("/")
async def root():
    """Basic health check"""
    return {"status": "ok", "message": "Restaurant Assistant API is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "restaurant-assistant-api"}

@app.get("/api/test")
async def test():
    """Test endpoint"""
    return {"message": "API is working correctly"}

@app.get("/api/env")
async def env_check():
    """Environment check endpoint (for debugging)"""
    # Only return non-sensitive environment variables
    safe_env = {
        "PORT": os.getenv("PORT"),
        "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT"),
        "RAILWAY_PROJECT_ID": os.getenv("RAILWAY_PROJECT_ID"),
        "RAILWAY_SERVICE_ID": os.getenv("RAILWAY_SERVICE_ID"),
        "PYTHON_VERSION": sys.version,
        "PWD": os.getcwd(),
    }
    return {"environment": safe_env}

if __name__ == "__main__":
    print("🚀 Starting minimal Restaurant Assistant API...")
    print("🔧 This is a minimal version for Railway healthcheck testing")
    
    # Get port from environment
    port = int(os.getenv("PORT", "8000"))
    print(f"🔧 Using port: {port}")
    print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")
    print(f"🔧 Python version: {sys.version}")
    print(f"🔧 Working directory: {os.getcwd()}")
    
    # Start server
    uvicorn.run(
        "minimal_startup:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    ) 