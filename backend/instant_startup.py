#!/usr/bin/env python3
"""
Instant startup script for Railway deployment
This script starts a minimal FastAPI server immediately for healthchecks
"""

import os
import sys
from fastapi import FastAPI
import uvicorn

print("🚀 Starting instant FastAPI server for Railway...")
print(f"🔧 Python version: {sys.version}")
print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Create minimal FastAPI app
app = FastAPI(title="Restaurant Assistant API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Restaurant Assistant API is running"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "restaurant-assistant-api"}

@app.get("/test")
async def test():
    return {"message": "Test endpoint working"}

@app.get("/api/status")
async def status():
    return {
        "status": "running",
        "message": "API is ready for requests",
        "timestamp": __import__("time").time()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"🔧 Using port: {port}")
    print("🔧 Starting uvicorn server immediately...")
    
    try:
        uvicorn.run(
            "instant_startup:app",
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