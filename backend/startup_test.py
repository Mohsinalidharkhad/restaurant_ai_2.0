#!/usr/bin/env python3
"""
Ultra-minimal FastAPI startup test for Railway
This script only imports the absolute minimum to test if FastAPI starts
"""

import os
import sys
from fastapi import FastAPI
import uvicorn

print("🚀 Starting ultra-minimal FastAPI test...")
print(f"🔧 Python version: {sys.version}")
print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")

# Create the most minimal FastAPI app possible
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Ultra-minimal API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/test")
async def test():
    return {"message": "Test endpoint working"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"🔧 Using port: {port}")
    print("🔧 Starting uvicorn server...")
    
    try:
        uvicorn.run(
            "startup_test:app",
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