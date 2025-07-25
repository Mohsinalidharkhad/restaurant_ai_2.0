#!/usr/bin/env python3
"""
Delayed startup script for Railway deployment
This script adds a small delay to ensure proper container initialization
"""

import os
import sys
import time
import subprocess

def main():
    print("🚀 Starting delayed startup for Railway deployment...")
    print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")
    print(f"🔧 Python version: {sys.version}")
    
    # Add a small delay to ensure container is fully ready
    print("⏳ Waiting 5 seconds for container initialization...")
    time.sleep(5)
    
    # Start the main API
    print("🚀 Starting main Restaurant Assistant API...")
    
    port = os.getenv("PORT", "8000")
    print(f"🔧 Using port: {port}")
    
    # Start uvicorn with the main API
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.services.backend_api:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload", "False",
        "--log-level", "info"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main() 