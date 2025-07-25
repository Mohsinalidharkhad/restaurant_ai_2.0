#!/usr/bin/env python3
"""
FastAPI Backend Bridge for Restaurant Assistant

This file creates a REST API bridge between the Next.js frontend and the existing
Python LangGraph application. It exposes the main.py functionality through HTTP endpoints.
"""

import sys
import os
from typing import Dict, Any, Optional
import time
import uuid
from contextlib import asynccontextmanager
import concurrent.futures
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio

# DO NOT import main.py at module level - causes hanging
# Instead, import it lazily when needed
# from main import graph, ensure_system_initialized

# Request/Response Models
class InitializeRequest(BaseModel):
    thread_id: str

class ChatRequest(BaseModel):
    message: str
    config: Dict[str, Any]
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    latency: Optional[float] = None
    error: Optional[str] = None
    # Simplify user_info to avoid nested serialization issues
    user_info: Optional[Dict[str, Any]] = None
    is_registered: Optional[bool] = None
    phone_number: Optional[str] = None
    
    class Config:
        # Ensure proper JSON serialization
        json_encoders = {
            float: lambda v: round(v, 3) if v is not None else None
        }

# Global initialization state
system_initialized = False
initialization_result = None
initialization_in_progress = False
initialization_start_time = None
initialization_thread = None

# Lazy imports to avoid hanging at startup
def get_main_module():
    """Lazy import of main.py to avoid hanging during FastAPI startup"""
    try:
        # Try relative import first (when run as module)
        try:
            from . import main
            return main
        except ImportError:
            # Fall back to absolute import (when run as script)
            import sys
            import os
            # Add the project root to the path
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from backend.services import main
            return main
    except Exception as e:
        print(f"❌ Failed to import main.py: {e}")
        raise e

def run_background_initialization():
    """Run enhanced initialization in background thread with Phase 1 optimizations"""
    global system_initialized, initialization_result, initialization_in_progress, initialization_start_time
    
    try:
        print("🚀 Starting enhanced background system initialization (Phase 1)...")
        initialization_in_progress = True
        initialization_start_time = time.time()
        
        # Import and run enhanced initialization
        main = get_main_module()
        initialization_result = main.ensure_system_initialized()
        
        system_initialized = initialization_result["success"]
        initialization_in_progress = False
        
        elapsed_time = time.time() - initialization_start_time
        
        if system_initialized:
            cached_queries = initialization_result.get("cached_queries", 0)
            print(f"✅ System initialized successfully in {elapsed_time:.3f}s with {cached_queries} cached queries")
        else:
            error_msg = initialization_result.get("error", "Unknown error")
            print(f"❌ System initialization failed after {elapsed_time:.3f}s: {error_msg}")
            
    except Exception as e:
        initialization_in_progress = False
        initialization_result = {"success": False, "error": str(e)}
        print(f"❌ Background initialization failed: {e}")
        import traceback
        traceback.print_exc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    global system_initialized, initialization_result, initialization_in_progress, initialization_start_time, initialization_thread
    
    print("🚀 Starting Restaurant Assistant Backend API...")
    print("📡 This API bridges the Next.js frontend with the Python LangGraph backend")
    print("🌐 Frontend should be running on http://localhost:3000")
    print("🔧 API will be available on http://localhost:8000")
    print("-" * 60)
    
    # Start background initialization immediately
    print("🔄 Starting background system initialization...")
    initialization_thread = threading.Thread(target=run_background_initialization, daemon=True)
    initialization_thread.start()
    
    yield
    
    print("🛑 Shutting down Restaurant Assistant Backend API...")

# Create FastAPI app
app = FastAPI(
    title="Restaurant Assistant API",
    description="AI-powered restaurant assistant with menu knowledge and reservation management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint - responds immediately for Railway healthchecks"""
    global initialization_start_time
    
    # Always return a basic response for Railway healthchecks
    # This prevents deployment failures due to slow initialization
    status_info = {
        "message": "Restaurant Assistant Backend API",
        "status": "starting",  # Changed from "running" to "starting"
        "system_initialized": system_initialized,
        "initialization_in_progress": initialization_in_progress,
        "ready": system_initialized,  # Add explicit ready flag
        "timestamp": time.time()
    }
    
    # Add progress info if initialization is running
    if initialization_in_progress and initialization_start_time:
        elapsed = time.time() - initialization_start_time
        status_info["initialization_elapsed_seconds"] = round(elapsed, 1)
        status_info["initialization_status"] = "initializing..."
    elif system_initialized:
        status_info["status"] = "running"
        status_info["initialization_status"] = "completed"
        status_info["ready"] = True
    else:
        status_info["initialization_status"] = "not_started" if not initialization_result else "failed"
    
    return status_info

@app.get("/health")
async def simple_health_check():
    """Simple health check for Railway - responds immediately"""
    return {
        "status": "ok",
        "service": "restaurant-assistant-api",
        "timestamp": time.time()
    }

@app.get("/api/initialization-status")
async def get_initialization_status():
    """Get detailed initialization status for frontend polling"""
    global initialization_start_time
    
    status = {
        "initialized": system_initialized,
        "in_progress": initialization_in_progress,
        "success": system_initialized,
    }
    
    if initialization_in_progress and initialization_start_time:
        elapsed = time.time() - initialization_start_time
        status.update({
            "elapsed_seconds": round(elapsed, 1),
            "status_message": f"Initializing system... ({elapsed:.1f}s)",
            "estimated_total_seconds": 60  # Rough estimate
        })
    elif system_initialized:
        status.update({
            "status_message": "System ready!",
            "completed_at": initialization_result.get("total_time", 0) if initialization_result else None
        })
    elif initialization_result and not initialization_result.get("success"):
        status.update({
            "status_message": f"Initialization failed: {initialization_result.get('error', 'Unknown error')}",
            "error": initialization_result.get('error')
        })
    else:
        status.update({
            "status_message": "Waiting to start initialization..."
        })
    
    return status

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes messages through the LangGraph system"""
    
    global system_initialized, initialization_result
    
    print(f"[API] Received chat request: {request.message[:50]}...")
    
    # Check if system is ready
    if not system_initialized:
        if initialization_in_progress:
            return JSONResponse(
                content={
                    "success": False, 
                    "error": "System is still initializing. Please try again in a few moments.",
                    "initialization_in_progress": True
                }, 
                status_code=503
            )
        else:
            return JSONResponse(
                content={
                    "success": False, 
                    "error": "System initialization failed. Please check the logs.",
                    "initialization_failed": True
                }, 
                status_code=500
            )
    
    start_time = time.time()
    
    try:
        # Lazy import of main module
        main = get_main_module()
        from langchain_core.messages import HumanMessage
        
        message = request.message
        config = request.config
        user_info = request.user_info or {}
        
        print(f"[API] Processing message: {message[:100]}...")
        
        # Create initial state for the graph
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "is_registered": user_info.get("isRegistered", False),
            "phone_number": config.get("configurable", {}).get("phone_number"),
            "dialog_state": []
        }
        
        # Process through the graph with timeout
        print("[API] *** CALLING LANGGRAPH ***")
        
        def run_graph():
            return main.graph.invoke(initial_state, config)
        
        # Run with timeout to prevent hanging
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_graph)
            try:
                result = future.result(timeout=180)  # 3 minute timeout
                print("[API] *** LANGGRAPH COMPLETED ***")
            except concurrent.futures.TimeoutError:
                print("[API] *** LANGGRAPH TIMEOUT - CANCELLING ***")
                future.cancel()
                raise Exception("LangGraph processing timed out after 3 minutes")
        
        # Calculate latency
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"[API] Graph execution latency: {latency:.2f}s")
        
        # Extract the assistant's response
        assistant_response = "I'm here to help! How can I assist you today?"
        
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content and last_message.content.strip():
                assistant_response = last_message.content.strip()
        
        # Extract user information
        response_user_info = None
        if result.get("is_registered") is not None or result.get("phone_number"):
            response_user_info = {
                "is_registered": bool(result.get("is_registered", False)),
                "phone_number": str(result.get("phone_number")) if result.get("phone_number") else None,
            }
        
        # Create response
        response_data = {
            "success": True,
            "response": assistant_response,
            "latency": round(latency, 3),
            "user_info": response_user_info,
            "is_registered": result.get("is_registered"),
            "phone_number": result.get("phone_number")
        }
        
        return JSONResponse(content=response_data, status_code=200)
        
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"[API] Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = {"success": False, "error": f"Error processing message: {str(e)}", "latency": round(latency, 3)}
        return JSONResponse(content=error_response, status_code=500)

if __name__ == "__main__":
    print("🚀 Starting Restaurant Assistant Backend API...")
    print("📡 This API bridges the Next.js frontend with the Python LangGraph backend")
    print("🌐 Frontend should be running on http://localhost:3000")
    print("🔧 API will be available on http://localhost:8000")
    print("-" * 60)
    
    # Log environment variables for debugging (without sensitive data)
    port = os.getenv("PORT", "8000")
    print(f"🔧 Using port: {port}")
    print(f"🔧 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")
    print(f"🔧 Python version: {sys.version}")
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=int(port),
        reload=False,
        log_level="info"
    ) 