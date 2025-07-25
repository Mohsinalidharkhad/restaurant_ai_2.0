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
            print(f"✅ Enhanced background initialization completed in {elapsed_time:.2f}s!")
            print(f"💡 {cached_queries} queries pre-warmed for instant responses")
            print(f"⚡ Expected performance: Cold start < 2s, Warm start < 1s")
        else:
            print(f"❌ Background initialization failed after {elapsed_time:.2f}s: {initialization_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Background initialization exception: {e}")
        import traceback
        traceback.print_exc()
        
        system_initialized = False
        initialization_in_progress = False
        initialization_result = {"success": False, "error": str(e)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the system on startup with Phase 1 optimizations"""
    global system_initialized, initialization_result, initialization_thread
    
    print("🚀 Starting FastAPI backend bridge...")
    print("⚡ Starting enhanced background system initialization (Phase 1) for optimal UX...")
    
    # Start enhanced initialization in background thread immediately
    initialization_thread = threading.Thread(target=run_background_initialization, daemon=True)
    initialization_thread.start()
    
    print("✅ FastAPI backend started! Enhanced initialization running in background...")
    print("💡 Pre-warming connections, models, and common queries for instant responses...")
    
    yield
    
    print("🛑 Shutting down FastAPI backend bridge...")

# Create FastAPI app
app = FastAPI(
    title="Restaurant Assistant API",
    description="REST API bridge for the Restaurant Assistant chat application",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Next.js frontend
# Get allowed origins from environment variable or use defaults
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    global initialization_start_time
    
    status_info = {
        "message": "Restaurant Assistant Backend API",
        "status": "running",
        "system_initialized": system_initialized,
        "initialization_in_progress": initialization_in_progress
    }
    
    # Add progress info if initialization is running
    if initialization_in_progress and initialization_start_time:
        elapsed = time.time() - initialization_start_time
        status_info["initialization_elapsed_seconds"] = round(elapsed, 1)
        status_info["initialization_status"] = "initializing..."
    elif system_initialized:
        status_info["initialization_status"] = "completed"
    else:
        status_info["initialization_status"] = "failed" if initialization_result else "not_started"
    
    return status_info

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

@app.post("/api/initialize")
async def initialize_system(request: InitializeRequest) -> Dict[str, Any]:
    """Initialize the system for a specific thread (or return current status)"""
    global system_initialized, initialization_result
    
    print(f"[API] Initialize request received for thread: {request.thread_id}")
    
    if system_initialized:
        print("[API] System already initialized")
        return {
            "success": True,
            "thread_id": request.thread_id,
            "message": "System already initialized and ready"
        }
    elif initialization_in_progress:
        print("[API] Initialization in progress, waiting...")
        # Wait for background initialization to complete
        if initialization_thread and initialization_thread.is_alive():
            initialization_thread.join(timeout=120)  # Wait up to 2 minutes
        
        if system_initialized:
            return {
                "success": True,
                "thread_id": request.thread_id,
                "message": "System initialized and ready",
                "initialization_time": initialization_result.get("total_time", 0) if initialization_result else None
            }
        else:
            return {
                "success": False,
                "error": f"Background initialization failed: {initialization_result.get('error', 'Unknown error') if initialization_result else 'Timeout'}",
                "details": initialization_result
            }
    else:
        # Manual initialization if background failed or didn't start
        print("[API] Starting manual initialization...")
        try:
            def run_initialization():
                main = get_main_module()
                return main.ensure_system_initialized()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_initialization)
                initialization_result = future.result(timeout=120)
            
            system_initialized = initialization_result["success"]
            
            if system_initialized:
                print("[API] ✅ Manual initialization successful!")
                return {
                    "success": True,
                    "thread_id": request.thread_id,
                    "message": "System initialized and ready",
                    "initialization_time": initialization_result.get("total_time", 0)
                }
            else:
                print(f"[API] ⚠️ Manual initialization failed: {initialization_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": f"System initialization failed: {initialization_result.get('error', 'Unknown error')}",
                    "details": initialization_result
                }
                
        except Exception as e:
            print(f"[API] ❌ Exception during manual initialization: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Initialization exception: {str(e)}",
                "details": str(e)
            }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes messages through the LangGraph system"""
    
    global system_initialized, initialization_result
    
    print(f"[API] Received chat request: {request.message[:50]}...")
    
    # Wait for initialization to complete if it's in progress
    if initialization_in_progress and not system_initialized:
        print("[API] Waiting for background initialization to complete...")
        if initialization_thread and initialization_thread.is_alive():
            initialization_thread.join(timeout=120)  # Wait up to 2 minutes
    
    # If still not initialized, try manual initialization
    if not system_initialized:
        print("[API] System not initialized, starting manual initialization...")
        try:
            def run_initialization():
                main = get_main_module()
                return main.ensure_system_initialized()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_initialization)
                initialization_result = future.result(timeout=120)
            
            system_initialized = initialization_result["success"]
            
            if not system_initialized:
                print(f"[API] Manual initialization failed: {initialization_result.get('error', 'Unknown error')}")
                error_response = {"success": False, "error": f"System initialization failed: {initialization_result.get('error', 'Unknown error')}"}
                return JSONResponse(content=error_response, status_code=500)
            else:
                print("[API] Manual initialization successful!")
                
        except Exception as e:
            print(f"[API] Manual initialization exception: {e}")
            import traceback
            traceback.print_exc()
            error_response = {"success": False, "error": f"System initialization failed: {str(e)}"}
            return JSONResponse(content=error_response, status_code=500)
    
    start_time = time.time()
    
    try:
        # Lazy import of main module
        main = get_main_module()
        from langchain_core.messages import HumanMessage
        
        message = request.message
        config = request.config
        user_info = request.user_info or {}
        
        print(f"[API] Processing message: {message[:100]}...")
        print(f"[API] Config: {config}")
        print(f"[API] User info: {user_info}")
        
        # Create initial state for the graph
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "is_registered": user_info.get("isRegistered", False),
            "phone_number": config.get("configurable", {}).get("phone_number"),
            "dialog_state": []
        }
        
        print(f"[API] Initial state: {initial_state}")
        
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
        print(f"[API] Graph result type: {type(result)}")
        print(f"[API] Graph result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # CRITICAL: Add detailed logging here
        try:
            print("[API] *** STARTING RESPONSE EXTRACTION ***")
            
            # Extract the assistant's response with better error handling
            assistant_response = "I'm here to help! How can I assist you today?"
            
            if result.get("messages"):
                print(f"[API] Found {len(result['messages'])} messages in result")
                last_message = result["messages"][-1]
                print(f"[API] Last message type: {type(last_message)}")
                
                if hasattr(last_message, 'content'):
                    print(f"[API] Last message content length: {len(last_message.content) if last_message.content else 0}")
                    if last_message.content and last_message.content.strip():
                        assistant_response = last_message.content.strip()
                        print("[API] *** RESPONSE EXTRACTED SUCCESSFULLY ***")
                    else:
                        print("[API] Warning: Last message has empty content")
                else:
                    print("[API] Warning: Last message has no content attribute")
            else:
                print("[API] Warning: No messages found in result")
                
            print("[API] *** RESPONSE EXTRACTION COMPLETED ***")
            
        except Exception as response_error:
            print(f"[API] ERROR during response extraction: {response_error}")
            import traceback
            traceback.print_exc()
            # Continue with default response
        
        # Extract user information with better error handling
        response_user_info = None
        try:
            print("[API] *** STARTING USER INFO EXTRACTION ***")
            if result.get("is_registered") is not None or result.get("phone_number"):
                # Ensure user info values are serializable
                is_reg = result.get("is_registered", False)
                phone_num = result.get("phone_number")
                
                # Convert to simple types
                if not isinstance(is_reg, (bool, type(None))):
                    is_reg = bool(is_reg)
                if phone_num is not None and not isinstance(phone_num, str):
                    phone_num = str(phone_num)
                
                response_user_info = {
                    "is_registered": is_reg,
                    "phone_number": phone_num,
                }
                print(f"[API] Extracted user info: {response_user_info}")
            print("[API] *** USER INFO EXTRACTION COMPLETED ***")
        except Exception as user_info_error:
            print(f"[API] ERROR during user info extraction: {user_info_error}")
            import traceback
            traceback.print_exc()
            # Continue without user info
            response_user_info = None
        
        print(f"[API] Final response length: {len(assistant_response)}")
        print(f"[API] Final response: {assistant_response[:200]}... (Latency: {latency:.2f}s)")
        
        # Create response as dict and return as JSONResponse
        print("[API] *** CREATING RESPONSE DATA ***")
        
        # Ensure all values are JSON serializable
        try:
            is_registered_value = result.get("is_registered")
            phone_number_value = result.get("phone_number")
            
            # Convert complex objects to simple types
            if is_registered_value is not None and not isinstance(is_registered_value, (bool, str, int, float, type(None))):
                print(f"[API] Warning: is_registered is complex type {type(is_registered_value)}, converting to bool")
                is_registered_value = bool(is_registered_value)
            
            if phone_number_value is not None and not isinstance(phone_number_value, (str, type(None))):
                print(f"[API] Warning: phone_number is complex type {type(phone_number_value)}, converting to string")
                phone_number_value = str(phone_number_value) if phone_number_value else None
            
            response_data = {
                "success": True,
                "response": assistant_response,
                "latency": round(latency, 3),
                "user_info": response_user_info,
                "is_registered": is_registered_value,
                "phone_number": phone_number_value
            }
            print("[API] *** RESPONSE DATA CREATED ***")
            
            # Ensure response text is valid
            if not response_data["response"] or len(response_data["response"].strip()) == 0:
                print("[API] Warning: Empty response, using fallback")
                response_data["response"] = "I'm here to help! How can I assist you today?"
            
            print(f"[API] *** CREATING JSON RESPONSE ***")
            print(f"[API] Response data keys: {list(response_data.keys())}")
            print(f"[API] Response data types: {[(k, type(v)) for k, v in response_data.items()]}")
            
            # Test JSON serialization before creating response
            import json
            json_test = json.dumps(response_data)
            print(f"[API] JSON serialization test passed ({len(json_test)} chars)")
            
            json_response = JSONResponse(content=response_data, status_code=200)
            print(f"[API] *** JSON RESPONSE CREATED SUCCESSFULLY ***")
            print(f"[API] *** RETURNING RESPONSE TO CLIENT ***")
            
            return json_response
            
        except Exception as serialization_error:
            print(f"[API] JSON serialization error: {serialization_error}")
            import traceback
            traceback.print_exc()
            
            # Fallback response with minimal data
            fallback_response = {
                "success": True,
                "response": assistant_response,
                "latency": round(latency, 3),
                "user_info": None,
                "is_registered": None,
                "phone_number": None
            }
            print("[API] Using fallback response due to serialization error")
            return JSONResponse(content=fallback_response, status_code=200)
        
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"[API] Error processing message: {e}")
        print(f"[API] Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        error_response = {"success": False, "error": f"Error processing message: {str(e)}", "latency": round(latency, 3)}
        return JSONResponse(content=error_response, status_code=500)

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "system_initialized": system_initialized,
        "initialization_in_progress": initialization_in_progress,
        "initialization_result": initialization_result,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print("🚀 Starting Restaurant Assistant Backend API...")
    print("📡 This API bridges the Next.js frontend with the Python LangGraph backend")
    print("🌐 Frontend should be running on http://localhost:3000")
    print("🔧 API will be available on http://localhost:8000")
    print("-" * 60)
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 