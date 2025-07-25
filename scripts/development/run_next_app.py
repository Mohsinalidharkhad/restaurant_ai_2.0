#!/usr/bin/env python3
"""
Next.js Restaurant Assistant Launcher

Script to launch the Next.js-based Restaurant Assistant with the FastAPI backend bridge.
"""

import sys
import subprocess
import os
import time
import signal
import socket
import threading
from pathlib import Path
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'

def is_port_in_use(port):
    """Check if a port is currently in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def kill_processes_on_port(port):
    """Kill any processes running on the specified port"""
    try:
        import psutil
        
        killed_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr.port == port:
                        print(f"🔄 Found process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                        proc.terminate()
                        killed_processes.append(proc.info['pid'])
                        
                        try:
                            proc.wait(timeout=3)
                            print(f"✅ Gracefully terminated process {proc.info['pid']}")
                        except psutil.TimeoutExpired:
                            proc.kill()
                            print(f"⚡ Force killed process {proc.info['pid']}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        if killed_processes:
            print(f"🗑️  Killed {len(killed_processes)} process(es) on port {port}")
            time.sleep(1)
        
        return len(killed_processes)
        
    except ImportError:
        print("⚠️  psutil not installed, trying alternative method...")
        return 0

def ensure_port_available(port, description):
    """Ensure the specified port is available"""
    print(f"🔍 Checking if port {port} is available for {description}...")
    
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is currently in use")
        killed_count = kill_processes_on_port(port)
        
        time.sleep(1)
        if is_port_in_use(port):
            print(f"❌ Port {port} is still in use after attempting to kill processes")
            return False
        else:
            print(f"✅ Port {port} is now available")
            return True
    else:
        print(f"✅ Port {port} is available")
        return True

def check_requirements():
    """Check if basic requirements are met"""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("📝 Please create a .env file with your API keys and database credentials.")
        return False
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ Node.js found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found. Please install Node.js first.")
        print("🌐 Download from: https://nodejs.org/")
        return False
    
    # Check if package.json exists in frontend directory
    if not os.path.exists('frontend/package.json'):
        print("❌ frontend/package.json not found. Please ensure the frontend is properly set up.")
        return False
    
    # Check if node_modules exists in frontend directory
    if not os.path.exists('frontend/node_modules'):
        print("📦 Frontend node_modules not found. Installing dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd='frontend', check=True)
            print("✅ Frontend dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install frontend dependencies: {e}")
            return False
    
    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import langchain
        import supabase
        import neo4j
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def get_timestamp():
    """Get current timestamp for log prefixes"""
    return datetime.now().strftime("%H:%M:%S")

def stream_process_output(process, name, color, startup_check_func=None):
    """Stream process output with color coding and timestamps"""
    startup_detected = False
    
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                timestamp = get_timestamp()
                line_clean = line.strip()
                
                # Check for startup completion
                if not startup_detected and startup_check_func and startup_check_func(line_clean):
                    startup_detected = True
                    print(f"{color}[{timestamp}] [{name}] {line_clean}{Colors.RESET}")
                    print(f"✅ {name} started successfully!")
                else:
                    print(f"{color}[{timestamp}] [{name}] {line_clean}{Colors.RESET}")
    except Exception as e:
        print(f"❌ Error streaming {name} output: {e}")
    finally:
        print(f"{Colors.YELLOW}[{get_timestamp()}] [{name}] Process output stream ended{Colors.RESET}")

def start_backend():
    """Start the FastAPI backend bridge with log streaming"""
    print("🚀 Starting FastAPI backend bridge...")
    
    try:
        process = subprocess.Popen([
            sys.executable, 'backend/services/backend_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Define startup check function
        def backend_startup_check(line):
            return "Application startup complete" in line
        
        # Start log streaming in a separate thread
        log_thread = threading.Thread(
            target=stream_process_output,
            args=(process, "Backend", Colors.BLUE, backend_startup_check),
            daemon=True
        )
        log_thread.start()
        
        # Wait a moment for startup to be detected
        time.sleep(2)
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Next.js frontend with log streaming"""
    print("🌐 Starting Next.js frontend...")
    
    try:
        process = subprocess.Popen([
            'npm', 'run', 'dev'
        ], cwd='frontend', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Define startup check function
        def frontend_startup_check(line):
            return "Ready in" in line or "Local:" in line
        
        # Start log streaming in a separate thread
        log_thread = threading.Thread(
            target=stream_process_output,
            args=(process, "Frontend", Colors.GREEN, frontend_startup_check),
            daemon=True
        )
        log_thread.start()
        
        # Wait a moment for startup to be detected
        time.sleep(2)
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main launcher function"""
    print("🍽️  Restaurant Assistant - Next.js Edition")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        return
    
    # Ensure ports are available
    if not ensure_port_available(8000, "FastAPI backend"):
        print("❌ Cannot start backend on port 8000. Exiting.")
        return
    
    if not ensure_port_available(3000, "Next.js frontend"):
        print("❌ Cannot start frontend on port 3000. Exiting.")
        return
    
    print("\n🚀 Starting Restaurant Assistant...")
    print("🔧 Backend will run on: http://localhost:8000")
    print("🌐 Frontend will run on: http://localhost:3000")
    print("⚡ Note: System will initialize automatically in background!")
    print("📝 Live logs will be shown below (Backend=Blue, Frontend=Green)")
    print("-" * 60)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend first
        backend_process = start_backend()
        if not backend_process:
            print("❌ Failed to start backend. Exiting.")
            return
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            print("❌ Failed to start frontend. Cleaning up...")
            if backend_process:
                backend_process.terminate()
            return
        
        # Wait for both processes to settle
        time.sleep(3)
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}=" * 60)
        print("🎉 Restaurant Assistant is now running!")
        print("🌐 Open your browser to: http://localhost:3000")
        print("🔧 Backend API available at: http://localhost:8000")
        print("⏹️  Press Ctrl+C to stop both services")
        print("📊 Logs will continue to stream below...")
        print(f"={Colors.RESET}" * 60)
        print()
        
        # Wait for processes and handle interruption
        try:
            while True:
                # Check if processes are still running
                if backend_process.poll() is not None:
                    print(f"{Colors.RED}❌ Backend process stopped unexpectedly{Colors.RESET}")
                    break
                if frontend_process.poll() is not None:
                    print(f"{Colors.RED}❌ Frontend process stopped unexpectedly{Colors.RESET}")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}👋 Shutting down Restaurant Assistant...{Colors.RESET}")
    
    finally:
        # Clean up processes
        if frontend_process:
            print(f"{Colors.YELLOW}🛑 Stopping frontend...{Colors.RESET}")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        if backend_process:
            print(f"{Colors.YELLOW}🛑 Stopping backend...{Colors.RESET}")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        print(f"{Colors.GREEN}✅ Cleanup complete. Goodbye!{Colors.RESET}")

if __name__ == "__main__":
    main() 