#!/usr/bin/env python3
"""
Neemsi Restaurant Assistant Launcher

Simple script to launch either the Streamlit UI or command line interface.
"""

import sys
import subprocess
import os
import time
import signal
import socket

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
                # Get process connections
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr.port == port:
                        print(f"🔄 Found process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                        proc.terminate()
                        killed_processes.append(proc.info['pid'])
                        
                        # Wait a bit for graceful termination
                        try:
                            proc.wait(timeout=3)
                            print(f"✅ Gracefully terminated process {proc.info['pid']}")
                        except psutil.TimeoutExpired:
                            # Force kill if it doesn't terminate gracefully
                            proc.kill()
                            print(f"⚡ Force killed process {proc.info['pid']}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process might have ended or we don't have permission
                continue
        
        if killed_processes:
            print(f"🗑️  Killed {len(killed_processes)} process(es) on port {port}")
            # Give a moment for the port to be released
            time.sleep(1)
        
        return len(killed_processes)
        
    except ImportError:
        print("⚠️  psutil not installed, trying alternative method...")
        return kill_processes_on_port_alternative(port)

def kill_processes_on_port_alternative(port):
    """Alternative method using lsof and kill commands (Unix/macOS)"""
    try:
        # Use lsof to find processes using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            killed_count = 0
            
            for pid in pids:
                if pid.strip():
                    try:
                        print(f"🔄 Killing process with PID: {pid}")
                        subprocess.run(['kill', '-TERM', pid], check=True)
                        killed_count += 1
                    except subprocess.CalledProcessError:
                        try:
                            # Try force kill if graceful termination fails
                            subprocess.run(['kill', '-KILL', pid], check=True)
                            print(f"⚡ Force killed process {pid}")
                            killed_count += 1
                        except subprocess.CalledProcessError:
                            print(f"❌ Failed to kill process {pid}")
            
            if killed_count > 0:
                print(f"🗑️  Killed {killed_count} process(es) on port {port}")
                time.sleep(1)
            
            return killed_count
            
    except FileNotFoundError:
        print("⚠️  lsof command not found. Unable to check for processes on port.")
    except Exception as e:
        print(f"❌ Error checking for processes on port {port}: {e}")
    
    return 0

def ensure_port_available(port=8501):
    """Ensure the specified port is available, killing any existing processes if needed"""
    print(f"🔍 Checking if port {port} is available...")
    
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is currently in use")
        killed_count = kill_processes_on_port(port)
        
        # Double-check if port is now available
        time.sleep(1)
        if is_port_in_use(port):
            print(f"❌ Port {port} is still in use after attempting to kill processes")
            print(f"💡 You may need to manually stop the process or use a different port")
            return False
        else:
            print(f"✅ Port {port} is now available")
            return True
    else:
        print(f"✅ Port {port} is available")
        return True

def run_streamlit():
    """Launch the Streamlit UI on port 8501"""
    print("🚀 Starting Neemsi Restaurant Assistant - Streamlit UI...")
    print("📱 The app will run at http://localhost:8501")
    print("🔧 Debug output will appear in this terminal")
    print("-" * 60)
    
    # Ensure port 8501 is available
    if not ensure_port_available(8501):
        print("❌ Unable to make port 8501 available. Exiting.")
        return
    
    try:
        # Start Streamlit with explicit port configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🔄 Launching Streamlit...")
        print(f"🌐 Opening browser at http://localhost:8501")
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("💡 Make sure you have installed the requirements: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
        print("🧹 Cleaning up...")
        # Optional: Kill any remaining processes on port 8501
        kill_processes_on_port(8501)

def run_command_line():
    """Launch the command line interface"""
    print("🚀 Starting Neemsi Restaurant Assistant - Command Line Interface...")
    print("💬 Type 'exit' to quit")
    print("-" * 60)
    
    try:
        from main import run_command_line_interface
        run_command_line_interface()
    except ImportError as e:
        print(f"❌ Error importing main module: {e}")
        print("💡 Make sure you have installed the requirements: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Command line interface stopped.")

def check_requirements():
    """Check if basic requirements are met"""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("📝 Please create a .env file with your API keys and database credentials.")
        print("📖 See README_streamlit.md for setup instructions.")
        return False
    
    # Check if main dependencies are available
    try:
        import streamlit
        import langchain
        import supabase
        import neo4j
        import psutil
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("💡 Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🍽️  Neemsi Restaurant Assistant Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        return
    
    print("\nChoose how you'd like to run the application:")
    print("1. 🌐 Streamlit UI (Recommended) - Beautiful web interface")
    print("2. 💻 Command Line - Terminal-based chat")
    print("3. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                run_streamlit()
                break
            elif choice == "2":
                run_command_line()
                break
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main() 