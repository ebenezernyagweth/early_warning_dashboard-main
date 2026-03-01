import os
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import time

#BASE_DIR = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/early_warning_dashboard-main"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Wards_counties = os.path.join(BASE_DIR, "wards_and_counties.py")  
Computation = os.path.join(BASE_DIR, "compute_trends.py")
App = os.path.join(BASE_DIR, "app.py")

def simple_stop_server(port=8080):
    """Simpler version that just tries to kill processes on the port"""
    print(f"Attempting to free port {port}...")
    
    commands = [
        f"lsof -ti:{port} | xargs kill -9",
        f"fuser -k {port}/tcp", 
        f"pkill -f ':{port}'"
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ Successfully executed: {cmd}")
                time.sleep(1)
                return True
        except subprocess.TimeoutExpired:
            print(f"✗ Command timed out: {cmd}")
        except Exception as e:
            print(f"✗ Command failed: {cmd} - {e}")
    
    print(f"No processes found or stopped on port {port}")
    return False

def run_script(script_path, is_flask_app=False):
    """Run a Python script using the same Python interpreter"""
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    try:
        if is_flask_app:
            # For Flask app, don't capture output and don't wait for completion
            print(f"🚀 Starting Flask app: {os.path.basename(script_path)}")
            process = subprocess.Popen([sys.executable, script_path], 
                                     cwd=BASE_DIR)
            
            # Give the Flask app time to start
            time.sleep(3)
            
            # Check if the process is still running (didn't crash immediately)
            if process.poll() is None:
                print(f"✓ Flask app started successfully (PID: {process.pid})")
                print(f"🌐 Dashboard should be available at http://localhost:8080")
                return True
            else:
                print(f"✗ Flask app failed to start (exited with code {process.returncode})")
                return False
        else:
            # For regular scripts, capture output and wait for completion
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, cwd=BASE_DIR, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ Successfully ran {os.path.basename(script_path)}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                return True
            else:
                print(f"✗ Failed to run {os.path.basename(script_path)}")
                print(f"Error: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"✗ Script {script_path} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Exception while running {script_path}: {e}")
        return False

def run_preparation_scripts():
    """Run the data preparation scripts"""
    print("=" * 50)
    print("STEP 1: Running data preparation scripts...")
    print("=" * 50)
    
    # Run wards_and_counties.py
    print("1/2 Running wards_and_counties.py...")
    if not run_script(Wards_counties):
        print("❌ Stopping pipeline due to failure in wards_and_counties.py")
        return False
    
    # Run compute_trends.py
    print("2/2 Running compute_trends.py...")
    if not run_script(Computation):
        print("❌ Stopping pipeline due to failure in compute_trends.py")
        return False
    
    print("✅ Data preparation completed successfully!")
    return True

def start_dashboard():
    """Start the Flask dashboard"""
    print("\n" + "=" * 50)
    print("STEP 2: Starting dashboard server...")
    print("=" * 50)
    
    # Stop any existing server
    simple_stop_server(8080)
    
    # Start the Flask app
    if run_script(App, is_flask_app=True):
        print("\n🎉 Dashboard started successfully!")
        print("📊 Access your dashboard at: http://localhost:8080")
        print("🛑 Press Ctrl+C to stop the server")
        return True
    else:
        print("❌ Failed to start dashboard")
        return False

def run_full_pipeline():
    """Run the complete pipeline"""
    print("🚀 Starting NDMA DEWS Dashboard Pipeline...")
    print(f"📁 Base directory: {BASE_DIR}")
    
    # Step 1: Run preparation scripts
    if not run_preparation_scripts():
        return False
    
    # Step 2: Start dashboard
    if not start_dashboard():
        return False
    
    # Keep the main process alive so the Flask app continues running
    try:
        print("\n⏳ Dashboard is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down dashboard...")
        simple_stop_server(8080)
        print("✅ Dashboard stopped successfully!")

if __name__ == "__main__":
    run_full_pipeline()
