"""
Run All Training & Competition Scripts
Jalankan 3 script sekaligus untuk testing dan ML learning
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def run_script(script_path, name, wait=False):
    """Run a Python script"""
    print_header(f"🚀 Starting: {name}")
    print(f"Script: {script_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        if wait:
            # Run and wait for completion
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print(f"\n✅ {name} completed successfully!\n")
            else:
                print(f"\n❌ {name} failed with code {result.returncode}\n")
            
            return result.returncode
        else:
            # Run in background
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✅ {name} started in background (PID: {process.pid})\n")
            return process
            
    except Exception as e:
        print(f"❌ Error running {name}: {e}\n")
        return None


def main():
    """Run all scripts"""
    base_dir = Path(__file__).parent.parent
    
    print_header("🤖 ASTERDEX TRADING BOT - FULL TRAINING SUITE")
    print("This will run:")
    print("  1. 3-Way Strategy Competition (run_competition.py)")
    print("  2. ML Continuous Learning (ml_continuous_learning.py)")
    print("  3. ML Benchmark (benchmark_ml.py)")
    print("\nAll scripts will run simultaneously for maximum efficiency!")
    
    input("\nPress ENTER to start...")
    
    # Scripts to run
    scripts = [
        {
            'path': base_dir / "demo" / "AsterDEX" / "run_competition.py",
            'name': "3-Way Strategy Competition",
            'wait': False
        },
        {
            'path': base_dir / "scripts" / "ml_continuous_learning.py",
            'name': "ML Continuous Learning",
            'wait': False
        },
        {
            'path': base_dir / "scripts" / "benchmark_ml.py",
            'name': "ML Benchmark",
            'wait': False
        }
    ]
    
    # Check if all scripts exist
    missing = []
    for script in scripts:
        if not script['path'].exists():
            missing.append(script['name'])
            print(f"❌ Script not found: {script['path']}")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} script(s). Please create them first.")
        return
    
    print_header("🚀 LAUNCHING ALL SCRIPTS")
    
    # Run all scripts
    processes = []
    start_time = time.time()
    
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Launching {script['name']}...")
        process = run_script(script['path'], script['name'], wait=script['wait'])
        
        if process:
            processes.append({
                'name': script['name'],
                'process': process
            })
        
        time.sleep(1)  # Small delay between launches
    
    print_header("⏳ ALL SCRIPTS RUNNING")
    print(f"Started {len(processes)} processes at {datetime.now().strftime('%H:%M:%S')}")
    print("\nMonitoring execution... (Press Ctrl+C to stop all)")
    print("-" * 80)
    
    try:
        # Monitor processes
        while any(p['process'].poll() is None for p in processes if hasattr(p['process'], 'poll')):
            time.sleep(5)
            
            # Check status
            for p in processes:
                if hasattr(p['process'], 'poll'):
                    if p['process'].poll() is not None:
                        print(f"✅ {p['name']} finished with code {p['process'].returncode}")
        
        elapsed = time.time() - start_time
        
        print_header("✅ ALL SCRIPTS COMPLETED")
        print(f"Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print("\n📊 Check the Reports folder for detailed results:")
        print("   • Reports/benchmark/AsterDEX/ - Competition results")
        print("   • Reports/ml_training/ - ML training reports")
        print("   • database/ml_training.db - Training history database")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Stopping all processes...")
        
        for p in processes:
            if hasattr(p['process'], 'terminate'):
                p['process'].terminate()
                print(f"   Stopped: {p['name']}")
        
        print("\n❌ All processes terminated.\n")


if __name__ == "__main__":
    main()
