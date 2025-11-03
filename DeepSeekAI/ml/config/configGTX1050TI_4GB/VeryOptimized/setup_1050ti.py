"""
Setup script untuk environment 1050 Ti
"""

import subprocess
import sys
import platform

def setup_1050ti_environment():
    """Setup Python environment untuk GTX 1050 Ti"""
    
    print("Setting up GTX 1050 Ti Optimized Environment...")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Install packages dengan versi yang compatible
    packages = [
        "pandas==2.1.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "xgboost==2.0.1",
        "tensorflow==2.13.0",  # Stable version untuk 1050 Ti
        "ccxt==4.2.36",
        "pandas-ta==0.3.14b0",
        "psutil==5.9.6",
        "pyyaml==6.0.1"
    ]
    
    print("Installing optimized packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nSetup completed!")
    print("\nImportant notes for GTX 1050 Ti:")
    print("1. Use config/gtx_1050ti_config.yaml for best performance")
    print("2. Monitor GPU memory usage during training")
    print("3. Start with small data sets and increase gradually")
    print("4. Use main_1050ti.py as entry point")

if __name__ == "__main__":
    setup_1050ti_environment()