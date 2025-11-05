"""
Quick Test - Run Competition & ML Training
Cara tercepat untuk test semua fitur
"""

import sys
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               🤖 ASTERDEX TRADING BOT - QUICK START                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pilih mode eksekusi:

1. 🏆 COMPETITION ONLY - Test 3 strategy (Technical vs ML vs Hybrid)
   • Cepat (~2-5 menit)
   • Test strategy comparison
   • Results: Reports/benchmark/AsterDEX/

2. 🎓 ML TRAINING ONLY - Train models dengan real market data
   • Medium (~5-10 menit)
   • Train 3 models (XGBoost, RandomForest, LightGBM)
   • Results: database/ml_training.db + CSV reports

3. 🚀 RUN ALL - Competition + ML Training + Benchmark (PARALLEL)
   • Lambat (~10-15 menit)
   • Maximum efficiency
   • All results saved

4. ❌ EXIT

""")

choice = input("Pilih mode (1-4): ").strip()

base_dir = Path(__file__).parent.parent

if choice == "1":
    print("\n🏆 Starting Competition...")
    script = base_dir / "demo" / "AsterDEX" / "run_competition.py"
    import subprocess
    subprocess.run([sys.executable, str(script)])

elif choice == "2":
    print("\n🎓 Starting ML Training...")
    script = base_dir / "scripts" / "ml_continuous_learning.py"
    import subprocess
    subprocess.run([sys.executable, str(script)])

elif choice == "3":
    print("\n🚀 Starting All Scripts...")
    script = base_dir / "scripts" / "run_all_training.py"
    import subprocess
    subprocess.run([sys.executable, str(script)])

elif choice == "4":
    print("\n👋 Goodbye!")
    sys.exit(0)

else:
    print("\n❌ Invalid choice. Please run again.")
    sys.exit(1)
