#!/usr/bin/env python3
"""
ML Continuous Learning Runner
Auto-screen ALL coins dari AsterDEX dan train ML models

Features:
- Mass screening 100+ coins
- Smart filtering berdasarkan volume & volatility
- Auto-training untuk top performers
- Database + CSV reports

Usage: python run_learning.py [options]
Options:
    --screening-only    : Hanya screening tanpa training
    --top N            : Train top N coins (default: 15)
    --max-concurrent N : Max concurrent screening (default: 15)
"""

import asyncio
import sys
import argparse
from pathlib import Path
import os

# Get script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Import learner
try:
    # Try relative import first
    if os.path.exists(SCRIPT_DIR / "ml_continuous_learner.py"):
        spec = __import__('importlib.util').util.spec_from_file_location(
            "ml_continuous_learner",
            str(SCRIPT_DIR / "ml_continuous_learner.py")
        )
        ml_module = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(ml_module)
        MLContinuousLearnerEnhanced = ml_module.MLContinuousLearnerEnhanced
    else:
        raise ImportError("File not found")
except Exception as e:
    print(f"❌ Error loading ml_continuous_learner.py: {e}")
    print(f"   Script dir: {SCRIPT_DIR}")
    print(f"   Expected file: {SCRIPT_DIR / 'ml_continuous_learner.py'}")
    print(f"   File exists: {os.path.exists(SCRIPT_DIR / 'ml_continuous_learner.py')}")
    sys.exit(1)


async def main():
    """Main runner function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='ML Continuous Learning')
    parser.add_argument('--screening-only', action='store_true', 
                       help='Hanya screening tanpa training')
    parser.add_argument('--top', type=int, default=15,
                       help='Jumlah top coins untuk training (default: 15)')
    parser.add_argument('--max-concurrent', type=int, default=15,
                       help='Max concurrent screening (default: 15)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚀 ML CONTINUOUS LEARNING - MULTI-COIN AUTO-SCREENING")
    print("=" * 80)
    print("")
    print("📍 Configuration:")
    print(f"   • Screening only: {'Yes' if args.screening_only else 'No'}")
    print(f"   • Top coins to train: {args.top}")
    print(f"   • Max concurrent: {args.max_concurrent}")
    print("")
    print("💡 This will:")
    print("   1. Fetch ALL available symbols from AsterDEX")
    print("   2. Screen each coin for trading potential")
    print("   3. Rank by volume, volatility, and trend")
    if not args.screening_only:
        print(f"   4. Train ML models on top {args.top} coins")
        print("   5. Save training results to database & CSV")
    print("")
    print("⏱️  Estimated time: 5-15 minutes")
    print("")
    
    try:
        # Initialize learner
        learner = MLContinuousLearnerEnhanced()
        
        # Step 1: Mass screening
        print("🔍 STEP 1: Mass Coin Screening")
        print("-" * 80)
        await learner.run_mass_screening(max_concurrent=args.max_concurrent)
        
        # Step 2: Generate report
        print("\n📊 STEP 2: Generate Screening Report")
        print("-" * 80)
        learner.generate_screening_report()
        
        if not args.screening_only:
            # Step 3: Smart training
            print(f"\n🎓 STEP 3: Train ML Models on Top {args.top} Coins")
            print("-" * 80)
            await learner.smart_training_selection(top_n=args.top)
        
        print("\n" + "=" * 80)
        print("✅ ML LEARNING COMPLETE")
        print("=" * 80)
        print("")
        print("📁 Results saved:")
        print(f"   • Database: {learner.db_path}")
        print(f"   • Reports: {learner.reports_dir}")
        print("")
        print("💡 Next steps:")
        print("   • Check screening results in database")
        print("   • Review training accuracy in CSV reports")
        print("   • Use top models in your trading bot")
        print("")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        print("   Partial results may be saved")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())