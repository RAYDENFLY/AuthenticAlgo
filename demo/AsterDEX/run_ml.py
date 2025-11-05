"""
Run ML Strategy Only
Parallel execution - Run this in separate terminal
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.logger import setup_logger
from trader_ml import MLTrader

# Setup logger with unique file
logger = setup_logger(log_to_file=True)


async def run_ml_strategy():
    """Run ML strategy only"""
    
    logger.info("=" * 80)
    logger.info("🤖 MACHINE LEARNING STRATEGY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Strategy: Pure ML (XGBoost 96% accuracy)")
    logger.info("Capital: $10")
    logger.info("Max Trades: 10")
    logger.info("Leverage: 5x-125x (dynamic)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize trader
    trader = MLTrader(capital=10.0)
    
    # Run competition
    logger.info("🚀 Starting ML strategy...\n")
    await trader.run_competition()
    
    # Calculate results
    roi = (trader.capital / trader.initial_capital - 1) * 100
    win_rate = (trader.win_count / trader.completed_trades * 100) if trader.completed_trades > 0 else 0
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("📊 MACHINE LEARNING - FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"\n💰 Capital:")
    logger.info(f"   Initial: ${trader.initial_capital:.2f}")
    logger.info(f"   Final:   ${trader.capital:.2f}")
    logger.info(f"   Profit:  ${trader.capital - trader.initial_capital:+.2f}")
    logger.info(f"\n📈 Performance:")
    logger.info(f"   ROI:       {roi:+.2f}%")
    logger.info(f"   Win Rate:  {win_rate:.1f}% ({trader.win_count}/{trader.completed_trades})")
    logger.info(f"   Trades:    {trader.completed_trades}")
    
    # Leverage analysis
    if trader.trades:
        avg_lev = sum(t['leverage'] for t in trader.trades) / len(trader.trades)
        max_lev = max(t['leverage'] for t in trader.trades)
        min_lev = min(t['leverage'] for t in trader.trades)
        logger.info(f"\n⚡ Leverage:")
        logger.info(f"   Average: {avg_lev:.1f}x")
        logger.info(f"   Range:   {min_lev}x - {max_lev}x")
    
    # Save individual report
    report_dir = Path(__file__).parent.parent.parent / "Reports" / "competition_parallel"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = {
        'strategy': 'Machine Learning',
        'timestamp': timestamp,
        'initial_capital': trader.initial_capital,
        'final_capital': trader.capital,
        'profit': trader.capital - trader.initial_capital,
        'roi_pct': roi,
        'total_trades': trader.completed_trades,
        'wins': trader.win_count,
        'losses': trader.loss_count,
        'win_rate_pct': win_rate,
        'trades': trader.trades
    }
    
    # Save JSON
    json_file = report_dir / f"ml_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"\n💾 Report saved: {json_file.name}")
    logger.info("\n" + "=" * 80)
    logger.info("✅ MACHINE LEARNING COMPLETE!")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    asyncio.run(run_ml_strategy())
