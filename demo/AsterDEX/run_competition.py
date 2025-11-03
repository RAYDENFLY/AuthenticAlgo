"""
AsterDEX Trading Competition
3 Strategies compete: Technical Analysis vs ML vs Hybrid

Run 10 trades each with $10 capital
Generate comprehensive comparison report
"""

import asyncio
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.logger import setup_logger
from trader_technical import TechnicalAnalysisTrader
from trader_ml import MLTrader
from trader_hybrid import HybridTrader

logger = setup_logger()


async def run_competition():
    """Run full 3-way competition"""
    
    logger.info("=" * 80)
    logger.info("🏆 ASTERDEX TRADING COMPETITION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 Competition Rules:")
    logger.info("   • 3 Strategies compete")
    logger.info("   • 10 trades each")
    logger.info("   • $10 starting capital")
    logger.info("   • Real-time AsterDEX data")
    logger.info("   • Auto coin screening")
    logger.info("   • Dynamic leverage (5x-125x)")
    logger.info("")
    logger.info("🤖 Competitors:")
    logger.info("   1. Technical Analysis (RSI + MACD)")
    logger.info("   2. Pure ML (XGBoost 96% accuracy)")
    logger.info("   3. Hybrid (TA + ML combined)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize traders
    traders = [
        TechnicalAnalysisTrader(capital=10.0),
        MLTrader(capital=10.0),
        HybridTrader(capital=10.0)
    ]
    
    # Run competitions in parallel (or sequential for cleaner logs)
    logger.info("🚀 Starting competitions...\n")
    
    # Sequential for cleaner output
    for trader in traders:
        await trader.run_competition()
        await asyncio.sleep(2)  # Pause between traders
    
    # Generate comparison report
    generate_comparison_report(traders)


def generate_comparison_report(traders):
    """Generate comprehensive comparison report"""
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL COMPARISON REPORT")
    logger.info("=" * 80)
    
    # Collect results
    results = []
    for trader in traders:
        roi = (trader.capital / trader.initial_capital - 1) * 100
        win_rate = (trader.win_count / trader.completed_trades * 100) if trader.completed_trades > 0 else 0
        
        results.append({
            'strategy': trader.strategy_name,
            'initial_capital': trader.initial_capital,
            'final_capital': trader.capital,
            'profit': trader.capital - trader.initial_capital,
            'roi_pct': roi,
            'total_trades': trader.completed_trades,
            'wins': trader.win_count,
            'losses': trader.loss_count,
            'win_rate_pct': win_rate,
            'trades': trader.trades
        })
    
    # Sort by ROI
    results_sorted = sorted(results, key=lambda x: x['roi_pct'], reverse=True)
    
    # Print rankings
    logger.info("\n🏆 RANKINGS (by ROI):")
    logger.info("-" * 80)
    
    for i, result in enumerate(results_sorted, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        logger.info(f"\n{medal} #{i}: {result['strategy']}")
        logger.info(f"   Capital: ${result['initial_capital']:.2f} → ${result['final_capital']:.2f}")
        logger.info(f"   Profit: ${result['profit']:+.2f}")
        logger.info(f"   ROI: {result['roi_pct']:+.2f}%")
        logger.info(f"   Win Rate: {result['win_rate_pct']:.1f}% ({result['wins']}/{result['total_trades']})")
    
    # Detailed comparison
    logger.info("\n\n📈 DETAILED METRICS:")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<20} {'ROI':<12} {'Win Rate':<12} {'Profit':<12} {'Trades':<8}")
    logger.info("-" * 80)
    
    for result in results_sorted:
        logger.info(
            f"{result['strategy']:<20} "
            f"{result['roi_pct']:>+10.2f}% "
            f"{result['win_rate_pct']:>10.1f}% "
            f"${result['profit']:>+10.2f} "
            f"{result['total_trades']:>6}"
        )
    
    # Winner analysis
    winner = results_sorted[0]
    logger.info("\n\n🎯 WINNER ANALYSIS:")
    logger.info("-" * 80)
    logger.info(f"🏆 Champion: {winner['strategy']}")
    logger.info(f"\n   Performance:")
    logger.info(f"   • ROI: {winner['roi_pct']:+.2f}%")
    logger.info(f"   • Win Rate: {winner['win_rate_pct']:.1f}%")
    logger.info(f"   • Total Profit: ${winner['profit']:+.2f}")
    logger.info(f"   • Capital Growth: {(winner['final_capital']/winner['initial_capital']):.2f}x")
    
    # Calculate average leverage per strategy
    logger.info("\n\n⚡ LEVERAGE ANALYSIS:")
    logger.info("-" * 80)
    
    for result in results_sorted:
        if result['trades']:
            avg_lev = sum(t['leverage'] for t in result['trades']) / len(result['trades'])
            max_lev = max(t['leverage'] for t in result['trades'])
            min_lev = min(t['leverage'] for t in result['trades'])
            logger.info(f"{result['strategy']:<20} Avg: {avg_lev:>5.1f}x  Min: {min_lev:>3}x  Max: {max_lev:>3}x")
    
    # Recommendation
    logger.info("\n\n💡 RECOMMENDATION:")
    logger.info("-" * 80)
    
    if winner['roi_pct'] > 50:
        logger.info("✅ EXCELLENT performance! Winner strategy is highly recommended.")
    elif winner['roi_pct'] > 20:
        logger.info("✅ GOOD performance! Winner strategy shows promise.")
    elif winner['roi_pct'] > 0:
        logger.info("⚠️ MODERATE performance. Consider optimizing parameters.")
    else:
        logger.info("❌ NEGATIVE returns. All strategies need improvement.")
    
    logger.info(f"\n   Best strategy for growth: {winner['strategy']}")
    logger.info(f"   Expected monthly ROI: ~{winner['roi_pct']:.1f}%")
    
    # Save comprehensive report
    save_comprehensive_report(results_sorted, winner)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ COMPETITION COMPLETE!")
    logger.info("=" * 80)
    logger.info("")


def save_comprehensive_report(results, winner):
    """Save detailed comparison report"""
    try:
        report_dir = Path(__file__).parent.parent.parent / "Reports" / "benchmark" / "AsterDEX"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = report_dir / f"competition_results_{timestamp}.json"
        report_data = {
            'timestamp': timestamp,
            'competition': 'AsterDEX_3_Way_Competition',
            'rules': {
                'capital': 10.0,
                'max_trades': 10,
                'leverage_range': '5x-125x',
                'strategies': ['Technical_Analysis', 'Pure_ML', 'Hybrid_TA_ML']
            },
            'results': results,
            'winner': {
                'strategy': winner['strategy'],
                'roi_pct': winner['roi_pct'],
                'win_rate_pct': winner['win_rate_pct'],
                'final_capital': winner['final_capital']
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Markdown report
        md_file = report_dir / f"competition_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# AsterDEX Trading Competition Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Competition Rules\n\n")
            f.write("- **Starting Capital:** $10\n")
            f.write("- **Number of Trades:** 10 per strategy\n")
            f.write("- **Leverage Range:** 5x - 125x (dynamic)\n")
            f.write("- **Data Source:** AsterDEX (real-time)\n")
            f.write("- **Coin Selection:** Automatic screening\n\n")
            
            f.write("## Strategies\n\n")
            f.write("1. **Technical Analysis** - RSI + MACD indicators\n")
            f.write("2. **Pure ML** - XGBoost model (96% accuracy)\n")
            f.write("3. **Hybrid** - TA + ML combined\n\n")
            
            f.write("## Final Rankings\n\n")
            f.write("| Rank | Strategy | ROI | Win Rate | Profit | Final Capital |\n")
            f.write("|------|----------|-----|----------|--------|---------------|\n")
            
            for i, result in enumerate(results, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                f.write(f"| {medal} #{i} | {result['strategy']} | "
                       f"{result['roi_pct']:+.2f}% | "
                       f"{result['win_rate_pct']:.1f}% | "
                       f"${result['profit']:+.2f} | "
                       f"${result['final_capital']:.2f} |\n")
            
            f.write("\n## Winner Analysis\n\n")
            f.write(f"**🏆 Champion:** {winner['strategy']}\n\n")
            f.write(f"- **ROI:** {winner['roi_pct']:+.2f}%\n")
            f.write(f"- **Win Rate:** {winner['win_rate_pct']:.1f}%\n")
            f.write(f"- **Total Profit:** ${winner['profit']:+.2f}\n")
            f.write(f"- **Capital Growth:** {(winner['final_capital']/winner['initial_capital']):.2f}x\n\n")
            
            f.write("## Leverage Analysis\n\n")
            f.write("| Strategy | Avg Leverage | Min | Max |\n")
            f.write("|----------|--------------|-----|-----|\n")
            
            for result in results:
                if result['trades']:
                    avg_lev = sum(t['leverage'] for t in result['trades']) / len(result['trades'])
                    max_lev = max(t['leverage'] for t in result['trades'])
                    min_lev = min(t['leverage'] for t in result['trades'])
                    f.write(f"| {result['strategy']} | {avg_lev:.1f}x | {min_lev}x | {max_lev}x |\n")
            
            f.write("\n## Recommendation\n\n")
            f.write(f"**Best strategy:** {winner['strategy']}\n\n")
            f.write(f"This strategy demonstrated the highest ROI of **{winner['roi_pct']:+.2f}%** ")
            f.write(f"with a win rate of **{winner['win_rate_pct']:.1f}%**.\n\n")
            
            if winner['roi_pct'] > 50:
                f.write("✅ **EXCELLENT** - Highly recommended for live trading\n")
            elif winner['roi_pct'] > 20:
                f.write("✅ **GOOD** - Shows strong potential\n")
            elif winner['roi_pct'] > 0:
                f.write("⚠️ **MODERATE** - Consider parameter optimization\n")
            else:
                f.write("❌ **NEEDS IMPROVEMENT** - Requires strategy adjustment\n")
        
        logger.info(f"\n💾 Reports saved:")
        logger.info(f"   • {json_file.name}")
        logger.info(f"   • {md_file.name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save report: {e}")


if __name__ == "__main__":
    asyncio.run(run_competition())
