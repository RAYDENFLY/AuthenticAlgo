"""
Combine Results from Parallel Competition
Run this after all 3 strategies complete
"""

import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.logger import setup_logger

logger = setup_logger()


def combine_results():
    """Combine results from 3 parallel strategies"""
    
    report_dir = Path(__file__).parent.parent.parent / "Reports" / "competition_parallel"
    
    if not report_dir.exists():
        logger.error(f"❌ Report directory not found: {report_dir}")
        logger.error("   Run the 3 strategies first!")
        return
    
    # Find latest results for each strategy
    technical_files = sorted(report_dir.glob("technical_results_*.json"), reverse=True)
    ml_files = sorted(report_dir.glob("ml_results_*.json"), reverse=True)
    hybrid_files = sorted(report_dir.glob("hybrid_results_*.json"), reverse=True)
    
    if not technical_files or not ml_files or not hybrid_files:
        logger.error("❌ Missing results! Make sure all 3 strategies completed.")
        logger.info(f"   Found: {len(technical_files)} Technical, {len(ml_files)} ML, {len(hybrid_files)} Hybrid")
        return
    
    # Load results
    with open(technical_files[0]) as f:
        technical_result = json.load(f)
    
    with open(ml_files[0]) as f:
        ml_result = json.load(f)
    
    with open(hybrid_files[0]) as f:
        hybrid_result = json.load(f)
    
    results = [technical_result, ml_result, hybrid_result]
    
    # Sort by ROI
    results_sorted = sorted(results, key=lambda x: x['roi_pct'], reverse=True)
    
    # Print comparison report
    logger.info("\n" + "=" * 80)
    logger.info("🏆 PARALLEL COMPETITION - COMBINED RESULTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 All 3 strategies ran in parallel (separate Python processes)")
    logger.info("")
    
    # Rankings
    logger.info("🏆 RANKINGS (by ROI):")
    logger.info("-" * 80)
    
    for i, result in enumerate(results_sorted, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        logger.info(f"\n{medal} #{i}: {result['strategy']}")
        logger.info(f"   Capital: ${result['initial_capital']:.2f} → ${result['final_capital']:.2f}")
        logger.info(f"   Profit: ${result['profit']:+.2f}")
        logger.info(f"   ROI: {result['roi_pct']:+.2f}%")
        logger.info(f"   Win Rate: {result['win_rate_pct']:.1f}% ({result['wins']}/{result['total_trades']})")
    
    # Detailed comparison table
    logger.info("\n\n📈 DETAILED COMPARISON:")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<25} {'ROI':<12} {'Win Rate':<12} {'Profit':<12}")
    logger.info("-" * 80)
    
    for result in results_sorted:
        logger.info(
            f"{result['strategy']:<25} "
            f"{result['roi_pct']:>+10.2f}% "
            f"{result['win_rate_pct']:>10.1f}% "
            f"${result['profit']:>+10.2f}"
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
    
    # Leverage analysis
    logger.info("\n\n⚡ LEVERAGE ANALYSIS:")
    logger.info("-" * 80)
    
    for result in results_sorted:
        if result['trades']:
            avg_lev = sum(t['leverage'] for t in result['trades']) / len(result['trades'])
            max_lev = max(t['leverage'] for t in result['trades'])
            min_lev = min(t['leverage'] for t in result['trades'])
            logger.info(f"{result['strategy']:<25} Avg: {avg_lev:>5.1f}x  Min: {min_lev:>3}x  Max: {max_lev:>3}x")
    
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
    
    logger.info(f"\n   Best strategy: {winner['strategy']}")
    logger.info(f"   Expected ROI: ~{winner['roi_pct']:.1f}%")
    
    # Save combined report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    combined_data = {
        'timestamp': timestamp,
        'competition_type': 'Parallel_Execution',
        'results': results_sorted,
        'winner': {
            'strategy': winner['strategy'],
            'roi_pct': winner['roi_pct'],
            'win_rate_pct': winner['win_rate_pct'],
            'final_capital': winner['final_capital']
        }
    }
    
    # Save JSON
    json_file = report_dir / f"combined_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    # Save Markdown
    md_file = report_dir / f"combined_report_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Parallel Competition - Combined Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Execution Type:** Parallel (3 separate Python processes)\n\n")
        
        f.write("## Final Rankings\n\n")
        f.write("| Rank | Strategy | ROI | Win Rate | Profit | Final Capital |\n")
        f.write("|------|----------|-----|----------|--------|---------------|\n")
        
        for i, result in enumerate(results_sorted, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            f.write(f"| {medal} #{i} | {result['strategy']} | "
                   f"{result['roi_pct']:+.2f}% | "
                   f"{result['win_rate_pct']:.1f}% | "
                   f"${result['profit']:+.2f} | "
                   f"${result['final_capital']:.2f} |\n")
        
        f.write("\n## Winner\n\n")
        f.write(f"**🏆 {winner['strategy']}**\n\n")
        f.write(f"- ROI: {winner['roi_pct']:+.2f}%\n")
        f.write(f"- Win Rate: {winner['win_rate_pct']:.1f}%\n")
        f.write(f"- Profit: ${winner['profit']:+.2f}\n")
    
    logger.info(f"\n\n💾 Combined reports saved:")
    logger.info(f"   • {json_file.name}")
    logger.info(f"   • {md_file.name}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ COMPARISON COMPLETE!")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("📁 Individual logs saved in:")
    logger.info("   • logs/technical_strategy.log")
    logger.info("   • logs/ml_strategy.log")
    logger.info("   • logs/hybrid_strategy.log")
    logger.info("")


if __name__ == "__main__":
    combine_results()
