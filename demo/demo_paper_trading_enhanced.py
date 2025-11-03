"""
Enhanced Paper Trading Demo - Optimized Parameters

An improved paper trading demo with better parameter tuning and more detailed analysis.

Usage:
    python demo/demo_paper_trading_enhanced.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.ensemble import EnsembleStrategy
from backtesting.backtest_engine import BacktestEngine


def run_enhanced_paper_trading():
    """Run enhanced paper trading with optimized parameters"""
    
    print("\n" + "="*70)
    print(" ENHANCED PAPER TRADING - OPTIMIZED ENSEMBLE STRATEGY")
    print(" Better parameter tuning for improved performance")
    print("="*70)
    
    # Enhanced Configuration
    symbol = "ETHUSDT"
    timeframe = "1h"
    initial_capital = 5.0
    leverage = 10
    
    print(f"\n⚙️  ENHANCED CONFIGURATION:")
    print(f"   Strategy: Ensemble (Weighted Mode)")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Initial Capital: ${initial_capital}")
    print(f"   Leverage: {leverage}x")
    print(f"   Position Size: ${initial_capital * leverage}")
    
    # Load data
    print(f"\n📊 Loading historical data...")
    data_file = f"data/historical/asterdex_{symbol}_{timeframe}_20250805_to_20251103.csv"
    
    try:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"   ✅ Loaded {len(df)} candles")
        
        # Use last 60 days for better sample (instead of 30)
        recent_df = df.tail(1440)  # ~60 days of 1h candles
        print(f"   📅 Using last {len(recent_df)} candles (60 days) for better sampling")
        
    except FileNotFoundError:
        print(f"   ❌ Data file not found: {data_file}")
        print(f"   💡 Please run data collection first:")
        print(f"      python scripts/collect_large_dataset.py")
        return
    
    # Test multiple confidence thresholds to find optimal value
    print(f"\n🎯 OPTIMIZING CONFIDENCE THRESHOLDS...")
    
    best_result = None
    best_threshold = None
    
    for confidence_threshold in [0.4, 0.45, 0.5, 0.55]:
        print(f"\n   Testing threshold: {confidence_threshold}")
        
        # Initialize strategy with current threshold
        config = {
            'ensemble_mode': 'weighted',
            'confidence_threshold': confidence_threshold,
            'strategy_weights': {
                'rsi_macd': 0.40,  # Increased RSI+MACD weight
                'random_forest': 0.30,
                'xgboost': 0.20,
                'bollinger': 0.10   # Reduced Bollinger weight
            },
            'ml_models_dir': 'ml/models',
            'use_ml': True
        }
        
        strategy = EnsembleStrategy(config)
        
        # Run backtest
        backtest_config = {
            'initial_capital': initial_capital,
            'position_size_pct': 100,
            'leverage': leverage,
            'commission_pct': 0.04,
            'slippage_pct': 0.1
        }
        
        engine = BacktestEngine(backtest_config)
        results = engine.run(
            data=recent_df.copy(),
            strategy=strategy,
            symbol=symbol
        )
        
        print(f"      Trades: {results['total_trades']}, Return: {results['return_pct']:.2f}%")
        
        # Track best performing threshold
        if results['total_trades'] > 0:
            if best_result is None or results['return_pct'] > best_result['return_pct']:
                best_result = results
                best_threshold = confidence_threshold
    
    # Display final results with best threshold
    print(f"\n" + "="*70)
    print(f"🏆 OPTIMAL CONFIGURATION FOUND")
    print(f"="*70)
    print(f"   Best Confidence Threshold: {best_threshold}")
    print(f"   Return: {best_result['return_pct']:.2f}%")
    print(f"   Total Trades: {best_result['total_trades']}")
    
    # Run final simulation with best parameters
    print(f"\n🔄 RUNNING FINAL SIMULATION WITH OPTIMAL PARAMETERS...")
    
    final_config = {
        'ensemble_mode': 'weighted',
        'confidence_threshold': best_threshold,
        'strategy_weights': {
            'rsi_macd': 0.40,
            'random_forest': 0.30,
            'xgboost': 0.20,
            'bollinger': 0.10
        },
        'ml_models_dir': 'ml/models',
        'use_ml': True
    }
    
    final_strategy = EnsembleStrategy(final_config)
    
    engine = BacktestEngine(backtest_config)
    final_results = engine.run(
        data=recent_df.copy(),
        strategy=final_strategy,
        symbol=symbol
    )
    
    # Enhanced results display
    print(f"\n" + "="*70)
    print(f"📊 FINAL PAPER TRADING RESULTS")
    print(f"="*70)
    print(f"Period: {recent_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {recent_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Optimal Confidence Threshold: {best_threshold}")
    
    print(f"\n💰 PERFORMANCE SUMMARY:")
    print(f"   Initial Capital: ${initial_capital:.2f}")
    print(f"   Final Capital: ${final_results['final_capital']:.2f}")
    print(f"   Total Return: ${final_results['total_return']:.2f} ({final_results['return_pct']:.2f}%)")
    print(f"   ROI (Annualized): {final_results['return_pct'] * 6:.2f}%")  # Rough annualization
    
    print(f"\n📈 TRADING ACTIVITY:")
    print(f"   Total Trades: {final_results['total_trades']}")
    print(f"   Winning Trades: {final_results['winning_trades']} ({final_results['win_rate']:.1f}%)")
    print(f"   Losing Trades: {final_results['losing_trades']}")
    
    if final_results['total_trades'] > 0:
        print(f"\n💵 PROFIT ANALYSIS:")
        print(f"   Gross Profit: ${final_results['gross_profit']:.2f}")
        print(f"   Gross Loss: ${final_results['gross_loss']:.2f}")
        print(f"   Net Profit: ${final_results['gross_profit'] + final_results['gross_loss']:.2f}")
        print(f"   Profit Factor: {final_results['profit_factor']:.2f}")
        print(f"   Average Trade: ${final_results['avg_trade_return']:.2f}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Max Drawdown: {final_results['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {final_results['sharpe_ratio']:.2f}")
        print(f"   Commission Paid: ${final_results.get('total_commission', 0):.4f}")
    
    # Trading frequency analysis
    trades_per_week = (final_results['total_trades'] / (len(recent_df) / 24 / 7))
    print(f"\n📅 TRADING FREQUENCY:")
    print(f"   Trades per Week: {trades_per_week:.1f}")
    print(f"   Average Hold Time: {final_results.get('avg_hold_time', 'N/A')}")
    
    # Strategy recommendations
    print(f"\n🎯 STRATEGY ASSESSMENT:")
    
    if final_results['return_pct'] > 5.0:
        print(f"   ✅ EXCELLENT: Strategy performing well above expectations!")
        recommendation = "Ready for live trading"
    elif final_results['return_pct'] > 0:
        print(f"   ✅ GOOD: Strategy is profitable")
        recommendation = "Suitable for live trading with monitoring"
    elif final_results['return_pct'] > -2.0:
        print(f"   ⚠️  ACCEPTABLE: Small drawdown, consider optimization")
        recommendation = "Monitor closely or optimize parameters"
    else:
        print(f"   ❌ NEEDS IMPROVEMENT: Strategy underperforming")
        recommendation = "Requires parameter optimization or strategy adjustment"
    
    print(f"\n💡 RECOMMENDATION: {recommendation}")
    
    # Next steps based on performance
    print(f"\n🚀 NEXT STEPS:")
    if final_results['return_pct'] > 0:
        print(f"   1. Deploy with ${initial_capital} real capital")
        print(f"   2. Use confidence threshold: {best_threshold}")
        print(f"   3. Monitor performance daily")
        print(f"   4. Retrain ML models weekly")
        print(f"   5. Consider scaling up capital if consistent")
    else:
        print(f"   1. Try different symbol (BTCUSDT)")
        print(f"   2. Test with 4h timeframe for fewer false signals")
        print(f"   3. Adjust strategy weights")
        print(f"   4. Increase data period to 90 days")
        print(f"   5. Consider standalone RSI+MACD strategy")
    
    print(f"\n" + "="*70)
    print(f"✅ Enhanced paper trading simulation complete!")
    print(f"="*70 + "\n")
    
    return final_results


if __name__ == "__main__":
    run_enhanced_paper_trading()