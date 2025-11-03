"""
Simple Paper Trading Demo - Quick Start

A simplified paper trading demo to quickly validate the ensemble strategy.
This version runs a simulation based on recent historical data.

Usage:
    python demo/demo_paper_trading_simple.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from loguru import logger
import inspect
from backtesting.backtest_engine import BacktestEngine
print(inspect.signature(BacktestEngine.run))

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.ensemble import EnsembleStrategy
from backtesting.backtest_engine import BacktestEngine


def run_paper_trading_simulation():
    """Run a paper trading simulation"""
    
    print("\n" + "="*60)
    print(" PAPER TRADING SIMULATION - ENSEMBLE STRATEGY")
    print(" Quick validation using recent historical data")
    print("="*60)
    
    # Configuration
    symbol = "ETHUSDT"
    timeframe = "1h"
    initial_capital = 5.0
    leverage = 10
    
    print(f"\n⚙️  Configuration:")
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
        
        # Use last 30 days for simulation
        recent_df = df.tail(720)  # ~30 days of 1h candles
        print(f"   📅 Using last {len(recent_df)} candles for simulation")
        
    except FileNotFoundError:
        print(f"   ❌ Data file not found: {data_file}")
        print(f"   💡 Please run data collection first:")
        print(f"      python scripts/collect_large_dataset.py")
        return
    
    # Initialize strategy
    print(f"\n🎯 Initializing Ensemble Strategy...")
    config = {
        'ensemble_mode': 'weighted',
        'confidence_threshold': 0.6,
        'strategy_weights': {
            'rsi_macd': 0.35,
            'random_forest': 0.30,
            'xgboost': 0.20,
            'bollinger': 0.15
        },
        'ml_models_dir': 'ml/models',
        'use_ml': True
    }
    strategy = EnsembleStrategy(config)
    print(f"   ✅ Ensemble strategy initialized (weighted mode)")
    
    # Run backtest as paper trading simulation
    print(f"\n🔄 Running paper trading simulation...")
    print(f"   (This simulates real-time trading on recent historical data)")
    print(f"\n" + "="*60)
    
    backtest_config = {
        'initial_capital': initial_capital,
        'position_size_pct': 100,  # 100% with 10x leverage = full position
        'leverage': leverage,
        'commission_pct': 0.04,  # 0.04% trading fee
        'slippage_pct': 0.1  # 0.1% slippage
    }
    engine = BacktestEngine(backtest_config)
    
    # Fixed: Remove timeframe parameter
    results = engine.run(
        data=recent_df,
        strategy=strategy,
        symbol=symbol
    )
    
    # Display results
    print(f"\n" + "="*60)
    print(f"📊 PAPER TRADING SIMULATION RESULTS")
    print(f"="*60)
    print(f"Period: Last 30 days ({recent_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {recent_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"\n💰 Performance:")
    print(f"   Initial Capital: ${initial_capital:.2f}")
    print(f"   Final Capital: ${results['final_capital']:.2f}")
    print(f"   Total Return: ${results['total_return']:.2f} ({results['return_pct']:.2f}%)")
    
    print(f"\n📈 Trading Statistics:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Winning Trades: {results['winning_trades']} ({results['win_rate']:.1f}%)")
    print(f"   Losing Trades: {results['losing_trades']}")
    
    if results['total_trades'] > 0:
        print(f"\n💵 Profit/Loss:")
        print(f"   Gross Profit: ${results['gross_profit']:.2f}")
        print(f"   Gross Loss: ${results['gross_loss']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Average Trade: ${results['avg_trade_return']:.2f}")
    
    print(f"\n" + "="*60)
    
    # Comparison with backtest expectations
    expected_return = 0.5
    expected_win_rate = 55
    
    print(f"\n📈 vs BACKTEST EXPECTATIONS:")
    print(f"   Return: {results['return_pct']:.2f}% (expected: +{expected_return}%)")
    print(f"   Win Rate: {results['win_rate']:.1f}% (expected: {expected_win_rate}%)")
    
    if results['return_pct'] >= expected_return * 0.5:
        print(f"   ✅ Performance within acceptable range!")
    else:
        print(f"   ⚠️  Performance below expectations")
    
    # Next steps
    print(f"\n🎯 NEXT STEPS:")
    if results['return_pct'] > 0:
        print(f"   ✅ Paper trading successful!")
        print(f"   ✅ Strategy validated on recent data")
        print(f"   📊 Ready for live deployment with ${initial_capital}")
        print(f"\n   💡 To deploy live:")
        print(f"      1. Start with ${initial_capital} real capital")
        print(f"      2. Use weighted ensemble mode")
        print(f"      3. Monitor performance daily")
        print(f"      4. Retrain ML models weekly")
    else:
        print(f"   ⚠️  Consider optimizing strategy parameters")
        print(f"   ⚠️  Or try different timeframe/symbol")
        print(f"   💡 Try RSI+MACD on ETHUSDT 4h for more conservative approach")
    
    print(f"\n" + "="*60)
    print(f"✅ Paper trading simulation complete!")
    print(f"="*60 + "\n")


if __name__ == "__main__":
    run_paper_trading_simulation()