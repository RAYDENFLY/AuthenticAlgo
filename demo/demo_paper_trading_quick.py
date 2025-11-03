"""
Simple Paper Trading Demo - ASCII Only
Quick validation without Unicode issues
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.ensemble import EnsembleStrategy
from backtesting.backtest_engine import BacktestEngine


def run_paper_trading():
    """Run paper trading simulation"""
    
    print("\n" + "="*70)
    print(" PAPER TRADING SIMULATION - ENSEMBLE STRATEGY")
    print(" Quick validation using recent historical data (last 30 days)")
    print("="*70)
    
    # Configuration
    symbol = "ETHUSDT"
    timeframe = "1h"
    initial_capital = 5.0
    leverage = 10
    
    print(f"\nConfiguration:")
    print(f"  Strategy: Ensemble (Weighted Mode)")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital}")
    print(f"  Leverage: {leverage}x")
    print(f"  Position Size: ${initial_capital * leverage}")
    
    # Load data
    print(f"\nLoading historical data...")
    data_file = f"data/historical/asterdex_{symbol}_{timeframe}_20250805_to_20251103.csv"
    
    try:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"  [OK] Loaded {len(df)} candles")
        
        # Use last 30 days
        recent_df = df.tail(720)
        print(f"  [OK] Using last {len(recent_df)} candles")
        start_date = recent_df['timestamp'].iloc[0].strftime('%Y-%m-%d')
        end_date = recent_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        print(f"  Period: {start_date} to {end_date}")
        
    except FileNotFoundError:
        print(f"  [ERROR] Data file not found: {data_file}")
        print(f"  Please run: python scripts/collect_large_dataset.py")
        return
    
    # Initialize ensemble strategy
    print(f"\nInitializing Ensemble Strategy...")
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
    print(f"  [OK] Ensemble strategy ready (weighted mode)")
    
    # Run backtest
    print(f"\nRunning paper trading simulation...")
    print(f"  (Simulating real-time trading on recent data)")
    print("\n" + "="*70)
    
    backtest_config = {
        'initial_capital': initial_capital,
        'position_size_pct': 100,
        'leverage': leverage,
        'commission_pct': 0.04,
        'slippage_pct': 0.1
    }
    engine = BacktestEngine(backtest_config)
    results = engine.run(
        data=recent_df,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe
    )
    
    # Display results
    print(f"\n" + "="*70)
    print(f" PAPER TRADING SIMULATION RESULTS")
    print(f"="*70)
    print(f"Period: {start_date} to {end_date} (30 days)")
    
    print(f"\nPerformance:")
    print(f"  Initial Capital: ${initial_capital:.2f}")
    print(f"  Final Capital: ${results['final_capital']:.2f}")
    print(f"  Total Return: ${results['total_return']:.2f} ({results['return_pct']:.2f}%)")
    
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winning_trades']} ({results['win_rate']:.1f}%)")
    print(f"  Losing Trades: {results['losing_trades']}")
    
    if results['total_trades'] > 0:
        print(f"\nProfit/Loss:")
        print(f"  Gross Profit: ${results['gross_profit']:.2f}")
        print(f"  Gross Loss: ${results['gross_loss']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Average Trade: ${results['avg_trade_return']:.2f}")
    
    print(f"\n" + "="*70)
    
    # Comparison
    expected_return = 0.5
    expected_win_rate = 55
    
    print(f"\nvs BACKTEST EXPECTATIONS:")
    print(f"  Return: {results['return_pct']:.2f}% (expected: +{expected_return}%)")
    print(f"  Win Rate: {results['win_rate']:.1f}% (expected: {expected_win_rate}%)")
    
    if results['return_pct'] >= expected_return * 0.5:
        print(f"  [OK] Performance within acceptable range!")
    else:
        print(f"  [WARNING] Performance below expectations")
    
    # Next steps
    print(f"\nNEXT STEPS:")
    if results['return_pct'] > 0:
        print(f"  [OK] Paper trading successful!")
        print(f"  [OK] Strategy validated on recent data")
        print(f"  [OK] Ready for live deployment with ${initial_capital}")
        print(f"\n  To deploy live:")
        print(f"    1. Start with ${initial_capital} real capital")
        print(f"    2. Use weighted ensemble mode")
        print(f"    3. Monitor performance daily")
        print(f"    4. Retrain ML models weekly")
    else:
        print(f"  [WARNING] Consider optimizing strategy parameters")
        print(f"  [INFO] Try RSI+MACD on ETHUSDT 4h for conservative approach")
    
    print(f"\n" + "="*70)
    print(f"Paper trading simulation complete!")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    run_paper_trading()
