"""
Paper Trading Simulation - No Unicode
Completely ASCII-safe version for Windows terminal
"""

import sys
import os
from pathlib import Path

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Disable loguru emojis
os.environ['LOGURU_FORMAT'] = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}'

import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.ensemble import EnsembleStrategy
from backtesting.backtest_engine import BacktestEngine


def run_paper_trading():
    """Run paper trading simulation"""
    
    print("\n" + "="*70)
    print(" PAPER TRADING SIMULATION - ENSEMBLE STRATEGY")
    print(" Quick validation using recent 30 days of data")
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
    
    try:
        strategy = EnsembleStrategy(config)
        print(f"  [OK] Ensemble strategy ready (weighted mode)")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize strategy: {e}")
        return
    
    # Run backtest
    print(f"\nRunning paper trading simulation...")
    print(f"  (Simulating real-time trading on recent data)")
    print(f"  This may take 30-60 seconds...")
    print("\n" + "="*70)
    
    backtest_config = {
        'initial_capital': initial_capital,
        'position_size_pct': 100,
        'leverage': leverage,
        'commission_pct': 0.04,
        'slippage_pct': 0.1
    }
    
    try:
        engine = BacktestEngine(backtest_config)
        results = engine.run(
            data=recent_df,
            strategy=strategy,
            symbol=symbol
        )
    except Exception as e:
        print(f"\n[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print(f"\n" + "="*70)
    print(f" PAPER TRADING SIMULATION RESULTS")
    print(f"="*70)
    print(f"Period: {start_date} to {end_date} (30 days)")
    
    print(f"\nPerformance:")
    print(f"  Initial Capital: ${initial_capital:.2f}")
    print(f"  Final Capital: ${results['final_equity']:.2f}")
    total_return = results['total_return']
    total_return_usd = results['final_equity'] - initial_capital
    print(f"  Total Return: ${total_return_usd:.2f} ({total_return:.2f}%)")
    
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winning_trades']} ({results['win_rate']:.1f}%)")
    print(f"  Losing Trades: {results['losing_trades']}")
    
    if results['total_trades'] > 0:
        print(f"\nProfit/Loss:")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Total Commission: ${results['total_commission']:.2f}")
        avg_pnl = results['total_pnl'] / results['total_trades']
        print(f"  Average Trade: ${avg_pnl:.2f}")
        
        print(f"\nRisk Metrics:")
        # Calculate max drawdown from equity curve
        equity_values = [e[1] for e in results['equity_curve']]
        peak = equity_values[0]
        max_dd = 0
        for val in equity_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        print(f"  Max Drawdown: {max_dd:.2f}%")
        
        # Show some sample trades
        if len(results['trades']) > 0:
            print(f"\nSample Trades (first 5):")
            for i, trade in enumerate(results['trades'][:5], 1):
                entry = trade.entry_price
                exit_p = trade.exit_price
                pnl = trade.pnl
                pnl_pct = (exit_p - entry) / entry * 100
                print(f"  {i}. Entry ${entry:.2f} -> Exit ${exit_p:.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    else:
        print(f"\n[WARNING] No trades executed!")
        print(f"  Possible reasons:")
        print(f"    - Confidence threshold too high (0.6)")
        print(f"    - Market conditions not favorable")
        print(f"    - Strategy signals too conservative")
    
    print(f"\n" + "="*70)
    
    # Comparison
    expected_return = 0.5
    expected_win_rate = 55
    
    print(f"\nvs BACKTEST EXPECTATIONS:")
    print(f"  Return: {total_return:.2f}% (expected: +{expected_return}%)")
    print(f"  Win Rate: {results['win_rate']:.1f}% (expected: {expected_win_rate}%)")
    
    if results['total_trades'] == 0:
        print(f"  [WARNING] No trades - cannot compare")
    elif total_return >= expected_return * 0.5:
        print(f"  [OK] Performance within acceptable range!")
    else:
        print(f"  [WARNING] Performance below expectations")
    
    # Next steps
    print(f"\nNEXT STEPS:")
    if total_return > 0 and results['total_trades'] > 0:
        print(f"  [OK] Paper trading successful!")
        print(f"  [OK] Strategy validated on recent data")
        print(f"  [OK] Ready for live deployment with ${initial_capital}")
        print(f"\n  To deploy live:")
        print(f"    1. Start with ${initial_capital} real capital")
        print(f"    2. Use weighted ensemble mode")
        print(f"    3. Monitor performance daily")
        print(f"    4. Retrain ML models weekly")
    elif results['total_trades'] == 0:
        print(f"  [ACTION] Lower confidence threshold:")
        print(f"    config['confidence_threshold'] = 0.5  # was 0.6")
        print(f"  [ACTION] Or try more aggressive mode:")
        print(f"    config['ensemble_mode'] = 'confidence'  # was weighted")
    else:
        print(f"  [WARNING] Consider optimizing strategy parameters")
        print(f"  [INFO] Try RSI+MACD on ETHUSDT 4h for conservative approach")
    
    print(f"\n" + "="*70)
    print(f"Paper trading simulation complete!")
    print(f"="*70 + "\n")
    
    return results


if __name__ == "__main__":
    print("\nStarting paper trading simulation...")
    print("Please wait, this may take 1-2 minutes...\n")
    
    try:
        results = run_paper_trading()
        if results:
            print("\n[SUCCESS] Simulation completed successfully!")
        else:
            print("\n[ERROR] Simulation failed or no results")
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Simulation cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
