"""
Paper Trading with Lower Threshold
Testing with 0.5 confidence threshold (from 0.6)
"""

import sys
import os
from pathlib import Path

# Force UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.environ['LOGURU_FORMAT'] = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}'

import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.ensemble import EnsembleStrategy
from backtesting.backtest_engine import BacktestEngine


def run_paper_trading_v2():
    """Run paper trading with lower threshold"""
    
    print("\n" + "="*70)
    print(" PAPER TRADING V2 - LOWER THRESHOLD TEST")
    print(" Confidence threshold: 0.5 (was 0.6)")
    print("="*70)
    
    symbol = "ETHUSDT"
    timeframe = "1h"
    initial_capital = 5.0
    leverage = 10
    threshold = 0.5  # LOWERED from 0.6
    
    print(f"\nConfiguration:")
    print(f"  Strategy: Ensemble (Weighted Mode)")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital}")
    print(f"  Leverage: {leverage}x")
    print(f"  Confidence Threshold: {threshold} (LOWERED)")
    
    # Load data
    print(f"\nLoading data...")
    data_file = f"data/historical/asterdex_{symbol}_{timeframe}_20250805_to_20251103.csv"
    
    try:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_df = df.tail(720)
        print(f"  [OK] Using last {len(recent_df)} candles")
        start_date = recent_df['timestamp'].iloc[0].strftime('%Y-%m-%d')
        end_date = recent_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        print(f"  Period: {start_date} to {end_date}")
    except FileNotFoundError:
        print(f"  [ERROR] Data file not found")
        return None
    
    # Initialize strategy with LOWER threshold
    print(f"\nInitializing Ensemble Strategy (threshold={threshold})...")
    config = {
        'ensemble_mode': 'weighted',
        'confidence_threshold': threshold,  # LOWERED
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
        print(f"  [OK] Strategy ready with threshold={threshold}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None
    
    # Run backtest
    print(f"\nRunning simulation (may take 1-2 minutes)...")
    print("="*70)
    
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
        print(f"\n[ERROR] {e}")
        return None
    
    # Display results
    print(f"\n" + "="*70)
    print(f" PAPER TRADING RESULTS (Threshold={threshold})")
    print(f"="*70)
    
    total_return = results['total_return']
    total_return_usd = results['final_equity'] - initial_capital
    
    print(f"\nPerformance:")
    print(f"  Initial Capital: ${initial_capital:.2f}")
    print(f"  Final Capital: ${results['final_equity']:.2f}")
    print(f"  Total Return: ${total_return_usd:.2f} ({total_return:.2f}%)")
    
    print(f"\nTrading:")
    print(f"  Total Trades: {results['total_trades']}")
    
    if results['total_trades'] > 0:
        print(f"  Winning: {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"  Losing: {results['losing_trades']}")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Commission: ${results['total_commission']:.2f}")
        print(f"  Net PnL: ${results['total_pnl'] - results['total_commission']:.2f}")
        
        # Show all trades
        print(f"\nAll Trades:")
        for i, trade in enumerate(results['trades'], 1):
            entry = trade.entry_price
            exit_p = trade.exit_price
            pnl = trade.pnl
            pnl_pct = (exit_p - entry) / entry * 100
            status = "[WIN]" if pnl > 0 else "[LOSS]"
            print(f"  {i}. {status} ${entry:.2f} -> ${exit_p:.2f} | ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        # Risk metrics
        equity_values = [e[1] for e in results['equity_curve']]
        peak = equity_values[0]
        max_dd = 0
        for val in equity_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        print(f"\nRisk:")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        
        # Verdict
        print(f"\n" + "="*70)
        if total_return > 0:
            print(f" [SUCCESS] Strategy PROFITABLE on recent 30 days!")
            print(f" Return: +{total_return:.2f}% (${total_return_usd:+.2f})")
            print(f" Ready for LIVE deployment with ${initial_capital}")
        elif total_return == 0:
            print(f" [NEUTRAL] Break-even performance")
            print(f" Consider: Try different symbol or timeframe")
        else:
            print(f" [WARNING] Strategy showing losses")
            print(f" Return: {total_return:.2f}% (${total_return_usd:.2f})")
            print(f" Recommendation: Optimize parameters or try RSI+MACD standalone")
        print(f"="*70)
    else:
        print(f"\n[WARNING] Still no trades with threshold={threshold}")
        print(f"\nTry:")
        print(f"  1. Lower threshold to 0.4")
        print(f"  2. Use 'confidence' mode instead of 'weighted'")
        print(f"  3. Try RSI+MACD standalone on ETHUSDT 4h")
    
    print()
    return results


if __name__ == "__main__":
    print("\nPaper Trading V2 - Adjusted Threshold")
    print("Testing with more lenient entry conditions\n")
    
    try:
        results = run_paper_trading_v2()
        if results and results['total_trades'] > 0:
            print("[SUCCESS] Found trading opportunities!")
        elif results and results['total_trades'] == 0:
            print("[INFO] No trades - market conditions may be sideways")
        else:
            print("[ERROR] Simulation failed")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
