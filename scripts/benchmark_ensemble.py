"""
Ensemble Strategy Benchmark
Compare Ensemble vs Individual Strategies
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import setup_logger
from loguru import logger
from strategies.ensemble import EnsembleStrategy

# Setup
setup_logger()


class EnsembleBenchmark:
    """Benchmark ensemble strategy against individual strategies"""
    
    def __init__(self):
        self.data_dir = Path("data/historical")
        self.results_dir = Path("Reports/benchmark")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load historical data"""
        filename = f"asterdex_{symbol}_{timeframe}_20250805_to_20251103.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        logger.info(f"✅ Loaded {len(df)} candles from {filename}")
        return df
    
    def backtest_ensemble(self, df: pd.DataFrame, mode: str, symbol: str, timeframe: str, initial_capital: float = 1000) -> Dict:
        """Backtest ensemble strategy"""
        logger.info(f"\n📊 Backtesting Ensemble ({mode}) - {symbol} {timeframe}")
        
        # Create ensemble strategy
        config = {
            'ensemble_mode': mode,
            'symbol': symbol,
            'timeframe': timeframe,
            'use_ml': True,
            'min_agreement': 0.5,
            'confidence_threshold': 0.6,
            'strategy_weights': {
                'rsi_macd': 0.35,
                'random_forest': 0.30,
                'xgboost': 0.20,
                'bollinger': 0.15
            }
        }
        
        strategy = EnsembleStrategy(config)
        
        # Backtest
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(50, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df.iloc[i]['close']
            
            # Generate signal
            signal_data = strategy.generate_signal(current_data)
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Entry
            if position == 0 and signal == 'BUY' and confidence >= 0.6:
                position_size = (capital * 0.1) / current_price
                position = position_size
                entry_price = current_price
                
                logger.debug(f"BUY at {current_price:.2f} - {signal_data['reason']}")
            
            # Exit
            elif position > 0:
                stop_loss = strategy.get_stop_loss(entry_price, current_data)
                
                should_exit = (
                    signal == 'SELL' or 
                    current_price < stop_loss
                )
                
                if should_exit:
                    pnl = (current_price - entry_price) * position
                    capital += pnl
                    
                    trade = {
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'return': (current_price - entry_price) / entry_price * 100,
                        'reason': 'SELL signal' if signal == 'SELL' else 'Stop loss'
                    }
                    trades.append(trade)
                    
                    logger.debug(f"SELL at {current_price:.2f} - PnL: ${pnl:.2f}")
                    
                    position = 0
                    entry_price = 0
            
            # Update equity curve
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        
        # Sharpe Ratio
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'mode': mode,
            'total_return': total_return,
            'final_capital': capital,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
        
        logger.info(f"✅ Return: {total_return:+.2f}% | Trades: {len(trades)} | Win Rate: {win_rate:.1f}%")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run benchmark on all symbols and modes"""
        logger.info("="*80)
        logger.info("🎯 ENSEMBLE STRATEGY COMPREHENSIVE BENCHMARK")
        logger.info("="*80)
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        timeframes = ['1h', '4h']
        modes = ['voting', 'weighted', 'unanimous', 'confidence']
        
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 Testing {symbol} {timeframe}")
                logger.info(f"{'='*80}")
                
                # Load data
                df = self.load_data(symbol, timeframe)
                if df.empty:
                    continue
                
                # Test each ensemble mode
                for mode in modes:
                    result = self.backtest_ensemble(df, mode, symbol, timeframe)
                    if result:
                        all_results.append(result)
                
                logger.info("")
        
        # Generate summary
        self.generate_summary(all_results)
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def generate_summary(self, results: List[Dict]):
        """Generate overall summary"""
        logger.info(f"\n{'='*80}")
        logger.info("🏆 ENSEMBLE MODES COMPARISON")
        logger.info(f"{'='*80}\n")
        
        # Group by mode
        modes_summary = {}
        for mode in ['voting', 'weighted', 'unanimous', 'confidence']:
            mode_results = [r for r in results if r['mode'] == mode]
            if mode_results:
                avg_return = np.mean([r['total_return'] for r in mode_results])
                avg_win_rate = np.mean([r['win_rate'] for r in mode_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in mode_results])
                best_return = max([r['total_return'] for r in mode_results])
                
                modes_summary[mode] = {
                    'avg_return': avg_return,
                    'avg_win_rate': avg_win_rate,
                    'avg_sharpe': avg_sharpe,
                    'best_return': best_return
                }
        
        # Print comparison table
        summary_df = pd.DataFrame([
            {
                'Mode': mode.capitalize(),
                'Avg Return': f"{data['avg_return']:+.2f}%",
                'Avg Win Rate': f"{data['avg_win_rate']:.1f}%",
                'Avg Sharpe': f"{data['avg_sharpe']:.2f}",
                'Best Return': f"{data['best_return']:+.2f}%"
            }
            for mode, data in modes_summary.items()
        ])
        
        logger.info(summary_df.to_string(index=False))
        
        # Find best mode
        best_mode = max(modes_summary.items(), key=lambda x: x[1]['avg_return'])
        logger.info(f"\n🏆 BEST MODE: {best_mode[0].upper()}")
        logger.info(f"   Avg Return: {best_mode[1]['avg_return']:+.2f}%")
        logger.info(f"   Avg Win Rate: {best_mode[1]['avg_win_rate']:.1f}%")
        
        # Best individual result
        best_result = max(results, key=lambda x: x['total_return'])
        logger.info(f"\n⭐ BEST INDIVIDUAL RESULT:")
        logger.info(f"   Mode: {best_result['mode']}")
        logger.info(f"   Symbol: {best_result['symbol']} {best_result['timeframe']}")
        logger.info(f"   Return: {best_result['total_return']:+.2f}%")
        logger.info(f"   Win Rate: {best_result['win_rate']:.1f}%")
        logger.info(f"   Sharpe: {best_result['sharpe_ratio']:.2f}")
        
        logger.info(f"\n{'='*80}")
    
    def save_results(self, results: List[Dict]):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"ensemble_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {filename}")
    
    def generate_report(self, results: List[Dict]):
        """Generate detailed markdown report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"ensemble_report_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write("# 🎯 Ensemble Strategy Benchmark Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y %H:%M')}\n")
            f.write(f"**Total Tests**: {len(results)}\n\n")
            
            f.write("## 📊 Results by Mode\n\n")
            
            for mode in ['voting', 'weighted', 'unanimous', 'confidence']:
                mode_results = [r for r in results if r['mode'] == mode]
                if not mode_results:
                    continue
                
                f.write(f"### {mode.capitalize()} Mode\n\n")
                f.write("| Symbol | Timeframe | Return | Win Rate | Sharpe | Trades |\n")
                f.write("|--------|-----------|--------|----------|--------|--------|\n")
                
                for r in mode_results:
                    f.write(f"| {r['symbol']} | {r['timeframe']} | {r['total_return']:+.2f}% | ")
                    f.write(f"{r['win_rate']:.1f}% | {r['sharpe_ratio']:.2f} | {r['total_trades']} |\n")
                
                avg_return = np.mean([r['total_return'] for r in mode_results])
                f.write(f"\n**Average Return**: {avg_return:+.2f}%\n\n")
            
            f.write("## 🏆 Best Configurations\n\n")
            
            best = max(results, key=lambda x: x['total_return'])
            f.write(f"**Best Result**:\n")
            f.write(f"- Mode: {best['mode']}\n")
            f.write(f"- Symbol: {best['symbol']} {best['timeframe']}\n")
            f.write(f"- Return: {best['total_return']:+.2f}%\n")
            f.write(f"- Win Rate: {best['win_rate']:.1f}%\n")
            f.write(f"- Sharpe Ratio: {best['sharpe_ratio']:.2f}\n\n")
        
        logger.info(f"📄 Report saved to: {filename}")


async def main():
    """Main execution"""
    benchmark = EnsembleBenchmark()
    
    logger.info("""
🎯 Ensemble Strategy Benchmark
   Testing 4 modes: voting, weighted, unanimous, confidence
   Combining: RSI+MACD, Bollinger, XGBoost, Random Forest
   Symbols: BTC, ETH, BNB
   Timeframes: 1h, 4h
""")
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate detailed report
    if results:
        benchmark.generate_report(results)
    
    logger.info("\n✅ Ensemble benchmark completed!")


if __name__ == "__main__":
    asyncio.run(main())
