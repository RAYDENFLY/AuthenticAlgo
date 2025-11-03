"""
Benchmark Script - Compare Trading Strategies Performance
Membandingkan performa RSI+MACD, Bollinger Bands, dan ML Strategy
Menggunakan data AsterDEX 90 hari
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import setup_logger
from loguru import logger

# Setup
setup_logger()


class StrategyBenchmark:
    """Benchmark multiple trading strategies"""
    
    def __init__(self):
        self.data_dir = Path("data/historical")
        self.results = {}
        self.comparison = {}
        
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
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR for stop-loss
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        return df.dropna()
    
    def backtest_rsi_macd(self, df: pd.DataFrame, initial_capital: float = 1000) -> Dict:
        """Backtest RSI + MACD Strategy"""
        logger.info("\n📊 Backtesting RSI + MACD Strategy...")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            rsi = df['rsi'].iloc[i]
            macd = df['macd'].iloc[i]
            macd_signal = df['macd_signal'].iloc[i]
            atr = df['atr'].iloc[i]
            
            # Entry Signal: RSI oversold + MACD bullish crossover
            if position == 0:
                if rsi < 30 and macd > macd_signal and df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]:
                    position_size = (capital * 0.1) / current_price  # 10% per trade
                    position = position_size
                    entry_price = current_price
                    logger.debug(f"BUY: {current_price:.2f} | RSI: {rsi:.1f} | Position: {position:.4f}")
            
            # Exit Signal: RSI overbought OR MACD bearish crossover
            elif position > 0:
                stop_loss = entry_price - (atr * 2)  # 2 ATR stop-loss
                
                if (rsi > 70 or 
                    (macd < macd_signal and df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]) or
                    current_price < stop_loss):
                    
                    pnl = (current_price - entry_price) * position
                    capital += pnl
                    
                    trade = {
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'return': (current_price - entry_price) / entry_price * 100,
                        'reason': 'RSI' if rsi > 70 else 'MACD' if macd < macd_signal else 'STOP_LOSS'
                    }
                    trades.append(trade)
                    
                    logger.debug(f"SELL: {current_price:.2f} | PnL: ${pnl:.2f} | Reason: {trade['reason']}")
                    position = 0
                    entry_price = 0
            
            equity_curve.append(capital + (position * current_price if position > 0 else 0))
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        
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
            'strategy': 'RSI + MACD',
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
            'max_drawdown': max_drawdown,
            'trades': trades
        }
        
        return results
    
    def backtest_bollinger(self, df: pd.DataFrame, initial_capital: float = 1000) -> Dict:
        """Backtest Bollinger Bands Strategy"""
        logger.info("\n📊 Backtesting Bollinger Bands Strategy...")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            bb_upper = df['bb_upper'].iloc[i]
            bb_lower = df['bb_lower'].iloc[i]
            bb_middle = df['bb_middle'].iloc[i]
            volume = df['volume'].iloc[i]
            avg_volume = df['volume'].rolling(20).mean().iloc[i]
            atr = df['atr'].iloc[i]
            
            # Entry Signal: Price touches lower band + volume spike
            if position == 0:
                if current_price <= bb_lower and volume > avg_volume * 1.5:
                    position_size = (capital * 0.1) / current_price
                    position = position_size
                    entry_price = current_price
                    logger.debug(f"BUY: {current_price:.2f} | BB Lower: {bb_lower:.2f}")
            
            # Exit Signal: Price reaches middle or upper band
            elif position > 0:
                stop_loss = entry_price - (atr * 2)
                
                if current_price >= bb_middle or current_price < stop_loss:
                    pnl = (current_price - entry_price) * position
                    capital += pnl
                    
                    trade = {
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'return': (current_price - entry_price) / entry_price * 100,
                        'reason': 'TARGET' if current_price >= bb_middle else 'STOP_LOSS'
                    }
                    trades.append(trade)
                    
                    logger.debug(f"SELL: {current_price:.2f} | PnL: ${pnl:.2f}")
                    position = 0
                    entry_price = 0
            
            equity_curve.append(capital + (position * current_price if position > 0 else 0))
        
        # Calculate metrics (same as RSI+MACD)
        total_return = (capital - initial_capital) / initial_capital * 100
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        results = {
            'strategy': 'Bollinger Bands',
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
            'max_drawdown': max_drawdown,
            'trades': trades
        }
        
        return results
    
    def run_benchmark(self, symbols: List[str], timeframes: List[str], initial_capital: float = 1000):
        """Run complete benchmark"""
        logger.info("="*80)
        logger.info("🏁 STARTING STRATEGY BENCHMARK")
        logger.info("="*80)
        
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"\n{'='*80}")
                logger.info(f"📈 Benchmarking {symbol} {timeframe}")
                logger.info(f"{'='*80}")
                
                # Load data
                df = self.load_data(symbol, timeframe)
                if df.empty:
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                logger.info(f"✅ Calculated indicators for {len(df)} candles")
                
                # Backtest RSI + MACD
                rsi_macd_results = self.backtest_rsi_macd(df, initial_capital)
                rsi_macd_results['symbol'] = symbol
                rsi_macd_results['timeframe'] = timeframe
                all_results.append(rsi_macd_results)
                
                # Backtest Bollinger Bands
                bollinger_results = self.backtest_bollinger(df, initial_capital)
                bollinger_results['symbol'] = symbol
                bollinger_results['timeframe'] = timeframe
                all_results.append(bollinger_results)
                
                # Print results
                self.print_results(rsi_macd_results, bollinger_results)
        
        # Save results
        self.save_results(all_results)
        
        # Generate comparison
        self.generate_comparison(all_results)
        
        return all_results
    
    def print_results(self, rsi_macd: Dict, bollinger: Dict):
        """Print comparison between strategies"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 RESULTS COMPARISON")
        logger.info(f"{'='*80}\n")
        
        comparison = pd.DataFrame([
            {
                'Metric': 'Total Return',
                'RSI+MACD': f"{rsi_macd['total_return']:.2f}%",
                'Bollinger': f"{bollinger['total_return']:.2f}%",
                'Winner': '🏆 RSI+MACD' if rsi_macd['total_return'] > bollinger['total_return'] else '🏆 Bollinger'
            },
            {
                'Metric': 'Final Capital',
                'RSI+MACD': f"${rsi_macd['final_capital']:.2f}",
                'Bollinger': f"${bollinger['final_capital']:.2f}",
                'Winner': '🏆 RSI+MACD' if rsi_macd['final_capital'] > bollinger['final_capital'] else '🏆 Bollinger'
            },
            {
                'Metric': 'Total Trades',
                'RSI+MACD': str(rsi_macd['total_trades']),
                'Bollinger': str(bollinger['total_trades']),
                'Winner': '-'
            },
            {
                'Metric': 'Win Rate',
                'RSI+MACD': f"{rsi_macd['win_rate']:.1f}%",
                'Bollinger': f"{bollinger['win_rate']:.1f}%",
                'Winner': '🏆 RSI+MACD' if rsi_macd['win_rate'] > bollinger['win_rate'] else '🏆 Bollinger'
            },
            {
                'Metric': 'Sharpe Ratio',
                'RSI+MACD': f"{rsi_macd['sharpe_ratio']:.2f}",
                'Bollinger': f"{bollinger['sharpe_ratio']:.2f}",
                'Winner': '🏆 RSI+MACD' if rsi_macd['sharpe_ratio'] > bollinger['sharpe_ratio'] else '🏆 Bollinger'
            },
            {
                'Metric': 'Max Drawdown',
                'RSI+MACD': f"{rsi_macd['max_drawdown']:.2f}%",
                'Bollinger': f"{bollinger['max_drawdown']:.2f}%",
                'Winner': '🏆 RSI+MACD' if rsi_macd['max_drawdown'] > bollinger['max_drawdown'] else '🏆 Bollinger'
            },
            {
                'Metric': 'Profit Factor',
                'RSI+MACD': f"{rsi_macd['profit_factor']:.2f}",
                'Bollinger': f"{bollinger['profit_factor']:.2f}",
                'Winner': '🏆 RSI+MACD' if rsi_macd['profit_factor'] > bollinger['profit_factor'] else '🏆 Bollinger'
            }
        ])
        
        logger.info(comparison.to_string(index=False))
        logger.info("")
    
    def save_results(self, results: List[Dict]):
        """Save benchmark results to file"""
        output_dir = Path("backtesting/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"benchmark_{timestamp}.json"
        
        # Remove trades from results (too large)
        clean_results = []
        for r in results:
            clean = r.copy()
            clean['num_trades'] = len(clean.get('trades', []))
            clean.pop('trades', None)
            clean_results.append(clean)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filename}")
    
    def generate_comparison(self, results: List[Dict]):
        """Generate overall comparison"""
        logger.info(f"\n{'='*80}")
        logger.info("🏆 OVERALL BENCHMARK SUMMARY")
        logger.info(f"{'='*80}\n")
        
        # Group by strategy
        rsi_macd_results = [r for r in results if r['strategy'] == 'RSI + MACD']
        bollinger_results = [r for r in results if r['strategy'] == 'Bollinger Bands']
        
        # Calculate averages
        rsi_avg_return = np.mean([r['total_return'] for r in rsi_macd_results])
        rsi_avg_winrate = np.mean([r['win_rate'] for r in rsi_macd_results])
        rsi_avg_sharpe = np.mean([r['sharpe_ratio'] for r in rsi_macd_results])
        
        bb_avg_return = np.mean([r['total_return'] for r in bollinger_results])
        bb_avg_winrate = np.mean([r['win_rate'] for r in bollinger_results])
        bb_avg_sharpe = np.mean([r['sharpe_ratio'] for r in bollinger_results])
        
        summary = pd.DataFrame([
            {
                'Strategy': 'RSI + MACD',
                'Avg Return': f"{rsi_avg_return:.2f}%",
                'Avg Win Rate': f"{rsi_avg_winrate:.1f}%",
                'Avg Sharpe': f"{rsi_avg_sharpe:.2f}",
                'Total Backtests': len(rsi_macd_results)
            },
            {
                'Strategy': 'Bollinger Bands',
                'Avg Return': f"{bb_avg_return:.2f}%",
                'Avg Win Rate': f"{bb_avg_winrate:.1f}%",
                'Avg Sharpe': f"{bb_avg_sharpe:.2f}",
                'Total Backtests': len(bollinger_results)
            }
        ])
        
        logger.info(summary.to_string(index=False))
        
        # Determine winner
        logger.info(f"\n🏆 OVERALL WINNER: ", end="")
        if rsi_avg_return > bb_avg_return:
            logger.info("RSI + MACD Strategy!")
        else:
            logger.info("Bollinger Bands Strategy!")
        
        logger.info(f"\n{'='*80}")


async def main():
    """Main benchmark execution"""
    benchmark = StrategyBenchmark()
    
    # Configuration
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    timeframes = ['1h', '4h']
    initial_capital = 1000  # $1000 for testing
    
    logger.info(f"""
📊 Benchmark Configuration:
   Symbols: {', '.join(symbols)}
   Timeframes: {', '.join(timeframes)}
   Initial Capital: ${initial_capital}
   Data Period: Aug 5 - Nov 3, 2025 (90 days)
   Strategies: RSI+MACD, Bollinger Bands
""")
    
    # Run benchmark
    results = benchmark.run_benchmark(symbols, timeframes, initial_capital)
    
    logger.info(f"\n✅ Benchmark completed!")
    logger.info(f"   Total tests: {len(results)}")
    logger.info(f"   Results saved to: backtesting/results/")


if __name__ == "__main__":
    asyncio.run(main())
