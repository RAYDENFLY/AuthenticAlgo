"""
Base Trader Class for AsterDEX Competition
Handles connection, data fetching, and trade execution
"""

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.logger import setup_logger
from core.config import Config
from tpsl_strategy import TPSLStrategy
from asterdex_client import AsterDEXClient

logger = setup_logger()


class BaseTrader:
    """Base class for all trading strategies"""
    
    def __init__(self, strategy_name: str, capital: float = 10.0):
        self.strategy_name = strategy_name
        self.capital = capital
        self.initial_capital = capital
        self.trades = []
        self.current_position = None
        self.completed_trades = 0
        self.max_trades = 10
        
        # TP/SL Strategy Manager
        self.tpsl = TPSLStrategy()
        
        # AsterDEX Real API Client
        self.client = AsterDEXClient()
        
        # Available symbols (will be fetched from API)
        self.symbols = []
        
        # Performance tracking
        self.win_count = 0
        self.loss_count = 0
        self.total_return = 0.0
        
        logger.info(f"🤖 Initialized {strategy_name}")
        logger.info(f"   Capital: ${capital}")
        logger.info(f"   Max trades: {self.max_trades}")
    
    async def connect_asterdex(self):
        """Connect to AsterDEX Real API"""
        try:
            logger.info(f"📡 [{self.strategy_name}] Connecting to AsterDEX API...")
            
            # Test connection
            results = await self.client.test_connection()
            
            if all(results.values()):
                logger.info(f"✅ [{self.strategy_name}] Connected to AsterDEX!")
                
                # Fetch available symbols
                symbols = await self.client.get_symbols()
                
                # Filter for USDT pairs with good liquidity
                self.symbols = [s for s in symbols if s.endswith('USDT')][:20]  # Top 20
                
                logger.info(f"   Available symbols: {len(self.symbols)}")
                logger.info(f"   Top symbols: {', '.join(self.symbols[:5])}")
                
                return True
            else:
                logger.error(f"❌ [{self.strategy_name}] Connection test failed")
                logger.error(f"   Results: {results}")
                return False
                
        except Exception as e:
            logger.error(f"❌ [{self.strategy_name}] Connection failed: {e}")
            return False
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV data from AsterDEX Real API
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1500)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch real data from AsterDEX
            df = await self.client.get_klines(symbol, timeframe, limit)
            
            if df is None or df.empty:
                logger.warning(f"⚠️ [{self.strategy_name}] No data for {symbol}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ [{self.strategy_name}] Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def screen_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Screen symbols and return best one
        Override this in subclasses
        """
        raise NotImplementedError("Subclass must implement screen_symbols()")
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Generate trading signal
        Override this in subclasses
        """
        raise NotImplementedError("Subclass must implement generate_signal()")
    
    def calculate_leverage(self, signal: Dict) -> int:
        """
        Calculate leverage based on confidence
        Override this in subclasses for dynamic leverage
        """
        # Default: fixed 10x
        return 10
    
    async def execute_trade(self, symbol: str, signal: Dict):
        """Execute trade based on signal with TP/SL levels"""
        try:
            if signal['direction'] == 'hold':
                return
            
            # Get current data for TP/SL calculation
            df = await self.fetch_ohlcv(symbol)
            if df.empty:
                return
            
            current_price = signal.get('price', float(df['close'].iloc[-1]))
            confidence = signal.get('confidence', 0.5)
            
            # Calculate TP/SL levels
            tp_sl_levels = self.tpsl.calculate_tp_sl_levels(
                entry_price=current_price,
                direction=signal['direction'],
                df=df,
                confidence=confidence
            )
            
            # Calculate optimal leverage based on SL distance
            leverage = self.tpsl.calculate_optimal_leverage(tp_sl_levels, max_risk_pct=0.02)
            leverage = max(5, min(125, leverage))  # Clamp to 5x-125x
            
            # Position size
            position_size = self.capital * leverage
            
            # Record trade entry
            trade = {
                'entry_time': datetime.now(),
                'symbol': symbol,
                'direction': signal['direction'],
                'entry_price': current_price,
                'leverage': leverage,
                'position_size': position_size,
                'capital': self.capital,
                'confidence': confidence,
                'tp_sl_levels': tp_sl_levels,
                'position_remaining': 1.0,  # 100% of position still open
                'partial_closes': []
            }
            
            self.current_position = trade
            
            # Format and log TP/SL message
            score = int(confidence * 100)
            tp_sl_msg = self.tpsl.format_tp_sl_message(symbol, tp_sl_levels, score)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 [{self.strategy_name}] OPENED {signal['direction'].upper()}")
            logger.info(tp_sl_msg)
            logger.info(f"   Leverage: {leverage}x (optimal for 2% risk)")
            logger.info(f"   Position: ${position_size:.2f}")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"❌ [{self.strategy_name}] Trade execution failed: {e}")
    
    async def monitor_position(self):
        """Monitor position and check for TP/SL hits"""
        try:
            if not self.current_position:
                return False
            
            pos = self.current_position
            symbol = pos['symbol']
            tp_sl_levels = pos['tp_sl_levels']
            
            # Simulate price movement (in production, get real price)
            df = await self.fetch_ohlcv(symbol, limit=10)
            if df.empty:
                return False
            
            current_price = float(df['close'].iloc[-1])
            
            # Check if TP/SL hit
            hit_result = self.tpsl.check_tp_sl_hit(
                current_price=current_price,
                levels=tp_sl_levels,
                position_remaining=pos['position_remaining']
            )
            
            if hit_result:
                # Close portion of position
                close_pct = hit_result['close_pct']
                pnl_pct = hit_result['pnl_pct']
                
                # Calculate PnL for this close
                capital_affected = pos['capital'] * pos['position_remaining']
                pnl_amount = capital_affected * pnl_pct * pos['leverage'] * close_pct
                
                # Update capital
                self.capital += pnl_amount
                
                # Update position remaining
                pos['position_remaining'] -= close_pct
                
                # Record partial close
                partial_close = {
                    'time': datetime.now(),
                    'type': hit_result['hit_type'],
                    'price': current_price,
                    'close_pct': close_pct,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_amount': pnl_amount,
                    'reason': hit_result['reason']
                }
                pos['partial_closes'].append(partial_close)
                
                # Log the hit
                emoji = "✅" if pnl_amount > 0 else "❌"
                logger.info(f"\n{emoji} [{self.strategy_name}] {hit_result['reason']}")
                logger.info(f"   Closed: {close_pct*100:.0f}% of position")
                logger.info(f"   PnL: {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
                logger.info(f"   Capital: ${self.capital:.2f}")
                logger.info(f"   Remaining: {pos['position_remaining']*100:.0f}%\n")
                
                # If position fully closed, finalize trade
                if pos['position_remaining'] <= 0.01:  # Closed (allow rounding error)
                    await self.finalize_trade()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ [{self.strategy_name}] Position monitoring failed: {e}")
            return False
    
    async def finalize_trade(self):
        """Finalize trade and record results"""
        try:
            if not self.current_position:
                return
            
            pos = self.current_position
            pos['exit_time'] = datetime.now()
            pos['new_capital'] = self.capital
            pos['total_pnl'] = self.capital - pos['capital']
            pos['total_pnl_pct'] = (pos['total_pnl'] / pos['capital']) * 100
            
            self.trades.append(pos)
            self.completed_trades += 1
            
            # Update win/loss
            if pos['total_pnl'] > 0:
                self.win_count += 1
                status = "🎉 WIN"
            else:
                self.loss_count += 1
                status = "💔 LOSS"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"{status} [{self.strategy_name}] Trade #{self.completed_trades} COMPLETE")
            logger.info(f"   Symbol: {pos['symbol']}")
            logger.info(f"   Direction: {pos['direction'].upper()}")
            logger.info(f"   Entry: ${pos['entry_price']:.4f}")
            logger.info(f"   Partial Closes: {len(pos['partial_closes'])}")
            
            for i, close in enumerate(pos['partial_closes'], 1):
                logger.info(f"      {i}. {close['type'].upper()}: ${close['price']:.4f} ({close['pnl_pct']:+.2f}%)")
            
            logger.info(f"   Total PnL: {pos['total_pnl_pct']:+.2f}% (${pos['total_pnl']:+.2f})")
            logger.info(f"   Capital: ${pos['capital']:.2f} → ${self.capital:.2f}")
            logger.info(f"{'='*60}\n")
            
            self.current_position = None
            
        except Exception as e:
            logger.error(f"❌ [{self.strategy_name}] Failed to finalize trade: {e}")
    
    async def close_position(self, current_price: float, reason: str = "Target reached"):
        """Legacy method - now uses monitor_position instead"""
        # This method is deprecated but kept for compatibility
        logger.warning("Using legacy close_position - should use monitor_position instead")
    
    async def run_single_trade(self):
        """Execute one complete trade cycle"""
        try:
            # Fetch data for all symbols
            logger.info(f"\n📊 [{self.strategy_name}] Screening {len(self.symbols)} symbols...")
            data_dict = {}
            
            for symbol in self.symbols:
                df = await self.fetch_ohlcv(symbol)
                if not df.empty:
                    data_dict[symbol] = df
            
            # Screen for best symbol
            best_symbol = self.screen_symbols(data_dict)
            
            if not best_symbol:
                logger.warning(f"⚠️ [{self.strategy_name}] No suitable symbol found")
                return False
            
            logger.info(f"🎯 [{self.strategy_name}] Selected: {best_symbol}")
            
            # Generate signal
            df = data_dict[best_symbol]
            signal = self.generate_signal(df, best_symbol)
            
            if signal['direction'] == 'hold':
                logger.info(f"⏸️ [{self.strategy_name}] No clear signal, skipping...")
                return False
            
            # Execute trade
            await self.execute_trade(best_symbol, signal)
            
            # Monitor position for TP/SL hits (continuous monitoring)
            logger.info(f"⏳ [{self.strategy_name}] Monitoring position until TP/SL hit...")
            
            # Keep monitoring until position is closed
            while self.current_position:
                await asyncio.sleep(0.5)  # Check every 0.5 seconds
                
                closed = await self.monitor_position()
                if closed:
                    break  # Position fully closed via TP/SL
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [{self.strategy_name}] Trade failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def run_competition(self):
        """Run full trading competition"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏁 STARTING: {self.strategy_name}")
        logger.info(f"{'='*80}\n")
        
        # Connect
        if not await self.connect_asterdex():
            return
        
        # Execute trades
        while self.completed_trades < self.max_trades:
            logger.info(f"\n--- Trade {self.completed_trades + 1}/{self.max_trades} ---")
            
            success = await self.run_single_trade()
            
            if not success:
                logger.warning("Retrying...")
                await asyncio.sleep(1)
                continue
            
            # Small delay between trades
            if self.completed_trades < self.max_trades:
                await asyncio.sleep(1)
        
        # Final report
        self.print_final_report()
        self.save_report()
    
    def print_final_report(self):
        """Print final performance report"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 FINAL REPORT: {self.strategy_name}")
        logger.info(f"{'='*80}")
        
        win_rate = (self.win_count / self.completed_trades * 100) if self.completed_trades > 0 else 0
        
        logger.info(f"\n📊 Performance:")
        logger.info(f"   Trades: {self.completed_trades}")
        logger.info(f"   Wins: {self.win_count}")
        logger.info(f"   Losses: {self.loss_count}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        
        logger.info(f"\n💰 Returns:")
        logger.info(f"   Initial: ${self.initial_capital:.2f}")
        logger.info(f"   Final: ${self.capital:.2f}")
        logger.info(f"   Profit: ${self.capital - self.initial_capital:+.2f}")
        logger.info(f"   ROI: {(self.capital/self.initial_capital - 1)*100:+.2f}%")
        
        if self.trades:
            avg_leverage = np.mean([t['leverage'] for t in self.trades])
            logger.info(f"\n📈 Stats:")
            logger.info(f"   Avg Leverage: {avg_leverage:.1f}x")
            logger.info(f"   Avg PnL: {np.mean([t['pnl_pct'] for t in self.trades]):.2f}%")
        
        logger.info(f"\n{'='*80}\n")
    
    def save_report(self):
        """Save detailed report to file"""
        try:
            report_dir = Path(__file__).parent.parent.parent / "Reports" / "benchmark" / "AsterDEX"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name.replace(' ', '_')}_{timestamp}.json"
            filepath = report_dir / filename
            
            report = {
                'strategy': self.strategy_name,
                'timestamp': timestamp,
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'profit': self.capital - self.initial_capital,
                'roi_pct': (self.capital / self.initial_capital - 1) * 100,
                'total_trades': self.completed_trades,
                'wins': self.win_count,
                'losses': self.loss_count,
                'win_rate_pct': (self.win_count / self.completed_trades * 100) if self.completed_trades > 0 else 0,
                'trades': self.trades
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"💾 Report saved: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
