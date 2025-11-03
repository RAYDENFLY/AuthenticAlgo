"""
Paper Trading Demo with Ensemble Strategy

This script demonstrates paper trading with virtual money to validate
the ensemble strategy before deploying with real capital.

Features:
- Real-time market data from AsterDEX
- Ensemble strategy (weighted mode recommended)
- Risk management (stop-loss, position sizing)
- Performance tracking vs backtest results
- Trade logging and metrics

Usage:
    python demo/demo_paper_trading.py
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yaml
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.logger import setup_logger
from data.asterdex_collector import AsterDEXCollector
from strategies.ensemble import EnsembleStrategy
from strategies.rsi_macd import RSIMACDStrategy
from execution.position_sizer import PositionSizer
from risk.risk_manager import RiskManager


class PaperTradingEngine:
    """Paper trading engine for strategy validation"""
    
    def __init__(self, config_path: str = "config/paper_trading.yaml"):
        """Initialize paper trading engine"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paper_config = self.config['paper_trading']
        
        # Setup logger
        setup_logger(self.paper_config['monitoring']['log_level'])
        
        # Initialize components
        self.capital = self.paper_config['initial_capital']
        self.initial_capital = self.capital
        self.positions: Dict = {}
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Data collector
        self.collector = AsterDEXCollector()
        
        # Strategy
        strategy_name = self.paper_config['strategy']
        if strategy_name == 'ensemble':
            mode = self.paper_config['ensemble']['mode']
            weights = self.paper_config['ensemble']['weights']
            threshold = self.paper_config['ensemble']['confidence_threshold']
            self.strategy = EnsembleStrategy(
                mode=mode,
                weights=weights,
                confidence_threshold=threshold
            )
            logger.info(f"✅ Loaded Ensemble Strategy ({mode} mode)")
        elif strategy_name == 'rsi_macd':
            self.strategy = RSIMACDStrategy()
            logger.info(f"✅ Loaded RSI+MACD Strategy")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Risk management
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer()
        
        # Performance tracking
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_losses': 0,
            'max_consecutive_losses': 0
        }
        
        # Safety limits
        self.safety = self.config['safety']
        self.emergency_stop_triggered = False
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        logger.info("🎯 Paper Trading Engine Initialized")
    
    async def load_historical_data(self, symbol: str, timeframe: str, 
                                   lookback_days: int = 90) -> pd.DataFrame:
        """Load historical data for strategy initialization"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"📊 Loading {symbol} {timeframe} data ({lookback_days} days)...")
        
        df = await self.collector.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            start_time=int(start_date.timestamp() * 1000),
            end_time=int(end_date.timestamp() * 1000)
        )
        
        if df is not None and not df.empty:
            logger.info(f"✅ Loaded {len(df)} candles")
            return df
        else:
            logger.error(f"❌ Failed to load data for {symbol}")
            return pd.DataFrame()
    
    def check_safety_limits(self) -> bool:
        """Check if safety limits are breached"""
        # Emergency stop
        if self.safety['emergency_stop']['enabled']:
            loss_pct = (self.initial_capital - self.capital) / self.initial_capital * 100
            if loss_pct >= self.safety['emergency_stop']['trigger_loss_pct']:
                self.emergency_stop_triggered = True
                logger.critical(f"🚨 EMERGENCY STOP TRIGGERED! Lost {loss_pct:.1f}%")
                return False
        
        # Circuit breaker
        if self.safety['circuit_breaker']['enabled']:
            max_losses = self.safety['circuit_breaker']['max_consecutive_losses']
            if self.metrics['consecutive_losses'] >= max_losses:
                if not self.circuit_breaker_active:
                    cooldown = self.safety['circuit_breaker']['cooldown_minutes']
                    self.circuit_breaker_until = datetime.now() + timedelta(minutes=cooldown)
                    self.circuit_breaker_active = True
                    logger.warning(f"⚠️ Circuit breaker activated! {max_losses} consecutive losses")
                    logger.warning(f"⏸️ Trading paused until {self.circuit_breaker_until.strftime('%H:%M:%S')}")
                return False
            
            # Check if cooldown expired
            if self.circuit_breaker_active:
                if datetime.now() >= self.circuit_breaker_until:
                    self.circuit_breaker_active = False
                    logger.info("✅ Circuit breaker reset, resuming trading")
                else:
                    return False
        
        return True
    
    def enter_position(self, symbol: str, signal_data: Dict, current_price: float, 
                      leverage: int, atr: float):
        """Enter a position (paper trading)"""
        if symbol in self.positions:
            logger.warning(f"⚠️ Already in position for {symbol}")
            return
        
        # Check safety limits
        if not self.check_safety_limits():
            return
        
        # Calculate position size
        risk_config = self.paper_config['risk']
        position_value = self.capital * (risk_config['max_position_size_pct'] / 100) * leverage
        quantity = position_value / current_price
        
        # Calculate stop loss and take profit
        stop_loss_multiplier = risk_config['stop_loss_atr_multiplier']
        take_profit_multiplier = risk_config['take_profit_atr_multiplier']
        
        stop_loss = current_price - (atr * stop_loss_multiplier)
        take_profit = current_price + (atr * take_profit_multiplier)
        
        # Create position
        position = {
            'symbol': symbol,
            'entry_price': current_price,
            'quantity': quantity,
            'leverage': leverage,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence']
        }
        
        self.positions[symbol] = position
        
        logger.info(f"🟢 ENTER LONG: {symbol}")
        logger.info(f"   Entry: ${current_price:.2f}")
        logger.info(f"   Quantity: {quantity:.4f} (${position_value:.2f})")
        logger.info(f"   Stop Loss: ${stop_loss:.2f} ({-stop_loss_multiplier:.1f} ATR)")
        logger.info(f"   Take Profit: ${take_profit:.2f} ({take_profit_multiplier:.1f} ATR)")
        logger.info(f"   Confidence: {signal_data['confidence']:.1%}")
        
        # Log to console alert
        if self.paper_config['alerts']['console']['on_trade']:
            self._log_trade_alert("ENTRY", position)
    
    def exit_position(self, symbol: str, current_price: float, reason: str):
        """Exit a position (paper trading)"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate PnL
        entry_price = position['entry_price']
        quantity = position['quantity']
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Update capital
        self.capital += pnl
        
        # Update metrics
        self.metrics['total_trades'] += 1
        if pnl > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['total_profit'] += pnl
            self.metrics['largest_win'] = max(self.metrics['largest_win'], pnl)
            self.metrics['consecutive_losses'] = 0  # Reset
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['total_loss'] += abs(pnl)
            self.metrics['largest_loss'] = min(self.metrics['largest_loss'], pnl)
            self.metrics['consecutive_losses'] += 1
            self.metrics['max_consecutive_losses'] = max(
                self.metrics['max_consecutive_losses'],
                self.metrics['consecutive_losses']
            )
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': entry_price,
            'exit_price': current_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'confidence': position['confidence']
        }
        self.trade_history.append(trade)
        
        # Log exit
        logger.info(f"🔴 EXIT {symbol}: {reason}")
        logger.info(f"   Entry: ${entry_price:.2f} → Exit: ${current_price:.2f}")
        logger.info(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"   New Capital: ${self.capital:.2f}")
        
        # Log to console alert
        if self.paper_config['alerts']['console']['on_trade']:
            self._log_trade_alert("EXIT", trade)
        
        # Remove position
        del self.positions[symbol]
    
    def _log_trade_alert(self, trade_type: str, data: Dict):
        """Log trade alert to console"""
        print("\n" + "="*60)
        print(f"📢 PAPER TRADING ALERT: {trade_type}")
        print("="*60)
        for key, value in data.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
    
    def check_position_management(self, symbol: str, current_data: pd.DataFrame):
        """Check if position should be exited"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = current_data['close'].iloc[-1]
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            self.exit_position(symbol, current_price, "Stop Loss Hit")
            return
        
        # Check take profit
        if current_price >= position['take_profit']:
            self.exit_position(symbol, current_price, "Take Profit Hit")
            return
        
        # Check strategy exit signal
        signal_data = self.strategy.generate_signal(current_data)
        if signal_data['signal'] == 'SELL':
            self.exit_position(symbol, current_price, "Strategy Exit Signal")
            return
    
    def print_statistics(self):
        """Print current statistics"""
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (self.metrics['winning_trades'] / self.metrics['total_trades'] * 100) if self.metrics['total_trades'] > 0 else 0
        
        print("\n" + "="*60)
        print("📊 PAPER TRADING STATISTICS")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Current Capital: ${self.capital:.2f}")
        print(f"Total Return: ${self.capital - self.initial_capital:.2f} ({total_return:+.2f}%)")
        print(f"\nTrades: {self.metrics['total_trades']}")
        print(f"Win Rate: {win_rate:.1f}% ({self.metrics['winning_trades']}W / {self.metrics['losing_trades']}L)")
        print(f"Total Profit: ${self.metrics['total_profit']:.2f}")
        print(f"Total Loss: ${self.metrics['total_loss']:.2f}")
        print(f"Largest Win: ${self.metrics['largest_win']:.2f}")
        print(f"Largest Loss: ${self.metrics['largest_loss']:.2f}")
        print(f"Max Consecutive Losses: {self.metrics['max_consecutive_losses']}")
        
        # Compare to backtest
        if self.paper_config['tracking']['compare_to_backtest']:
            backtest = self.paper_config['tracking']['backtest_reference']
            print(f"\n📈 vs BACKTEST EXPECTATIONS:")
            print(f"Return: {total_return:+.2f}% (expected: {backtest['expected_return_pct']:+.1f}%)")
            print(f"Win Rate: {win_rate:.1f}% (expected: {backtest['expected_win_rate_pct']:.1f}%)")
        
        print("="*60 + "\n")
    
    async def run(self, duration_minutes: int = 60):
        """Run paper trading for specified duration"""
        logger.info(f"🚀 Starting Paper Trading (Duration: {duration_minutes} minutes)")
        logger.info(f"💰 Initial Capital: ${self.capital:.2f}")
        logger.info(f"📊 Strategy: {self.paper_config['strategy']}")
        
        # Get trading pairs
        symbols_config = [s for s in self.paper_config['symbols'] if s['enabled']]
        if not symbols_config:
            logger.error("❌ No enabled trading pairs!")
            return
        
        symbol_config = symbols_config[0]  # Use first enabled pair
        symbol = symbol_config['symbol']
        timeframe = symbol_config['timeframe']
        leverage = symbol_config['leverage']
        
        logger.info(f"📈 Trading Pair: {symbol} {timeframe} (leverage: {leverage}x)")
        
        # Load historical data for strategy initialization
        historical_data = await self.load_historical_data(symbol, timeframe)
        if historical_data.empty:
            logger.error("❌ Failed to load historical data")
            return
        
        # Initialize strategy
        self.strategy.calculate_indicators(historical_data)
        
        logger.info(f"✅ Strategy initialized with {len(historical_data)} historical candles")
        logger.info(f"⏰ Running for {duration_minutes} minutes...")
        logger.info("="*60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        iteration = 0
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                
                # Fetch latest data
                df = await self.load_historical_data(symbol, timeframe, lookback_days=30)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                
                # Calculate indicators
                self.strategy.calculate_indicators(df)
                current_price = df['close'].iloc[-1]
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                
                # Check existing positions
                self.check_position_management(symbol, df)
                
                # Check for new signals (if not in position)
                if symbol not in self.positions:
                    signal_data = self.strategy.generate_signal(df)
                    
                    if signal_data['signal'] == 'BUY':
                        confidence = signal_data['confidence']
                        threshold = self.paper_config['ensemble'].get('confidence_threshold', 0.6)
                        
                        if confidence >= threshold:
                            logger.info(f"🔔 BUY Signal: {symbol} @ ${current_price:.2f} (confidence: {confidence:.1%})")
                            self.enter_position(symbol, signal_data, current_price, leverage, atr)
                
                # Print statistics every 10 iterations
                if iteration % 10 == 0:
                    self.print_statistics()
                
                # Wait before next check (check every minute for 1h timeframe)
                await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n⚠️ Paper trading interrupted by user")
        
        finally:
            # Close any open positions
            for symbol in list(self.positions.keys()):
                df = await self.load_historical_data(symbol, timeframe, lookback_days=1)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    self.exit_position(symbol, current_price, "Session End")
            
            # Final statistics
            logger.info(f"\n🏁 Paper Trading Session Complete")
            self.print_statistics()
            
            # Save results
            self.save_results()
    
    def save_results(self):
        """Save paper trading results"""
        if self.paper_config['monitoring']['save_trade_history']:
            # Save trade history
            if self.trade_history:
                df = pd.DataFrame(self.trade_history)
                history_file = self.paper_config['monitoring']['trade_history_file']
                df.to_csv(history_file, index=False)
                logger.info(f"💾 Trade history saved to {history_file}")
            
            # Save metrics
            if self.paper_config['tracking']['enabled']:
                import json
                metrics_file = self.paper_config['tracking']['metrics_file']
                
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'initial_capital': self.initial_capital,
                    'final_capital': self.capital,
                    'total_return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
                    'metrics': self.metrics,
                    'trade_count': len(self.trade_history)
                }
                
                with open(metrics_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Metrics saved to {metrics_file}")


async def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       📊 PAPER TRADING DEMO - ENSEMBLE STRATEGY         ║
    ║                                                          ║
    ║  Testing strategies with virtual money before live      ║
    ║  trading to validate backtest results in real-time.     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize paper trading engine
    engine = PaperTradingEngine()
    
    # Run for 60 minutes (or until Ctrl+C)
    await engine.run(duration_minutes=60)
    
    print("\n✅ Paper trading demo complete!")
    print("📊 Review logs/paper_trades.csv for trade history")
    print("📈 Review logs/paper_trading_metrics.json for performance metrics")


if __name__ == "__main__":
    asyncio.run(main())
