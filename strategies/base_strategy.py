"""
Base Strategy Class
Abstract base class for all trading strategies with comprehensive 
position management, risk management, and performance tracking
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from core.logger import logger


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logger
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.trade_history = []
        
        # Risk management parameters
        self.initial_stop_loss = None
        self.trailing_stop = None
        self.take_profit = None
        
        self.logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def should_enter(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine if we should enter a position
        Returns: dict with keys 'signal' (BUY/SELL/HOLD), 'confidence', 'metadata'
        """
        pass
    
    @abstractmethod
    def should_exit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine if we should exit current position
        Returns: dict with keys 'signal' (EXIT/HOLD), 'reason', 'metadata'
        """
        pass
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators for the strategy"""
        return data
    
    def update_position(self, signal: str, price: float, timestamp: datetime, 
                       reason: str = None) -> None:
        """Update current position state"""
        if signal in ['BUY', 'SELL'] and self.position is None:
            self.position = 'long' if signal == 'BUY' else 'short'
            self.entry_price = price
            self.entry_time = timestamp
            self._set_stop_loss_take_profit(price, signal)
            
            trade = {
                'action': 'ENTER',
                'position': self.position,
                'price': price,
                'timestamp': timestamp,
                'reason': reason
            }
            self.trade_history.append(trade)
            self.logger.info(f"Entered {self.position} position at {price}")
            
        elif signal == 'EXIT' and self.position is not None:
            profit_pct = ((price - self.entry_price) / self.entry_price * 100 
                         if self.position == 'long' else 
                         (self.entry_price - price) / self.entry_price * 100)
            
            trade = {
                'action': 'EXIT',
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': price,
                'profit_pct': profit_pct,
                'timestamp': timestamp,
                'reason': reason,
                'duration': (timestamp - self.entry_time).total_seconds() / 3600  # hours
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"Exited {self.position} position at {price} "
                           f"({profit_pct:.2f}%) - Reason: {reason}")
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.entry_time = None
            self.initial_stop_loss = None
            self.trailing_stop = None
            self.take_profit = None
    
    def _set_stop_loss_take_profit(self, entry_price: float, signal: str) -> None:
        """Set initial stop loss and take profit levels"""
        stop_loss_pct = self.config.get('stop_loss_pct', 2.0)
        take_profit_pct = self.config.get('take_profit_pct', 4.0)
        use_trailing_stop = self.config.get('use_trailing_stop', False)
        
        if signal == 'BUY':
            self.initial_stop_loss = entry_price * (1 - stop_loss_pct / 100)
            self.take_profit = entry_price * (1 + take_profit_pct / 100)
            if use_trailing_stop:
                self.trailing_stop = self.initial_stop_loss
        else:  # SELL (short)
            self.initial_stop_loss = entry_price * (1 + stop_loss_pct / 100)
            self.take_profit = entry_price * (1 - take_profit_pct / 100)
            if use_trailing_stop:
                self.trailing_stop = self.initial_stop_loss
    
    def check_risk_management(self, current_price: float, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Check if we should exit due to risk management"""
        if self.position is None:
            return None
            
        reason = None
        current_low = current_price  # In reality, you might want to use the low of the candle
        
        if self.position == 'long':
            # Check stop loss
            if current_low <= self.initial_stop_loss:
                reason = f"Stop loss hit ({self.initial_stop_loss:.2f})"
            
            # Check trailing stop
            elif (self.trailing_stop and 
                  current_low <= self.trailing_stop):
                reason = f"Trailing stop hit ({self.trailing_stop:.2f})"
            
            # Check take profit
            elif current_price >= self.take_profit:
                reason = f"Take profit hit ({self.take_profit:.2f})"
                
            # Update trailing stop for long position
            elif (self.trailing_stop and 
                  current_price > self.entry_price):
                new_trailing = current_price * (1 - self.config.get('trailing_stop_pct', 1.5) / 100)
                if new_trailing > self.trailing_stop:
                    self.trailing_stop = new_trailing
                    
        else:  # short position
            # Check stop loss
            if current_price >= self.initial_stop_loss:
                reason = f"Stop loss hit ({self.initial_stop_loss:.2f})"
            
            # Check trailing stop
            elif (self.trailing_stop and 
                  current_price >= self.trailing_stop):
                reason = f"Trailing stop hit ({self.trailing_stop:.2f})"
            
            # Check take profit
            elif current_price <= self.take_profit:
                reason = f"Take profit hit ({self.take_profit:.2f})"
                
            # Update trailing stop for short position
            elif (self.trailing_stop and 
                  current_price < self.entry_price):
                new_trailing = current_price * (1 + self.config.get('trailing_stop_pct', 1.5) / 100)
                if new_trailing < self.trailing_stop:
                    self.trailing_stop = new_trailing
        
        if reason:
            return {
                'signal': 'EXIT',
                'reason': reason,
                'metadata': {
                    'current_price': current_price,
                    'stop_loss': self.initial_stop_loss,
                    'trailing_stop': self.trailing_stop,
                    'take_profit': self.take_profit
                }
            }
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate strategy performance metrics"""
        if not self.trade_history:
            return {}
            
        exit_trades = [t for t in self.trade_history if t['action'] == 'EXIT']
        
        if not exit_trades:
            return {}
        
        wins = [t for t in exit_trades if t['profit_pct'] > 0]
        losses = [t for t in exit_trades if t['profit_pct'] <= 0]
        
        total_trades = len(exit_trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_profit = sum(t['profit_pct'] for t in exit_trades) / total_trades
        avg_win = sum(t['profit_pct'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['profit_pct'] for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t['profit_pct'] for t in wins) / 
                           sum(t['profit_pct'] for t in losses)) if losses else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_profit': sum(t['profit_pct'] for t in exit_trades),
            'largest_win': max(t['profit_pct'] for t in exit_trades) if wins else 0,
            'largest_loss': min(t['profit_pct'] for t in exit_trades) if losses else 0
        }
