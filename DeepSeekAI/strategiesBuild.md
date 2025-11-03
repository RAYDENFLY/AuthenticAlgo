TRADING STRATEGIES MODULE



1. strategies/init.py

```python
"""
Trading Strategies Module
Modular and configurable trading strategies
"""

from .base_strategy import BaseStrategy
from .rsi_macd import RSIMACDStrategy
from .bollinger import BollingerBandsStrategy
from .ml_strategy import MLStrategy

STRATEGY_REGISTRY = {
    'RSI_MACD_Strategy': RSIMACDStrategy,
    'BollingerBands_Strategy': BollingerBandsStrategy,
    'ML_Strategy': MLStrategy
}

def create_strategy(strategy_name: str, config: dict) -> BaseStrategy:
    """Factory function to create strategy instances"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy {strategy_name} not found. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[strategy_name](config)

__all__ = ['BaseStrategy', 'RSIMACDStrategy', 'BollingerBandsStrategy', 
           'MLStrategy', 'create_strategy', 'STRATEGY_REGISTRY']
```


           
2. strategies/base_strategy.py

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")
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
```


3. strategies/rsi_macd.py


```python
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy
import logging

class RSIMACDStrategy(BaseStrategy):
    """
    RSI + MACD Strategy
    Entry: RSI oversold + MACD bullish crossover
    Exit: RSI overbought + MACD bearish crossover
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI_MACD_Strategy", config)
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # Confirmation settings
        self.require_volume_confirmation = config.get('require_volume_confirmation', True)
        self.min_volume_multiplier = config.get('min_volume_multiplier', 1.2)
        
        self.logger.info(f"RSI Period: {self.rsi_period}, "
                        f"RSI Levels: {self.rsi_oversold}/{self.rsi_overbought}, "
                        f"MACD: {self.macd_fast}/{self.macd_slow}/{self.macd_signal}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and MACD indicators"""
        # RSI Calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        ema_fast = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.macd_slow, adjust=False).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Volume analysis
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data
    
    def should_enter(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for entry conditions:
        - RSI oversold (below 30)
        - MACD bullish crossover (MACD crosses above signal)
        - Optional: Volume confirmation
        """
        if len(data) < max(self.rsi_period, self.macd_slow) + 1:
            return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'Insufficient data'}}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Check for BUY signal (long position)
        buy_conditions = [
            current['rsi'] < self.rsi_oversold,  # RSI oversold
            current['macd'] > current['macd_signal'],  # MACD above signal
            prev['macd'] <= prev['macd_signal'],  # MACD crossover occurred
            current['macd_histogram'] > 0,  # Histogram positive
        ]
        
        # Volume confirmation (optional)
        if self.require_volume_confirmation and 'volume_ratio' in data.columns:
            buy_conditions.append(current['volume_ratio'] >= self.min_volume_multiplier)
        
        if all(buy_conditions):
            confidence = self._calculate_confidence(current, 'BUY')
            return {
                'signal': 'BUY',
                'confidence': confidence,
                'metadata': {
                    'reason': 'RSI oversold + MACD bullish crossover',
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal'],
                    'volume_ratio': current.get('volume_ratio', 1.0)
                }
            }
        
        # Check for SELL signal (short position)
        sell_conditions = [
            current['rsi'] > self.rsi_overbought,  # RSI overbought
            current['macd'] < current['macd_signal'],  # MACD below signal
            prev['macd'] >= prev['macd_signal'],  # MACD crossover occurred
            current['macd_histogram'] < 0,  # Histogram negative
        ]
        
        # Volume confirmation (optional)
        if self.require_volume_confirmation and 'volume_ratio' in data.columns:
            sell_conditions.append(current['volume_ratio'] >= self.min_volume_multiplier)
        
        if all(sell_conditions):
            confidence = self._calculate_confidence(current, 'SELL')
            return {
                'signal': 'SELL', 
                'confidence': confidence,
                'metadata': {
                    'reason': 'RSI overbought + MACD bearish crossover',
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal'],
                    'volume_ratio': current.get('volume_ratio', 1.0)
                }
            }
        
        return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'No entry conditions met'}}
    
    def should_exit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for exit conditions:
        - For long: RSI overbought OR MACD bearish crossover
        - For short: RSI oversold OR MACD bullish crossover
        """
        if self.position is None or len(data) < 2:
            return {'signal': 'HOLD', 'reason': 'No position to exit'}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        if self.position == 'long':
            # Exit conditions for long position
            exit_conditions = [
                current['rsi'] > self.rsi_overbought,  # RSI overbought
                (current['macd'] < current['macd_signal'] and  # MACD bearish crossover
                 prev['macd'] >= prev['macd_signal'])
            ]
            
            if any(exit_conditions):
                reason = "RSI overbought" if exit_conditions[0] else "MACD bearish crossover"
                return {
                    'signal': 'EXIT',
                    'reason': reason,
                    'metadata': {
                        'rsi': current['rsi'],
                        'macd': current['macd'],
                        'macd_signal': current['macd_signal']
                    }
                }
        
        else:  # short position
            # Exit conditions for short position
            exit_conditions = [
                current['rsi'] < self.rsi_oversold,  # RSI oversold
                (current['macd'] > current['macd_signal'] and  # MACD bullish crossover
                 prev['macd'] <= prev['macd_signal'])
            ]
            
            if any(exit_conditions):
                reason = "RSI oversold" if exit_conditions[0] else "MACD bullish crossover"
                return {
                    'signal': 'EXIT',
                    'reason': reason,
                    'metadata': {
                        'rsi': current['rsi'],
                        'macd': current['macd'],
                        'macd_signal': current['macd_signal']
                    }
                }
        
        return {'signal': 'HOLD', 'reason': 'No exit conditions met'}
    
    def _calculate_confidence(self, current_data: pd.Series, signal: str) -> float:
        """Calculate confidence score for the signal (0-1)"""
        confidence = 0.5  # Base confidence
        
        # RSI extreme adds confidence
        if signal == 'BUY':
            rsi_extreme = max(0, (self.rsi_oversold - current_data['rsi']) / self.rsi_oversold)
            confidence += rsi_extreme * 0.3
        else:  # SELL
            rsi_extreme = max(0, (current_data['rsi'] - self.rsi_overbought) / (100 - self.rsi_overbought))
            confidence += rsi_extreme * 0.3
        
        # MACD histogram strength
        macd_strength = min(1.0, abs(current_data['macd_histogram']) / 
                           (abs(current_data['macd']) * 0.1))  # Normalize
        confidence += macd_strength * 0.2
        
        # Volume confirmation (if available)
        if 'volume_ratio' in current_data:
            volume_boost = min(0.5, (current_data['volume_ratio'] - 1) * 0.5)
            confidence += volume_boost
        
        return min(1.0, confidence)  # Cap at 1.0
```



4. strategies/bollinger.py


```python
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy
import logging

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy
    Entry: Price touches lower/upper band with volume confirmation
    Exit: Price reaches middle band or opposite band
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BollingerBands_Strategy", config)
        
        # Strategy parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.require_volume_spike = config.get('require_volume_spike', True)
        self.min_volume_multiplier = config.get('min_volume_multiplier', 1.5)
        self.use_rsi_confirmation = config.get('use_rsi_confirmation', True)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        self.logger.info(f"Bollinger Bands: {self.bb_period} period, {self.bb_std} STD")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and related indicators"""
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=self.bb_period).mean()
        bb_std = data['close'].rolling(window=self.bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * self.bb_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std * self.bb_std)
        
        # Band width and position
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume analysis
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # RSI for confirmation
        if self.use_rsi_confirmation:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def should_enter(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for entry conditions:
        - Long: Price touches lower band + volume spike + optional RSI oversold
        - Short: Price touches upper band + volume spike + optional RSI overbought
        """
        if len(data) < self.bb_period + 1:
            return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'Insufficient data'}}
        
        current = data.iloc[-1]
        current_low = current['low']
        current_high = current['high']
        current_close = current['close']
        
        # Check for BUY signal (mean reversion from lower band)
        buy_conditions = [
            current_low <= current['bb_lower'],  # Price touched lower band
            current_close > current['bb_lower'],  # Closed above lower band
        ]
        
        # Volume confirmation
        if self.require_volume_spike and 'volume_ratio' in data.columns:
            buy_conditions.append(current['volume_ratio'] >= self.min_volume_multiplier)
        
        # RSI confirmation (optional)
        if self.use_rsi_confirmation and 'rsi' in data.columns:
            buy_conditions.append(current['rsi'] < self.rsi_overbought)  # Not overbought
        
        if all(buy_conditions):
            confidence = self._calculate_confidence(current, 'BUY')
            return {
                'signal': 'BUY',
                'confidence': confidence,
                'metadata': {
                    'reason': 'Price at lower Bollinger Band with confirmation',
                    'bb_position': current['bb_position'],
                    'bb_width': current['bb_width'],
                    'volume_ratio': current.get('volume_ratio', 1.0),
                    'rsi': current.get('rsi', 50)
                }
            }
        
        # Check for SELL signal (mean reversion from upper band)
        sell_conditions = [
            current_high >= current['bb_upper'],  # Price touched upper band
            current_close < current['bb_upper'],  # Closed below upper band
        ]
        
        # Volume confirmation
        if self.require_volume_spike and 'volume_ratio' in data.columns:
            sell_conditions.append(current['volume_ratio'] >= self.min_volume_multiplier)
        
        # RSI confirmation (optional)
        if self.use_rsi_confirmation and 'rsi' in data.columns:
            sell_conditions.append(current['rsi'] > self.rsi_oversold)  # Not oversold
        
        if all(sell_conditions):
            confidence = self._calculate_confidence(current, 'SELL')
            return {
                'signal': 'SELL',
                'confidence': confidence,
                'metadata': {
                    'reason': 'Price at upper Bollinger Band with confirmation',
                    'bb_position': current['bb_position'],
                    'bb_width': current['bb_width'],
                    'volume_ratio': current.get('volume_ratio', 1.0),
                    'rsi': current.get('rsi', 50)
                }
            }
        
        return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'No entry conditions met'}}
    
    def should_exit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for exit conditions:
        - For long: Price reaches middle band or upper band
        - For short: Price reaches middle band or lower band
        """
        if self.position is None:
            return {'signal': 'HOLD', 'reason': 'No position to exit'}
        
        current = data.iloc[-1]
        current_close = current['close']
        
        if self.position == 'long':
            # Exit conditions for long position
            exit_conditions = [
                current_close >= current['bb_middle'],  # Reached middle band
                current_close >= current['bb_upper'],   # Reached upper band
            ]
            
            if any(exit_conditions):
                reason = "Reached middle band" if exit_conditions[0] else "Reached upper band"
                return {
                    'signal': 'EXIT',
                    'reason': reason,
                    'metadata': {
                        'bb_position': current['bb_position'],
                        'current_price': current_close,
                        'bb_middle': current['bb_middle'],
                        'bb_upper': current['bb_upper']
                    }
                }
        
        else:  # short position
            # Exit conditions for short position
            exit_conditions = [
                current_close <= current['bb_middle'],  # Reached middle band
                current_close <= current['bb_lower'],   # Reached lower band
            ]
            
            if any(exit_conditions):
                reason = "Reached middle band" if exit_conditions[0] else "Reached lower band"
                return {
                    'signal': 'EXIT',
                    'reason': reason,
                    'metadata': {
                        'bb_position': current['bb_position'],
                        'current_price': current_close,
                        'bb_middle': current['bb_middle'],
                        'bb_lower': current['bb_lower']
                    }
                }
        
        return {'signal': 'HOLD', 'reason': 'No exit conditions met'}
    
    def _calculate_confidence(self, current_data: pd.Series, signal: str) -> float:
        """Calculate confidence score for the signal (0-1)"""
        confidence = 0.6  # Base confidence for Bollinger strategy
        
        # Band width adds confidence (wider bands = stronger signal)
        band_width_boost = min(0.2, current_data['bb_width'] * 10)  # Normalize
        confidence += band_width_boost
        
        # Extreme band position adds confidence
        if signal == 'BUY':
            position_boost = (1 - current_data['bb_position']) * 0.2
        else:  # SELL
            position_boost = current_data['bb_position'] * 0.2
        confidence += position_boost
        
        # Volume confirmation
        if 'volume_ratio' in current_data:
            volume_boost = min(0.2, (current_data['volume_ratio'] - 1) * 0.4)
            confidence += volume_boost
        
        # RSI confirmation
        if self.use_rsi_confirmation and 'rsi' in current_data:
            if (signal == 'BUY' and current_data['rsi'] < self.rsi_oversold) or \
               (signal == 'SELL' and current_data['rsi'] > self.rsi_overbought):
                confidence += 0.1
        
        return min(1.0, confidence)
```



5. strategies/ml_strategy.py


```python
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging
import joblib
from pathlib import Path

class MLStrategy(BaseStrategy):
    """
    Machine Learning Based Strategy
    Uses trained ML models for prediction with traditional indicator fallback
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ML_Strategy", config)
        
        # ML model settings
        self.model_path = config.get('model_path', 'ml/models/')
        self.model_name = config.get('model_name', 'xgboost_model.pkl')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.use_fallback = config.get('use_fallback', True)
        
        # Feature settings
        self.lookback_period = config.get('lookback_period', 50)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        
        # Fallback strategy
        self.fallback_strategy = None
        if self.use_fallback:
            from .rsi_macd import RSIMACDStrategy
            fallback_config = config.get('fallback_config', {})
            self.fallback_strategy = RSIMACDStrategy(fallback_config)
        
        # Load ML model
        self.model = self._load_model()
        
        self.logger.info(f"ML Strategy: {self.model_name}, "
                        f"Confidence Threshold: {self.confidence_threshold}")
    
    def _load_model(self) -> Optional[Any]:
        """Load trained ML model"""
        try:
            model_file = Path(self.model_path) / self.model_name
            if model_file.exists():
                model = joblib.load(model_file)
                self.logger.info(f"Loaded ML model: {self.model_name}")
                return model
            else:
                self.logger.warning(f"Model file not found: {model_file}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for ML model and fallback indicators"""
        # Calculate traditional indicators as features
        from indicators import TechnicalIndicators
        indicators = TechnicalIndicators()
        data = indicators.get_all_indicators(data)
        
        # Add additional features for ML model
        data = self._create_ml_features(data)
        
        return data
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model prediction"""
        # Price-based features
        data['price_change'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['open_close_ratio'] = data['close'] / data['open']
        
        # Volatility features
        data['volatility_5'] = data['close'].pct_change().rolling(5).std()
        data['volatility_20'] = data['close'].pct_change().rolling(20).std()
        
        # Momentum features
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
            data['volume_volatility'] = data['volume'].rolling(10).std()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            data[f'close_ma_{window}'] = data['close'].rolling(window).mean()
            data[f'volume_ma_{window}'] = data['volume'].rolling(window).mean() if 'volume' in data.columns else 0
        
        # Target variable (future price movement)
        data['target'] = (data['close'].shift(-self.prediction_horizon) / data['close'] - 1) * 100
        
        return data
    
    def should_enter(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Use ML model for prediction with fallback to traditional strategy
        """
        if len(data) < self.lookback_period:
            return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'Insufficient data'}}
        
        ml_signal = self._get_ml_signal(data)
        
        # Use ML signal if confidence is high enough
        if ml_signal and ml_signal['confidence'] >= self.confidence_threshold:
            return ml_signal
        
        # Fallback to traditional strategy
        if self.use_fallback and self.fallback_strategy:
            fallback_data = self.fallback_strategy.calculate_indicators(data.copy())
            fallback_signal = self.fallback_strategy.should_enter(fallback_data)
            
            if fallback_signal['signal'] != 'HOLD':
                fallback_signal['metadata']['source'] = 'fallback'
                return fallback_signal
        
        return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'No high-confidence signals'}}
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get trading signal from ML model"""
        if self.model is None:
            return None
        
        try:
            # Prepare features for prediction
            features = self._prepare_features(data)
            if features is None:
                return None
            
            # Make prediction
            prediction = self.model.predict(features)
            prediction_proba = self.model.predict_proba(features) if hasattr(self.model, 'predict_proba') else None
            
            # Interpret prediction
            # Assuming model predicts: -1 (SELL), 0 (HOLD), 1 (BUY)
            signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            signal = signal_map.get(prediction[0], 'HOLD')
            
            # Calculate confidence
            if prediction_proba is not None:
                confidence = np.max(prediction_proba[0])
            else:
                confidence = 0.8 if signal != 'HOLD' else 0.0  # Default confidence
            
            if signal != 'HOLD':
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'metadata': {
                        'source': 'ml_model',
                        'prediction': prediction[0],
                        'confidence': confidence,
                        'model': self.model_name
                    }
                }
            
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
        
        return None
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML model prediction"""
        try:
            # Select feature columns (you should customize this based on your model)
            feature_columns = [
                'rsi_14', 'macd', 'macd_signal', 'bb_position', 'bb_width',
                'atr_14', 'volume_ratio', 'price_change', 'volatility_5',
                'momentum_5', 'close_ma_5', 'close_ma_20'
            ]
            
            # Check if all required features are available
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) < len(feature_columns) * 0.8:  # At least 80% of features
                self.logger.warning("Insufficient features for ML prediction")
                return None
            
            # Get the latest row with all features
            latest_data = data[available_features].iloc[-1:].fillna(0)
            
            return latest_data
            
        except Exception as e:
            self.logger.error(f"Feature preparation error: {e}")
            return None
    
    def should_exit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for exit conditions using ML model and traditional methods
        """
        if self.position is None:
            return {'signal': 'HOLD', 'reason': 'No position to exit'}
        
        # Check ML model for exit signal
        ml_exit_signal = self._get_ml_exit_signal(data)
        if ml_exit_signal:
            return ml_exit_signal
        
        # Fallback to traditional exit logic
        if self.use_fallback and self.fallback_strategy:
            fallback_data = self.fallback_strategy.calculate_indicators(data.copy())
            fallback_exit = self.fallback_strategy.should_exit(fallback_data)
            
            if fallback_exit['signal'] == 'EXIT':
                fallback_exit['metadata']['source'] = 'fallback'
                return fallback_exit
        
        return {'signal': 'HOLD', 'reason': 'No exit conditions met'}
    
    def _get_ml_exit_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get exit signal from ML model"""
        if self.model is None:
            return None
        
        try:
            ml_signal = self._get_ml_signal(data)
            if ml_signal and ml_signal['signal'] != 'HOLD':
                # If ML suggests opposite direction, exit
                if (self.position == 'long' and ml_signal['signal'] == 'SELL') or \
                   (self.position == 'short' and ml_signal['signal'] == 'BUY'):
                    return {
                        'signal': 'EXIT',
                        'reason': f'ML model suggests opposite direction',
                        'metadata': {
                            'source': 'ml_model',
                            'suggested_signal': ml_signal['signal'],
                            'confidence': ml_signal['confidence']
                        }
                    }
        
        except Exception as e:
            self.logger.error(f"ML exit signal error: {e}")
        
        return None
```

6. tests/test_strategies.py

```python
import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import RSIMACDStrategy, BollingerBandsStrategy, MLStrategy, create_strategy
from strategies.base_strategy import BaseStrategy

class TestStrategies:
    """Test suite for trading strategies"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        np.random.seed(42)
        
        # Create trending data with some noise
        trend = np.cumsum(np.random.randn(200) * 0.01) + 100
        noise = np.random.randn(200) * 0.5
        
        data = {
            'open': trend + noise,
            'high': trend + noise + 0.5,
            'low': trend + noise - 0.5,
            'close': trend + noise,
            'volume': np.random.randint(1000, 10000, 200)
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_rsi_macd_strategy_creation(self):
        """Test RSI+MACD strategy creation"""
        config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0
        }
        
        strategy = RSIMACDStrategy(config)
        assert strategy.name == "RSI_MACD_Strategy"
        assert strategy.rsi_period == 14
        assert strategy.rsi_oversold == 30
    
    def test_rsi_macd_indicator_calculation(self, sample_data):
        """Test RSI+MACD indicator calculation"""
        config = {'rsi_period': 14}
        strategy = RSIMACDStrategy(config)
        
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        
        assert 'rsi' in data_with_indicators.columns
        assert 'macd' in data_with_indicators.columns
        assert 'macd_signal' in data_with_indicators.columns
        assert 'macd_histogram' in data_with_indicators.columns
        
        # RSI should be between 0 and 100
        assert 0 <= data_with_indicators['rsi'].min() <= 100
        assert 0 <= data_with_indicators['rsi'].max() <= 100
    
    def test_rsi_macd_entry_signals(self, sample_data):
        """Test RSI+MACD entry signal generation"""
        config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        strategy = RSIMACDStrategy(config)
        
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        signal = strategy.should_enter(data_with_indicators)
        
        assert 'signal' in signal
        assert 'confidence' in signal
        assert 'metadata' in signal
        assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal['confidence'] <= 1
    
    def test_bollinger_bands_strategy(self, sample_data):
        """Test Bollinger Bands strategy"""
        config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'require_volume_spike': False
        }
        strategy = BollingerBandsStrategy(config)
        
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())
        signal = strategy.should_enter(data_with_indicators)
        
        assert 'signal' in signal
        assert 'confidence' in signal
        assert 'bb_position' in signal['metadata']
        
        # Test indicator calculation
        assert 'bb_upper' in data_with_indicators.columns
        assert 'bb_middle' in data_with_indicators.columns
        assert 'bb_lower' in data_with_indicators.columns
        assert 'bb_width' in data_with_indicators.columns
    
    def test_ml_strategy_creation(self):
        """Test ML strategy creation"""
        config = {
            'model_path': 'ml/models/',
            'confidence_threshold': 0.7,
            'use_fallback': True
        }
        
        strategy = MLStrategy(config)
        assert strategy.name == "ML_Strategy"
        assert strategy.confidence_threshold == 0.7
        assert strategy.use_fallback == True
    
    def test_ml_strategy_feature_creation(self, sample_data):
        """Test ML strategy feature engineering"""
        config = {'use_fallback': False}
        strategy = MLStrategy(config)
        
        data_with_features = strategy.calculate_indicators(sample_data.copy())
        
        # Check that basic features are created
        assert 'price_change' in data_with_features.columns
        assert 'volatility_5' in data_with_features.columns
        assert 'momentum_5' in data_with_features.columns
        
        # Even without ML model, it should calculate traditional indicators
        assert 'rsi_14' in data_with_features.columns
        assert 'macd' in data_with_features.columns
    
    def test_strategy_factory(self):
        """Test strategy factory function"""
        config = {'rsi_period': 14}
        
        # Test creating RSI+MACD strategy
        rsi_strategy = create_strategy('RSI_MACD_Strategy', config)
        assert isinstance(rsi_strategy, RSIMACDStrategy)
        
        # Test creating Bollinger Bands strategy
        bb_config = {'bb_period': 20}
        bb_strategy = create_strategy('BollingerBands_Strategy', bb_config)
        assert isinstance(bb_strategy, BollingerBandsStrategy)
        
        # Test error for unknown strategy
        with pytest.raises(ValueError):
            create_strategy('Unknown_Strategy', config)
    
    def test_position_management(self, sample_data):
        """Test position management functionality"""
        config = {'rsi_period': 14}
        strategy = RSIMACDStrategy(config)
        
        # Test entering long position
        strategy.update_position('BUY', 100.0, datetime.now(), 'test entry')
        assert strategy.position == 'long'
        assert strategy.entry_price == 100.0
        assert len(strategy.trade_history) == 1
        assert strategy.trade_history[0]['action'] == 'ENTER'
        
        # Test exiting position
        strategy.update_position('EXIT', 105.0, datetime.now(), 'test exit')
        assert strategy.position is None
        assert strategy.entry_price is None
        assert len(strategy.trade_history) == 2
        assert strategy.trade_history[1]['action'] == 'EXIT'
        assert strategy.trade_history[1]['profit_pct'] == 5.0
    
    def test_risk_management(self, sample_data):
        """Test risk management functionality"""
        config = {
            'rsi_period': 14,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'use_trailing_stop': True,
            'trailing_stop_pct': 1.5
        }
        strategy = RSIMACDStrategy(config)
        
        # Enter long position
        strategy.update_position('BUY', 100.0, datetime.now())
        
        # Test stop loss hit
        risk_signal = strategy.check_risk_management(98.0, datetime.now())
        assert risk_signal is not None
        assert risk_signal['signal'] == 'EXIT'
        assert 'Stop loss' in risk_signal['reason']
        
        # Re-enter position for take profit test
        strategy.update_position('BUY', 100.0, datetime.now())
        risk_signal = strategy.check_risk_management(104.0, datetime.now())
        assert risk_signal is not None
        assert risk_signal['signal'] == 'EXIT'
        assert 'Take profit' in risk_signal['reason']
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        config = {'rsi_period': 14}
        strategy = RSIMACDStrategy(config)
        
        # Add some trade history
        strategy.trade_history = [
            {'action': 'ENTER', 'position': 'long', 'price': 100, 'timestamp': datetime.now()},
            {'action': 'EXIT', 'position': 'long', 'entry_price': 100, 'exit_price': 105, 
             'profit_pct': 5.0, 'timestamp': datetime.now()},
            {'action': 'ENTER', 'position': 'short', 'price': 105, 'timestamp': datetime.now()},
            {'action': 'EXIT', 'position': 'short', 'entry_price': 105, 'exit_price': 102, 
             'profit_pct': 2.86, 'timestamp': datetime.now()},
        ]
        
        metrics = strategy.get_performance_metrics()
        
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'avg_profit' in metrics
        assert 'profit_factor' in metrics
        
        assert metrics['total_trades'] == 2
        assert metrics['win_rate'] == 1.0  # Both trades profitable
        assert abs(metrics['avg_profit'] - 3.93) < 0.1  # Average of 5% and 2.86%

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```


7. Example Usage & Backtesting


```python
# examples/strategy_backtest.py
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import create_strategy
from data.collector import DataCollector

def run_strategy_backtest():
    """Example of how to backtest a strategy"""
    
    # 1. Get historical data
    collector = DataCollector()
    data = collector.get_historical_data('BTC/USDT', '1h', limit=1000)
    
    # 2. Create strategy
    strategy_config = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0,
        'use_trailing_stop': True
    }
    
    strategy = create_strategy('RSI_MACD_Strategy', strategy_config)
    
    # 3. Run backtest simulation
    initial_capital = 10000
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(data)):
        if i < 50:  # Wait for enough data
            continue
            
        current_data = data.iloc[:i+1].copy()
        current_data = strategy.calculate_indicators(current_data)
        
        # Get current price and timestamp
        current_price = current_data['close'].iloc[-1]
        current_time = current_data.index[-1]
        
        # Check risk management first
        risk_signal = strategy.check_risk_management(current_price, current_time)
        if risk_signal and strategy.position:
            # Execute risk-based exit
            if position > 0:
                capital = position * current_price
                position = 0
            trades.append(('RISK_EXIT', current_price, current_time, risk_signal['reason']))
        
        # Check for entry/exit signals
        if strategy.position is None:
            # Look for entry
            entry_signal = strategy.should_enter(current_data)
            if entry_signal['signal'] in ['BUY', 'SELL'] and entry_signal['confidence'] > 0.6:
                if entry_signal['signal'] == 'BUY':
                    position = capital / current_price
                    capital = 0
                else:  # SELL (short - simplified)
                    # For simplicity, we'll just do long-only in this example
                    pass
                
                strategy.update_position(
                    entry_signal['signal'], 
                    current_price, 
                    current_time,
                    entry_signal['metadata']['reason']
                )
                trades.append(('ENTER', current_price, current_time, entry_signal['signal']))
        
        else:
            # Look for exit
            exit_signal = strategy.should_exit(current_data)
            if exit_signal['signal'] == 'EXIT':
                if position > 0:
                    capital = position * current_price
                    position = 0
                
                strategy.update_position(
                    'EXIT',
                    current_price,
                    current_time,
                    exit_signal['reason']
                )
                trades.append(('EXIT', current_price, current_time, exit_signal['reason']))
    
    # 4. Calculate final results
    if position > 0:
        capital = position * data['close'].iloc[-1]
    
    total_return = (capital - initial_capital) / initial_capital * 100
    total_trades = len([t for t in trades if t[0] == 'ENTER'])
    
    print("=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    
    # Strategy performance metrics
    metrics = strategy.get_performance_metrics()
    if metrics:
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Average Profit: {metrics['avg_profit']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    return trades, metrics

if __name__ == "__main__":
    trades, metrics = run_strategy_backtest()
```

🚀 Key Features Implemented
✅ Modular Strategy Architecture - Easy to extend
✅ Comprehensive Risk Management - Stop loss, take profit, trailing stops
✅ Confidence Scoring - Quantified signal strength
✅ ML Integration - With traditional fallback
✅ Performance Metrics - Win rate, profit factor, etc.
✅ Unit Tests - Comprehensive testing
✅ Position Management - Track entry/exit with metadata

📊 Strategy Comparison
Strategy	Best For	Risk	Complexity
RSI+MACD	Trending markets	Medium	Low
Bollinger Bands	Range-bound markets	Low	Medium
ML Strategy	Complex patterns	High	High
