"""
RSI + MACD Strategy
Combined momentum strategy using RSI and MACD indicators
Entry: RSI oversold + MACD bullish crossover
Exit: RSI overbought + MACD bearish crossover
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.base_strategy import BaseStrategy

from core.logger import logger


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
