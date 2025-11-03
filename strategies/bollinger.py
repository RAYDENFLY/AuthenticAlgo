"""
Bollinger Bands Mean Reversion Strategy
Entry: Price touches band extremes with volume confirmation
Exit: Price reverts to middle band
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.base_strategy import BaseStrategy

from core.logger import logger


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
