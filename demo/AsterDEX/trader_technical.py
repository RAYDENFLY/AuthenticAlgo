"""
Strategy 1: Technical Analysis Only (RSI + MACD)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import ta
from base_trader import BaseTrader


class TechnicalAnalysisTrader(BaseTrader):
    """Pure technical analysis strategy"""
    
    def __init__(self, capital: float = 10.0):
        super().__init__("Technical_Analysis", capital)
        
        # TA parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_fast=self.macd_fast, 
                            window_slow=self.macd_slow, window_sign=self.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        return df
    
    def score_symbol(self, df: pd.DataFrame) -> float:
        """Score a symbol for trading potential (0-1)"""
        try:
            df = self.calculate_indicators(df)
            
            if df.empty or len(df) < 50:
                return 0.0
            
            latest = df.iloc[-1]
            score = 0.0
            
            # RSI signals
            if latest['rsi'] < self.rsi_oversold:
                score += 0.3  # Oversold = buy opportunity
            elif latest['rsi'] > self.rsi_overbought:
                score += 0.3  # Overbought = sell opportunity
            
            # MACD signals
            if latest['macd_hist'] > 0:
                score += 0.3  # Bullish
            
            # Trend signals
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                score += 0.2  # Uptrend
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                score += 0.2  # Downtrend
            
            # Volatility bonus (prefer volatile coins)
            atr_pct = latest['atr'] / latest['close']
            if atr_pct > 0.02:  # More than 2% ATR
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            return 0.0
    
    def screen_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Screen symbols and return best one"""
        best_symbol = None
        best_score = 0.0
        
        for symbol, df in data_dict.items():
            score = self.score_symbol(df)
            if score > best_score:
                best_score = score
                best_symbol = symbol
        
        return best_symbol if best_score > 0.3 else None
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate trading signal"""
        try:
            df = self.calculate_indicators(df)
            
            if df.empty or len(df) < 50:
                return {'direction': 'hold', 'confidence': 0.0, 'price': 0}
            
            latest = df.iloc[-1]
            signals = []
            
            # RSI signals
            if latest['rsi'] < self.rsi_oversold:
                signals.append(('long', 0.7))
            elif latest['rsi'] > self.rsi_overbought:
                signals.append(('short', 0.7))
            
            # MACD signals
            if latest['macd'] > latest['macd_signal']:
                signals.append(('long', 0.6))
            elif latest['macd'] < latest['macd_signal']:
                signals.append(('short', 0.6))
            
            # Price vs MA signals
            if latest['close'] > latest['sma_20']:
                signals.append(('long', 0.5))
            elif latest['close'] < latest['sma_20']:
                signals.append(('short', 0.5))
            
            if not signals:
                return {'direction': 'hold', 'confidence': 0.0, 'price': latest['close']}
            
            # Aggregate signals
            long_conf = np.mean([s[1] for s in signals if s[0] == 'long']) if any(s[0] == 'long' for s in signals) else 0
            short_conf = np.mean([s[1] for s in signals if s[0] == 'short']) if any(s[0] == 'short' for s in signals) else 0
            
            if long_conf > short_conf and long_conf > 0.5:
                direction = 'long'
                confidence = long_conf
            elif short_conf > long_conf and short_conf > 0.5:
                direction = 'short'
                confidence = short_conf
            else:
                direction = 'hold'
                confidence = 0.5
            
            return {
                'direction': direction,
                'confidence': confidence,
                'price': latest['close'],
                'rsi': latest['rsi'],
                'macd_hist': latest['macd_hist']
            }
            
        except Exception as e:
            return {'direction': 'hold', 'confidence': 0.0, 'price': 0}
    
    def calculate_leverage(self, signal: Dict) -> int:
        """Fixed leverage for Technical Analysis"""
        # Conservative: 10x-20x based on confidence
        confidence = signal.get('confidence', 0.5)
        leverage = int(10 + (confidence - 0.5) * 20)
        return max(10, min(20, leverage))
