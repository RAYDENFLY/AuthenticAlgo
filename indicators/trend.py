"""
Trend Indicators Module
Implements trend-following technical indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from core import get_logger


class TrendIndicators:
    """Trend-following indicators for identifying market direction"""
    
    def __init__(self):
        self.logger = get_logger()
    
    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods for SMA
            column: Column name to calculate SMA on
            
        Returns:
            SMA values as Series
        """
        return df[column].rolling(window=period).mean()
    
    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods for EMA
            column: Column name to calculate EMA on
            
        Returns:
            EMA values as Series
        """
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column name to calculate MACD on
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX)
        Measures trend strength (not direction)
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods for ADX
            
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / 
                         pd.Series(true_range).ewm(alpha=1/period, adjust=False).mean())
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / 
                          pd.Series(true_range).ewm(alpha=1/period, adjust=False).mean())
        
        # Calculate ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52
    ) -> dict:
        """
        Ichimoku Cloud
        Comprehensive indicator showing support, resistance, and trend
        
        Args:
            df: DataFrame with OHLCV data
            tenkan: Conversion line period
            kijun: Base line period
            senkou: Leading span B period
            
        Returns:
            Dictionary with all Ichimoku components
        """
        high = df['high']
        low = df['low']
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan).max() + 
                     low.rolling(window=tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun).max() + 
                    low.rolling(window=kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou).max() + 
                         low.rolling(window=senkou).min()) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = df['close'].shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def supertrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator
        Trend-following indicator with dynamic support/resistance
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (Supertrend values, Direction: 1=up, -1=down)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if i < period:
                supertrend.iloc[i] = 0
                direction.iloc[i] = 1
                continue
                
            # Current values
            current_close = close.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            prev_supertrend = supertrend.iloc[i-1]
            prev_direction = direction.iloc[i-1]
            
            if prev_supertrend == 0:  # Initialization
                supertrend.iloc[i] = current_upper
                direction.iloc[i] = 1
            else:
                if prev_direction == 1:  # Previous uptrend
                    if current_close > current_lower:
                        supertrend.iloc[i] = max(current_lower, prev_supertrend)
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = current_upper
                        direction.iloc[i] = -1
                else:  # Previous downtrend
                    if current_close < current_upper:
                        supertrend.iloc[i] = min(current_upper, prev_supertrend)
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = current_lower
                        direction.iloc[i] = 1
        
        return supertrend, direction
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all trend indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all trend indicators added
        """
        self.logger.debug("Calculating all trend indicators")
        
        # SMA & EMA
        df['sma_20'] = self.sma(df, 20)
        df['sma_50'] = self.sma(df, 50)
        df['sma_200'] = self.sma(df, 200)
        df['ema_9'] = self.ema(df, 9)
        df['ema_20'] = self.ema(df, 20)
        df['ema_50'] = self.ema(df, 50)
        
        # MACD
        macd, signal, histogram = self.macd(df)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # ADX
        adx, plus_di, minus_di = self.adx(df)
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Ichimoku
        ichimoku_data = self.ichimoku(df)
        for key, value in ichimoku_data.items():
            df[f'ichimoku_{key}'] = value
        
        # Supertrend
        supertrend, direction = self.supertrend(df)
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        
        self.logger.debug(f"Calculated {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} trend indicators")
        
        return df
    
    def __repr__(self) -> str:
        return "TrendIndicators(SMA, EMA, MACD, ADX, Ichimoku, Supertrend)"
