"""
Custom Indicators Module
Implements composite and custom technical indicators
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Dict, List

from core import get_logger


class CustomIndicators:
    """Custom and composite indicators for advanced market analysis"""
    
    def __init__(self):
        self.logger = get_logger()
        self._cache = {}
    
    @lru_cache(maxsize=100)
    def cached_indicator(self, indicator_name: str, *args, **kwargs) -> str:
        """
        LRU cache for expensive indicator calculations
        
        Args:
            indicator_name: Name of the indicator
            *args: Arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        cache_key = f"{indicator_name}_{hash(str(args))}_{hash(str(kwargs))}"
        return cache_key
    
    def trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite Trend Strength Indicator
        Combines ADX, RSI, and MACD for overall trend strength
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Trend strength values (0-1 scale)
        """
        # Get or use default values
        adx = df.get('adx', pd.Series(50, index=df.index))
        rsi = df.get('rsi_14', pd.Series(50, index=df.index))
        macd_hist = df.get('macd_histogram', pd.Series(0, index=df.index))
        
        # Normalize and weight components
        adx_strength = np.clip((adx - 25) / 50, 0, 1)  # ADX > 25 indicates trend
        rsi_strength = 1 - 2 * abs(rsi - 50) / 100  # Centered around 50
        macd_strength = np.tanh(macd_hist * 10)  # Normalize MACD histogram
        
        # Combined strength (0-1 scale)
        trend_strength = (adx_strength * 0.4 + rsi_strength * 0.3 + abs(macd_strength) * 0.3)
        return pd.Series(trend_strength, index=df.index)
    
    def volatility_regime(self, df: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """
        Volatility Regime Detection (Low, Normal, High)
        
        Args:
            df: DataFrame with calculated indicators
            lookback: Lookback period for percentiles
            
        Returns:
            Volatility regime classification
        """
        # Use ATR percent if available, otherwise calculate volatility
        if 'atr_percent' in df.columns:
            volatility = df['atr_percent']
        else:
            volatility = self._calculate_volatility(df)
        
        # Use rolling percentiles to determine regime
        low_threshold = volatility.rolling(window=lookback).apply(
            lambda x: np.percentile(x.dropna(), 33) if len(x.dropna()) > 0 else 0
        )
        high_threshold = volatility.rolling(window=lookback).apply(
            lambda x: np.percentile(x.dropna(), 66) if len(x.dropna()) > 0 else 0
        )
        
        regime = pd.Series('Normal', index=df.index)
        regime[volatility < low_threshold] = 'Low'
        regime[volatility > high_threshold] = 'High'
        
        return regime
    
    def market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite Market Regime Detection
        Combines trend strength and volatility regime
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Market regime classification
        """
        trend_strength = self.trend_strength(df)
        volatility_regime = self.volatility_regime(df)
        
        # Define market regimes
        market_regime = pd.Series('Normal_Market', index=df.index)
        
        # Strong Trend, Low Vol - Best for trend following
        mask = (trend_strength > 0.7) & (volatility_regime == 'Low')
        market_regime[mask] = 'Strong_Trend_Low_Vol'
        
        # Strong Trend, High Vol - Trend but risky
        mask = (trend_strength > 0.7) & (volatility_regime == 'High')
        market_regime[mask] = 'Strong_Trend_High_Vol'
        
        # Weak Trend, Low Vol - Ranging market
        mask = (trend_strength < 0.3) & (volatility_regime == 'Low')
        market_regime[mask] = 'Ranging_Low_Vol'
        
        # Weak Trend, High Vol - Chaotic, avoid trading
        mask = (trend_strength < 0.3) & (volatility_regime == 'High')
        market_regime[mask] = 'Chaotic_High_Vol'
        
        return market_regime
    
    def support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, any]:
        """
        Dynamic Support and Resistance Levels
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for finding local extrema
            
        Returns:
            Dictionary with support and resistance levels
        """
        closes = df['close']
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            window_data = closes.iloc[i-window:i+window]
            current_price = closes.iloc[i]
            
            if current_price == window_data.max():
                resistance_levels.append(current_price)
            elif current_price == window_data.min():
                support_levels.append(current_price)
        
        # Return recent levels (last 5 of each)
        recent_resistance = sorted(list(set(resistance_levels)))[-5:] if resistance_levels else []
        recent_support = sorted(list(set(support_levels)))[-5:] if support_levels else []
        
        return {
            'support_levels': recent_support,
            'resistance_levels': recent_resistance,
            'current_support': min(recent_support) if recent_support else None,
            'current_resistance': max(recent_resistance) if recent_resistance else None
        }
    
    def price_momentum_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Custom Price Momentum Oscillator
        Deviation of ROC from its average
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for calculation
            
        Returns:
            PMO values as Series
        """
        roc = ((df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)) * 100
        roc_sma = roc.rolling(window=period).mean()
        pmo = roc - roc_sma
        return pmo
    
    def multi_timeframe_trend(
        self,
        df: pd.DataFrame,
        short_period: int = 20,
        medium_period: int = 50,
        long_period: int = 200
    ) -> pd.Series:
        """
        Multi-Timeframe Trend Alignment
        1 = all aligned up, -1 = all aligned down, 0 = mixed
        
        Args:
            df: DataFrame with OHLCV data
            short_period: Short MA period
            medium_period: Medium MA period
            long_period: Long MA period
            
        Returns:
            Trend alignment score (-1 to 1)
        """
        close = df['close']
        
        # Calculate EMAs
        ema_short = close.ewm(span=short_period, adjust=False).mean()
        ema_medium = close.ewm(span=medium_period, adjust=False).mean()
        ema_long = close.ewm(span=long_period, adjust=False).mean()
        
        # Check alignment
        uptrend = (close > ema_short) & (ema_short > ema_medium) & (ema_medium > ema_long)
        downtrend = (close < ema_short) & (ema_short < ema_medium) & (ema_medium < ema_long)
        
        alignment = pd.Series(0, index=df.index)
        alignment[uptrend] = 1
        alignment[downtrend] = -1
        
        return alignment
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Helper method to calculate volatility if not present
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for calculation
            
        Returns:
            Volatility values as Series
        """
        high_low_range = (df['high'] - df['low']) / (df['close'] + 1e-10)
        return high_low_range.rolling(window=period).std()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all custom indicators
        
        Args:
            df: DataFrame with OHLCV data and basic indicators
            
        Returns:
            DataFrame with all custom indicators added
        """
        self.logger.debug("Calculating all custom indicators")
        
        # Trend Strength
        df['trend_strength'] = self.trend_strength(df)
        
        # Volatility Regime
        df['volatility_regime'] = self.volatility_regime(df)
        
        # Market Regime
        df['market_regime'] = self.market_regime(df)
        
        # Price Momentum Oscillator
        df['price_momentum_oscillator'] = self.price_momentum_oscillator(df)
        
        # Multi-Timeframe Trend
        df['mtf_trend_alignment'] = self.multi_timeframe_trend(df)
        
        self.logger.debug(f"Calculated {5} custom indicators")
        
        return df
    
    def __repr__(self) -> str:
        return "CustomIndicators(TrendStrength, VolatilityRegime, MarketRegime, PMO, MTF)"
