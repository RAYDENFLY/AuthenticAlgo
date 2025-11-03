"""
Momentum Indicators Module
Implements momentum and oscillator indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple

from core import get_logger


class MomentumIndicators:
    """Momentum and oscillator indicators for identifying overbought/oversold conditions"""
    
    def __init__(self):
        self.logger = get_logger()
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index
        Oscillator between 0-100, >70 overbought, <30 oversold
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods for RSI
            column: Column name to calculate RSI on
            
        Returns:
            RSI values as Series
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator %K and %D
        
        Args:
            df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period (SMA of %K)
            
        Returns:
            Tuple of (%K, %D)
        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R
        Momentum indicator, values between -100 and 0
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Williams %R values as Series
        """
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min + 1e-10))
        return williams_r
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        Oscillator with no bounds, typically between -100 and +100
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            CCI values as Series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        return cci
    
    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index
        Volume-weighted RSI, values between 0-100
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            MFI values as Series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1e-10))))
        return mfi
    
    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12, column: str = 'close') -> pd.Series:
        """
        Rate of Change
        Measures percentage change over period
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            column: Column name to calculate ROC on
            
        Returns:
            ROC values as Series
        """
        return ((df[column] - df[column].shift(period)) / (df[column].shift(period) + 1e-10)) * 100
    
    @staticmethod
    def momentum(df: pd.DataFrame, period: int = 10, column: str = 'close') -> pd.Series:
        """
        Momentum Indicator
        Simple difference between current and past price
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            column: Column name to calculate momentum on
            
        Returns:
            Momentum values as Series
        """
        return df[column] - df[column].shift(period)
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all momentum indicators added
        """
        self.logger.debug("Calculating all momentum indicators")
        
        # RSI
        df['rsi_14'] = self.rsi(df, 14)
        df['rsi_21'] = self.rsi(df, 21)
        
        # Stochastic
        stoch_k, stoch_d = self.stochastic(df)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = self.williams_r(df)
        
        # CCI
        df['cci'] = self.cci(df)
        
        # MFI
        df['mfi'] = self.mfi(df)
        
        # ROC
        df['roc_12'] = self.roc(df, 12)
        df['roc_25'] = self.roc(df, 25)
        
        # Momentum
        df['momentum_10'] = self.momentum(df, 10)
        
        self.logger.debug(f"Calculated {9} momentum indicators")
        
        return df
    
    def __repr__(self) -> str:
        return "MomentumIndicators(RSI, Stochastic, Williams%R, CCI, MFI, ROC, Momentum)"
