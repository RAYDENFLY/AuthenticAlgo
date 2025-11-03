"""
Volatility Indicators Module
Implements volatility-based technical indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple

from core import get_logger


class VolatilityIndicators:
    """Volatility-based indicators for measuring price variance"""
    
    def __init__(self):
        self.logger = get_logger()
    
    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands (Upper, Middle, Lower)
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods for SMA
            std_dev: Standard deviation multiplier
            column: Column name to calculate bands on
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle_band = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range
        Measures market volatility
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods for ATR
            
        Returns:
            ATR values as Series
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift()
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def keltner_channels(
        df: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels (Upper, Middle, Lower)
        
        Args:
            df: DataFrame with OHLCV data
            ema_period: EMA period for middle line
            atr_period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (Upper Channel, Middle Line, Lower Channel)
        """
        middle = df['close'].ewm(span=ema_period, adjust=False).mean()
        atr = VolatilityIndicators.atr(df, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower
    
    @staticmethod
    def donchian_channels(
        df: pd.DataFrame,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels (Upper, Middle, Lower)
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Tuple of (Upper Channel, Middle Channel, Lower Channel)
        """
        upper = df['high'].rolling(window=period).max()
        lower = df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    @staticmethod
    def standard_deviation(
        df: pd.DataFrame,
        period: int = 20,
        column: str = 'close'
    ) -> pd.Series:
        """
        Standard Deviation
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            column: Column name to calculate std on
            
        Returns:
            Standard deviation values as Series
        """
        return df[column].rolling(window=period).std()
    
    @staticmethod
    def volatility_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Volatility Ratio (Current Range vs Average Range)
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Volatility ratio values as Series
        """
        current_range = (df['high'] - df['low']) / (df['close'] + 1e-10)
        avg_range = current_range.rolling(window=period).mean()
        return current_range / (avg_range + 1e-10)
    
    @staticmethod
    def historical_volatility(
        df: pd.DataFrame,
        period: int = 20,
        column: str = 'close'
    ) -> pd.Series:
        """
        Historical Volatility (Annualized)
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            column: Column name to calculate volatility on
            
        Returns:
            Historical volatility values as Series
        """
        log_returns = np.log(df[column] / df[column].shift())
        volatility = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all volatility indicators added
        """
        self.logger.debug("Calculating all volatility indicators")
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(df)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)  # Normalized width
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)  # Position within bands
        
        # ATR
        df['atr_14'] = self.atr(df, 14)
        df['atr_percent'] = self.atr(df, 14) / (df['close'] + 1e-10)  # Normalized ATR
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.keltner_channels(df)
        df['kc_upper'] = kc_upper
        df['kc_middle'] = kc_middle
        df['kc_lower'] = kc_lower
        
        # Donchian Channels
        dc_upper, dc_middle, dc_lower = self.donchian_channels(df)
        df['dc_upper'] = dc_upper
        df['dc_middle'] = dc_middle
        df['dc_lower'] = dc_lower
        
        # Standard Deviation
        df['std_dev_20'] = self.standard_deviation(df, 20)
        
        # Volatility Ratio
        df['volatility_ratio'] = self.volatility_ratio(df)
        
        # Historical Volatility
        df['historical_volatility'] = self.historical_volatility(df)
        
        self.logger.debug(f"Calculated {18} volatility indicators")
        
        return df
    
    def __repr__(self) -> str:
        return "VolatilityIndicators(BollingerBands, ATR, KeltnerChannels, Donchian, StdDev)"
