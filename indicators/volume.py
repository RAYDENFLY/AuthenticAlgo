"""
Volume Indicators Module  
Implements volume-based technical indicators
"""

import pandas as pd
import numpy as np
from typing import Dict

from core import get_logger


class VolumeIndicators:
    """Volume-based indicators for analyzing buying/selling pressure"""
    
    def __init__(self):
        self.logger = get_logger()
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            VWAP values as Series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_vp = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()
        return cumulative_vp / (cumulative_volume + 1e-10)
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume
        Running total of volume with direction
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            OBV values as Series
        """
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, price_bins: int = 20) -> Dict[float, float]:
        """
        Volume Profile (Volume at Price)
        
        Args:
            df: DataFrame with OHLCV data
            price_bins: Number of price bins
            
        Returns:
            Dictionary mapping price levels to volume
        """
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / price_bins
        
        volume_at_price = {}
        for i in range(len(df)):
            price_level = round(df['close'].iloc[i] / bin_size) * bin_size
            volume_at_price[price_level] = volume_at_price.get(price_level, 0) + df['volume'].iloc[i]
        
        return volume_at_price
    
    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow
        Measures buying and selling pressure
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            CMF values as Series
        """
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        mf_volume = mf_multiplier * df['volume']
        cmf = mf_volume.rolling(window=period).sum() / (df['volume'].rolling(window=period).sum() + 1e-10)
        return cmf
    
    @staticmethod
    def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Volume Simple Moving Average
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Volume SMA values as Series
        """
        return df['volume'].rolling(window=period).mean()
    
    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Volume Ratio (Current Volume vs Average Volume)
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods
            
        Returns:
            Volume ratio values as Series
        """
        avg_volume = df['volume'].rolling(window=period).mean()
        return df['volume'] / (avg_volume + 1e-10)
    
    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
        """
        Accumulation/Distribution Line
        Similar to OBV but considers price position within range
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            A/D Line values as Series
        """
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        clv = clv.fillna(0)  # Handle division by zero
        ad = (clv * df['volume']).cumsum()
        return ad
    
    @staticmethod
    def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
        """
        Force Index
        Combines price and volume to measure buying/selling pressure
        
        Args:
            df: DataFrame with OHLCV data
            period: EMA period
            
        Returns:
            Force Index values as Series
        """
        force = df['close'].diff() * df['volume']
        force_index = force.ewm(span=period, adjust=False).mean()
        return force_index
    
    @staticmethod
    def ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Ease of Movement
        Relates price change to volume
        
        Args:
            df: DataFrame with OHLCV data
            period: SMA period
            
        Returns:
            EOM values as Series
        """
        distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
        box_ratio = (df['volume'] / 1000000) / (df['high'] - df['low'] + 1e-10)
        eom = distance / (box_ratio + 1e-10)
        eom_sma = eom.rolling(window=period).mean()
        return eom_sma
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volume indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all volume indicators added
        """
        self.logger.debug("Calculating all volume indicators")
        
        # VWAP
        df['vwap'] = self.vwap(df)
        
        # OBV
        df['obv'] = self.obv(df)
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()  # OBV EMA for trend
        
        # CMF
        df['cmf'] = self.cmf(df)
        
        # Volume SMA and Ratio
        df['volume_sma_20'] = self.volume_sma(df, 20)
        df['volume_ratio'] = self.volume_ratio(df, 20)
        
        # Accumulation/Distribution
        df['ad_line'] = self.accumulation_distribution(df)
        
        # Force Index
        df['force_index'] = self.force_index(df)
        
        # Ease of Movement
        df['eom'] = self.ease_of_movement(df)
        
        self.logger.debug(f"Calculated {9} volume indicators")
        
        return df
    
    def __repr__(self) -> str:
        return "VolumeIndicators(VWAP, OBV, CMF, VolumeRatio, A/D, ForceIndex, EOM)"
