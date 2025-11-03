"""
Technical Indicators Module
Comprehensive collection of trading indicators
"""

from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators
from .custom import CustomIndicators

from core import get_logger


class TechnicalIndicators:
    """Main interface for all technical indicators"""
    
    def __init__(self):
        self.logger = get_logger()
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()
        self.custom = CustomIndicators()
        
        self.logger.info("TechnicalIndicators initialized with 5 indicator modules")
    
    def get_all_indicators(self, df):
        """
        Calculate all indicators for a dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators calculated
        """
        self.logger.info(f"Calculating all indicators for {len(df)} rows")
        
        # Calculate indicators in order
        df = self.trend.calculate_all(df)
        df = self.momentum.calculate_all(df)
        df = self.volatility.calculate_all(df)
        df = self.volume.calculate_all(df)
        df = self.custom.calculate_all(df)
        
        indicator_cols = len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])
        self.logger.info(f"Calculated {indicator_cols} total indicators")
        
        return df
    
    def __repr__(self) -> str:
        return "TechnicalIndicators(Trend, Momentum, Volatility, Volume, Custom)"


__all__ = [
    'TechnicalIndicators',
    'TrendIndicators',
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'CustomIndicators'
]
