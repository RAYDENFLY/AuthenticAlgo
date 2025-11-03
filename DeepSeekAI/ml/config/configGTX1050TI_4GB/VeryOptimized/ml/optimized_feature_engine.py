"""
Optimized Feature Engine for GTX 1050 Ti
Reduced feature set untuk hemat memory
"""

import pandas as pd
import numpy as np
from typing import List
from .feature_engine import FeatureEngine

class OptimizedFeatureEngine(FeatureEngine):
    """Feature Engine yang dioptimalkan untuk 1050 Ti"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        # Override dengan setting yang lebih konservatif
        self.n_features = config.get('feature_engineering', {}).get('n_features', 25)
        self.logger.info(f"1050Ti Optimized Feature Engine - Max {self.n_features} features")
    
    def transform(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Transform dengan optimisasi memory"""
        # Gunakan method parent tapi dengan additional optimization
        df = super().transform(data, fit_scaler)
        
        # Additional memory optimization
        df = self._reduce_memory_usage(df)
        
        return df
    
    def _reduce_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage of DataFrame"""
        # Convert to float32 untuk hemat memory
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = df[col].astype(np.float32)
            
        # Convert to int16/int8 jika possible
        for col in df.select_dtypes(include=[np.int64]).columns:
            if df[col].max() < 32767 and df[col].min() > -32768:
                df[col] = df[col].astype(np.int16)
            elif df[col].max() < 127 and df[col].min() > -128:
                df[col] = df[col].astype(np.int8)
        
        self.logger.debug("Memory usage optimized for 1050Ti")
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical features yang dioptimalkan"""
        from indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Hanya calculate essential indicators
        # Trend
        df['sma_20'] = indicators.trend.sma(df, 20)
        df['ema_20'] = indicators.trend.ema(df, 20)
        
        # MACD (simplified)
        macd, signal, _ = indicators.trend.macd(df)
        df['macd'] = macd
        df['macd_signal'] = signal
        
        # RSI
        df['rsi_14'] = indicators.momentum.rsi(df, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = indicators.volatility.bollinger_bands(df, 20, 2.0)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df['atr_14'] = indicators.volatility.atr(df, 14)
        
        # Volume
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduced lag features untuk hemat memory"""
        # Hanya lag features yang essential
        essential_features = ['close', 'volume', 'rsi_14', 'macd']
        available_features = [f for f in essential_features if f in df.columns]
        
        for feature in available_features:
            # Hanya 3 lags saja
            for lag in [1, 2, 3]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df