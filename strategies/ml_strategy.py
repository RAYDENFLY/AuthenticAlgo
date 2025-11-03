"""
Machine Learning Based Strategy
Uses ML models with traditional indicator fallback
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from strategies.base_strategy import BaseStrategy
from strategies.rsi_macd import RSIMACDStrategy
from pathlib import Path

from core.logger import logger


class MLStrategy(BaseStrategy):
    """
    Machine Learning Based Strategy
    Uses trained ML models for prediction with traditional indicator fallback
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ML_Strategy", config)
        
        # ML model settings
        self.model_path = config.get('model_path', 'ml/models/')
        self.model_name = config.get('model_name', 'xgboost_model.pkl')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.use_fallback = config.get('use_fallback', True)
        
        # Feature settings
        self.lookback_period = config.get('lookback_period', 50)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        
        # Fallback strategy
        self.fallback_strategy = None
        if self.use_fallback:
            fallback_config = config.get('fallback_config', {})
            self.fallback_strategy = RSIMACDStrategy(fallback_config)
        
        # Load ML model (will be None until model is trained)
        self.model = self._load_model()
        
        self.logger.info(f"ML Strategy: {self.model_name}, "
                        f"Confidence Threshold: {self.confidence_threshold}")
    
    def _load_model(self) -> Optional[Any]:
        """Load trained ML model"""
        try:
            model_file = Path(self.model_path) / self.model_name
            if model_file.exists():
                # Future: Load model with joblib
                # model = joblib.load(model_file)
                self.logger.info(f"Model file found: {self.model_name}")
                return None  # Placeholder
            else:
                self.logger.warning(f"Model file not found: {model_file}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for ML model and fallback indicators"""
        # Calculate traditional indicators as features
        from indicators import TechnicalIndicators
        indicators = TechnicalIndicators()
        data = indicators.get_all_indicators(data)
        
        # Add additional ML features
        data = self._create_ml_features(data)
        
        return data
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model prediction"""
        # Price-based features
        data['price_change'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['open_close_ratio'] = data['close'] / data['open']
        
        # Volatility features
        data['volatility_5'] = data['close'].pct_change().rolling(5).std()
        data['volatility_20'] = data['close'].pct_change().rolling(20).std()
        
        # Momentum features
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
            data['volume_volatility'] = data['volume'].rolling(10).std()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            data[f'close_ma_{window}'] = data['close'].rolling(window).mean()
            if 'volume' in data.columns:
                data[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
        
        return data
    
    def should_enter(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Use ML model for prediction with fallback to traditional strategy
        """
        if len(data) < self.lookback_period:
            return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'Insufficient data'}}
        
        # Try ML signal first (if model is available)
        ml_signal = self._get_ml_signal(data)
        
        # Use ML signal if confidence is high enough
        if ml_signal and ml_signal['confidence'] >= self.confidence_threshold:
            return ml_signal
        
        # Fallback to traditional strategy
        if self.use_fallback and self.fallback_strategy:
            fallback_data = self.fallback_strategy.calculate_indicators(data.copy())
            fallback_signal = self.fallback_strategy.should_enter(fallback_data)
            
            if fallback_signal['signal'] != 'HOLD':
                fallback_signal['metadata']['source'] = 'fallback'
                return fallback_signal
        
        return {'signal': 'HOLD', 'confidence': 0, 'metadata': {'reason': 'No high-confidence signals'}}
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get trading signal from ML model"""
        if self.model is None:
            return None
        
        # TODO: Implement ML prediction when model is trained
        # For now, return None to use fallback
        return None
    
    def should_exit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for exit conditions using ML model and traditional methods
        """
        if self.position is None:
            return {'signal': 'HOLD', 'reason': 'No position to exit'}
        
        # Check ML model for exit signal
        ml_exit_signal = self._get_ml_exit_signal(data)
        if ml_exit_signal:
            return ml_exit_signal
        
        # Fallback to traditional exit logic
        if self.use_fallback and self.fallback_strategy:
            fallback_data = self.fallback_strategy.calculate_indicators(data.copy())
            fallback_exit = self.fallback_strategy.should_exit(fallback_data)
            
            if fallback_exit['signal'] == 'EXIT':
                fallback_exit['metadata']['source'] = 'fallback'
                return fallback_exit
        
        return {'signal': 'HOLD', 'reason': 'No exit conditions met'}
    
    def _get_ml_exit_signal(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get exit signal from ML model"""
        if self.model is None:
            return None
        
        # TODO: Implement ML exit logic
        return None
