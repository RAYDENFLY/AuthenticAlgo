"""
Machine Learning Module
Feature engineering, model training, and prediction for trading strategies
"""

import pandas as pd
from core.logger import get_logger

from ml.feature_engine import FeatureEngine
from ml.model_trainer import ModelTrainer
from ml.predictor import Predictor


class MLModule:
    """
    Main interface for all ML components
    
    Integrates:
    - FeatureEngine: Feature extraction and engineering
    - ModelTrainer: Model training and evaluation
    - Predictor: Real-time prediction
    
    Example:
        config = {
            'feature_engineering': {...},
            'model_training': {...},
            'prediction': {...}
        }
        
        ml_module = MLModule(config)
        
        # Prepare features
        features = ml_module.prepare_features(data)
        
        # Train models
        results = ml_module.train_models(features, features['target'])
        
        # Make predictions
        prediction = ml_module.predict(data)
    """
    
    def __init__(self, config: dict):
        """Initialize ML module with configuration"""
        self.config = config
        self.logger = get_logger()
        
        self.feature_engine = FeatureEngine(config)
        self.model_trainer = ModelTrainer(config)
        self.predictor = Predictor(config)
        
        self.logger.info("ML Module initialized")
    
    def prepare_features(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Prepare features for training or prediction
        
        Args:
            data: Raw OHLCV data
            fit_scaler: Whether to fit scaler (True for training)
            
        Returns:
            DataFrame with engineered features
        """
        return self.feature_engine.transform(data, fit_scaler=fit_scaler)
    
    def train_models(self, data: pd.DataFrame, target: pd.Series) -> dict:
        """
        Train all models and return performance metrics
        
        Args:
            data: Feature data
            target: Target variable
            
        Returns:
            Dict with training results
        """
        return self.model_trainer.train_models(data, target)
    
    def predict(self, data: pd.DataFrame) -> dict:
        """
        Make predictions using trained models
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Dict with prediction and metadata
        """
        return self.predictor.predict(data)
    
    def get_feature_summary(self, data: pd.DataFrame) -> dict:
        """Get summary of engineered features"""
        return self.feature_engine.get_feature_summary(data)
    
    def get_model_summary(self) -> dict:
        """Get summary of trained models"""
        return self.model_trainer.get_model_summary()
    
    def get_prediction_stats(self) -> dict:
        """Get statistics about recent predictions"""
        return self.predictor.get_prediction_stats()


__all__ = ['MLModule', 'FeatureEngine', 'ModelTrainer', 'Predictor']

