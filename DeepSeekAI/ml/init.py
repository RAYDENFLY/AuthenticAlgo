"""
Machine Learning Module
Feature engineering, model training, and prediction for trading strategies
"""

from .feature_engine import FeatureEngine
from .model_trainer import ModelTrainer
from .predictor import Predictor

class MLModule:
    """Main interface for all ML components"""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engine = FeatureEngine(config)
        self.model_trainer = ModelTrainer(config)
        self.predictor = Predictor(config)
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training or prediction"""
        return self.feature_engine.transform(data)
    
    def train_models(self, data: pd.DataFrame, target: pd.Series) -> dict:
        """Train all models and return performance metrics"""
        return self.model_trainer.train_models(data, target)
    
    def predict(self, data: pd.DataFrame) -> dict:
        """Make predictions using trained models"""
        return self.predictor.predict(data)

__all__ = ['MLModule', 'FeatureEngine', 'ModelTrainer', 'Predictor']


