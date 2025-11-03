import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engine import FeatureEngine
from ml.model_trainer import ModelTrainer
from ml.predictor import Predictor

class TestMachineLearning:
    """Test suite for machine learning module"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        np.random.seed(42)
        
        data = {
            'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(200) * 0.5) + 0.5,
            'low': 100 + np.cumsum(np.random.randn(200) * 0.5) - 0.5,
            'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'volume': np.random.randint(1000, 10000, 200)
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for ML module"""
        return {
            'feature_engineering': {
                'lookback_periods': [5, 10],
                'technical_indicators': ['rsi', 'macd'],
                'price_features': ['returns', 'volatility'],
                'volume_features': ['volume_ratio'],
                'time_features': ['hour', 'day_of_week'],
                'feature_scaling': 'standard',
                'feature_selection': True,
                'n_features': 20
            },
            'model_training': {
                'model_types': ['linear', 'rf'],
                'validation_method': 'timeseries',
                'test_size': 0.2,
                'n_splits': 3,
                'hyperparameter_optimization': False
            },
            'prediction': {
                'model_types': ['linear', 'rf'],
                'confidence_threshold': 0.6,
                'use_ensemble': True
            }
        }
    
    def test_feature_engine_creation(self, sample_config):
        """Test feature engine creation"""
        feature_engine = FeatureEngine(sample_config)
        
        assert feature_engine is not None
        assert len(feature_engine.lookback_periods) == 2
        assert 'rsi' in feature_engine.technical_indicators
    
    def test_feature_transformation(self, sample_config, sample_data):
        """Test feature transformation"""
        feature_engine = FeatureEngine(sample_config)
        features = feature_engine.transform(sample_data, fit_scaler=True)
        
        assert features is not None
        assert len(features) > 0
        assert 'target' in features.columns
        assert features.isnull().sum().sum() == 0
        
        # Check that we have the expected feature types
        feature_summary = feature_engine.get_feature_summary(features)
        assert feature_summary['total_features'] > 0
    
    def test_model_trainer_creation(self, sample_config):
        """Test model trainer creation"""
        model_trainer = ModelTrainer(sample_config)
        
        assert model_trainer is not None
        assert 'linear' in model_trainer.model_types
        assert 'rf' in model_trainer.model_types
    
    def test_model_training(self, sample_config, sample_data):
        """Test model training workflow"""
        # Prepare features
        feature_engine = FeatureEngine(sample_config)
        features = feature_engine.transform(sample_data, fit_scaler=True)
        
        # Train models
        model_trainer = ModelTrainer(sample_config)
        training_result = model_trainer.train_models(features, features['target'])
        
        assert training_result is not None
        assert 'best_model' in training_result
        assert 'performance' in training_result
        assert 'feature_importance' in training_result
        
        # Check that models were trained
        assert len(model_trainer.trained_models) > 0
        assert len(model_trainer.model_performance) > 0
    
    def test_predictor_creation(self, sample_config):
        """Test predictor creation"""
        predictor = Predictor(sample_config)
        
        assert predictor is not None
        assert predictor.confidence_threshold == 0.6
        assert predictor.use_ensemble == True
    
    def test_prediction(self, sample_config, sample_data):
        """Test prediction workflow"""
        # First train models
        feature_engine = FeatureEngine(sample_config)
        features = feature_engine.transform(sample_data, fit_scaler=True)
        
        model_trainer = ModelTrainer(sample_config)
        model_trainer.train_models(features, features['target'])
        
        # Then test prediction
        predictor = Predictor(sample_config)
        
        # Use recent data for prediction
        recent_data = sample_data.iloc[-50:].copy()
        prediction = predictor.predict(recent_data)
        
        assert prediction is not None
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'individual_predictions' in prediction
        assert 'metadata' in prediction
    
    def test_feature_importance(self, sample_config, sample_data):
        """Test feature importance calculation"""
        feature_engine = FeatureEngine(sample_config)
        features = feature_engine.transform(sample_data, fit_scaler=True)
        
        model_trainer = ModelTrainer(sample_config)
        model_trainer.train_models(features, features['target'])
        
        importance = model_trainer.feature_importance
        
        assert importance is not None
        assert len(importance) > 0
        
        # Check that we have importance for each trained model
        for model_type in model_trainer.trained_models.keys():
            assert model_type in importance
            assert len(importance[model_type]) > 0
    
    def test_prediction_cache(self, sample_config, sample_data):
        """Test prediction caching"""
        predictor = Predictor(sample_config)
        
        # Make multiple predictions
        for i in range(5):
            recent_data = sample_data.iloc[-(50+i):].copy()
            predictor.predict(recent_data)
        
        stats = predictor.get_prediction_stats()
        
        assert stats['total_predictions'] == 5
        assert 'avg_confidence' in stats
        assert 'recent_trend' in stats
    
    def test_model_health_check(self, sample_config, sample_data):
        """Test model health check"""
        # Train models first
        feature_engine = FeatureEngine(sample_config)
        features = feature_engine.transform(sample_data, fit_scaler=True)
        
        model_trainer = ModelTrainer(sample_config)
        model_trainer.train_models(features, features['target'])
        
        predictor = Predictor(sample_config)
        health = predictor.get_model_health()
        
        assert health is not None
        # Note: In test environment, models may not be loaded in predictor

if __name__ == '__main__':
    pytest.main([__file__, '-v'])