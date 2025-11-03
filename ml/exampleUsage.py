# examples/ml_pipeline_example.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import MLModule
from data.collector import DataCollector

def run_ml_pipeline_example():
    """Example of using the ML pipeline"""
    
    # Configuration
    config = {
        'feature_engineering': {
            'lookback_periods': [5, 10, 20],
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'atr'],
            'price_features': ['returns', 'volatility', 'momentum'],
            'volume_features': ['volume_ratio', 'volume_velocity'],
            'time_features': ['hour', 'day_of_week', 'is_weekend'],
            'feature_scaling': 'standard',
            'feature_selection': True,
            'n_features': 30
        },
        'model_training': {
            'model_types': ['xgb', 'rf', 'linear'],
            'validation_method': 'timeseries',
            'test_size': 0.2,
            'n_splits': 5,
            'hyperparameter_optimization': True,
            'models_dir': 'ml/models'
        },
        'prediction': {
            'model_types': ['xgb', 'rf', 'linear'],
            'confidence_threshold': 0.7,
            'use_ensemble': True,
            'ensemble_weights': {
                'xgb': 0.5,
                'rf': 0.3,
                'linear': 0.2
            }
        }
    }
    
    # Initialize ML module
    ml_module = MLModule(config)
    
    # 1. Get historical data
    collector = DataCollector()
    data = collector.get_historical_data('BTC/USDT', '1h', limit=1000)
    
    print("=== FEATURE ENGINEERING ===")
    # 2. Prepare features
    features = ml_module.prepare_features(data)
    print(f"Generated {features.shape[1]} features")
    
    # Feature summary
    feature_summary = ml_module.feature_engine.get_feature_summary(features)
    print(f"Feature categories: {feature_summary['feature_categories']}")
    
    # 3. Train models
    print("\n=== MODEL TRAINING ===")
    training_result = ml_module.train_models(features, features['target'])
    
    print(f"Best model: {training_result['best_model']}")
    for model, performance in training_result['performance'].items():
        print(f"{model.upper()} - Test RMSE: {performance['test_rmse']:.4f}, R²: {performance['test_r2']:.4f}")
    
    # 4. Make predictions
    print("\n=== PREDICTION ===")
    recent_data = data.iloc[-100:].copy()  # Use recent data for prediction
    prediction = ml_module.predict(recent_data)
    
    print(f"Ensemble Prediction: {prediction['prediction']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Models used: {prediction['metadata']['models_used']}")
    
    # 5. Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    for model_type, importance in training_result['feature_importance'].items():
        if importance is not None and len(importance) > 0:
            top_features = importance.head(3)
            print(f"{model_type.upper()} Top Features:")
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return ml_module, prediction

if __name__ == "__main__":
    ml_module, prediction = run_ml_pipeline_example()