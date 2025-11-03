"""
ML Pipeline Demo - Complete machine learning workflow

This demo includes GTX 1050 Ti optimized configuration examples.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml

from ml import MLModule
from core.logger import get_logger


def load_config_1050ti():
    """Load GTX 1050 Ti optimized configuration"""
    try:
        with open('config/ml_config_1050ti.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default optimized config if file not found
        return {
            'feature_engineering': {
                'lookback_periods': [10, 20],
                'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'atr', 'obv'],
                'price_features': ['returns', 'volatility', 'momentum'],
                'volume_features': ['volume_ratio'],
                'time_features': ['hour', 'day_of_week'],
                'feature_scaling': 'standard',
                'feature_selection': True,
                'n_features': 30
            },
            'model_training': {
                'model_types': ['xgb', 'rf'],
                'validation_method': 'timeseries',
                'test_size': 0.3,
                'n_splits': 3,
                'hyperparameter_optimization': False,
                'gpu_memory_limit_mb': 3072,
                'xgboost_params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor'
                },
                'random_forest_params': {
                    'n_estimators': 50,
                    'max_depth': 8,
                    'min_samples_split': 5,
                    'n_jobs': -1
                }
            },
            'prediction': {
                'model_types': ['xgb', 'rf'],
                'confidence_threshold': 0.7,
                'use_ensemble': True,
                'ensemble_weights': {
                    'xgb': 0.6,
                    'rf': 0.4
                }
            }
        }


def generate_sample_data(symbol: str = "BTC/USDT", days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='1H')  # Hourly data
    
    # Generate realistic price movements
    np.random.seed(42)
    base_price = 45000
    returns = np.random.normal(0.0001, 0.015, days*24)  # Small hourly returns
    prices = base_price * (1 + returns).cumprod()
    
    # Generate OHLCV
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, days*24)),
        'high': prices * (1 + np.random.uniform(0, 0.01, days*24)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, days*24)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, days*24)
    })
    
    data['symbol'] = symbol
    data.set_index('timestamp', inplace=True)
    
    return data


def demo_feature_engineering():
    """Demo 1: Feature Engineering with GTX 1050 Ti optimization"""
    logger = get_logger()
    logger.info("=" * 80)
    logger.info("DEMO 1: Feature Engineering (GTX 1050 Ti Optimized)")
    logger.info("=" * 80)
    
    # Load optimized configuration
    config = load_config_1050ti()
    
    logger.info("\n🎯 GTX 1050 Ti Configuration:")
    logger.info(f"   - Max Features: {config['feature_engineering']['n_features']}")
    logger.info(f"   - Lookback Periods: {config['feature_engineering']['lookback_periods']}")
    logger.info(f"   - Feature Selection: {config['feature_engineering']['feature_selection']}")
    logger.info(f"   - Technical Indicators: {len(config['feature_engineering']['technical_indicators'])}")
    
    # Use optimized configuration directly
    config_demo = config
    
    # Generate sample data (reduced for GTX 1050 Ti)
    logger.info("\n1. Generating sample data...")
    data = generate_sample_data("BTC/USDT", days=90)  # 3 months (reduced from 6)
    logger.info(f"   Data shape: {data.shape}")
    logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize ML module
    logger.info("\n2. Initializing ML Module...")
    ml_module = MLModule(config_demo)
    
    # Prepare features
    logger.info("\n3. Extracting features...")
    features = ml_module.prepare_features(data, fit_scaler=True)
    
    logger.info(f"\n   Features generated: {features.shape[1]} features")
    logger.info(f"   Samples: {features.shape[0]}")
    
    # Feature summary
    logger.info("\n4. Feature Summary:")
    summary = ml_module.get_feature_summary(features)
    logger.info(f"   Total features: {summary['total_features']}")
    
    for category, count in summary['feature_categories'].items():
        logger.info(f"   {category}: {count}")
    
    logger.info(f"   Missing values: {summary['missing_values']}")
    logger.info("\n✅ GTX 1050 Ti Mode: Features optimized for 4GB GPU")
    
    # Show sample features
    logger.info("\n5. Sample Features (first 5 rows, first 10 columns):")
    print(features.iloc[:5, :10].to_string())
    
    logger.info("\n✓ Demo 1 completed successfully")
    return features


def demo_model_training():
    """Demo 2: Model Training"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Model Training")
    logger.info("=" * 80)
    
    # Configuration
    config = {
        'feature_engineering': {
            'lookback_periods': [5, 10],
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands'],
            'price_features': ['returns', 'volatility'],
            'volume_features': ['volume_ratio'],
            'time_features': ['hour', 'day_of_week'],
            'feature_scaling': 'standard',
            'feature_selection': True,
            'n_features': 30
        },
        'model_training': {
            'model_types': ['linear', 'rf'],  # Start with simple models
            'validation_method': 'timeseries',
            'test_size': 0.2,
            'n_splits': 3,
            'hyperparameter_optimization': False,
            'models_dir': 'ml/models'
        }
    }
    
    # Generate data
    logger.info("\n1. Generating training data...")
    data = generate_sample_data("BTC/USDT", days=180)
    
    # Initialize ML module
    logger.info("\n2. Initializing ML Module...")
    ml_module = MLModule(config)
    
    # Prepare features
    logger.info("\n3. Preparing features...")
    features = ml_module.prepare_features(data, fit_scaler=True)
    
    # Remove rows with missing target
    features_clean = features.dropna(subset=['target'])
    logger.info(f"   Clean samples: {len(features_clean)}")
    
    # Train models
    logger.info("\n4. Training models...")
    training_results = ml_module.train_models(
        features_clean.drop(columns=['target']),
        features_clean['target']
    )
    
    # Display results
    logger.info("\n5. Training Results:")
    logger.info(f"   Best model: {training_results['best_model']}")
    
    logger.info("\n   Model Performance:")
    for model_type, performance in training_results['performance'].items():
        logger.info(f"   {model_type.upper()}:")
        logger.info(f"     Train RMSE: {performance['train_rmse']:.4f}")
        logger.info(f"     Test RMSE:  {performance['test_rmse']:.4f}")
        logger.info(f"     Test R²:    {performance['test_r2']:.4f}")
    
    # Feature importance
    logger.info("\n6. Top 10 Important Features:")
    for model_type, importance in training_results['feature_importance'].items():
        if importance is not None and len(importance) > 0:
            logger.info(f"\n   {model_type.upper()}:")
            top_10 = importance.head(10)
            for idx, row in top_10.iterrows():
                logger.info(f"     {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\n✓ Demo 2 completed successfully")
    return ml_module


def demo_prediction():
    """Demo 3: Real-time Prediction"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: Real-time Prediction")
    logger.info("=" * 80)
    
    # Train models first
    logger.info("\n1. Training models (this may take a moment)...")
    ml_module = demo_model_training()
    
    # Generate new data for prediction
    logger.info("\n2. Generating new data for prediction...")
    new_data = generate_sample_data("BTC/USDT", days=7)  # 1 week of data
    
    # Make predictions
    logger.info("\n3. Making predictions...")
    prediction = ml_module.predict(new_data)
    
    # Display prediction results
    logger.info("\n4. Prediction Results:")
    logger.info(f"   Timestamp: {prediction['timestamp']}")
    logger.info(f"   Prediction: {prediction['prediction']:.4f}%")
    logger.info(f"   Confidence: {prediction['confidence']:.4f}")
    
    if prediction['individual_predictions']:
        logger.info("\n   Individual Model Predictions:")
        for model, pred in prediction['individual_predictions'].items():
            conf = prediction['individual_confidence'].get(model, 0.0)
            logger.info(f"     {model}: {pred:.4f}% (confidence: {conf:.4f})")
    
    logger.info(f"\n   Metadata:")
    logger.info(f"     Models used: {prediction['metadata']['models_used']}")
    logger.info(f"     Feature count: {prediction['metadata']['feature_count']}")
    
    # Get prediction statistics
    logger.info("\n5. Prediction Statistics:")
    stats = ml_module.get_prediction_stats()
    if 'total_predictions' in stats:
        logger.info(f"   Total predictions: {stats['total_predictions']}")
        logger.info(f"   Average prediction: {stats.get('avg_prediction', 0):.4f}%")
        logger.info(f"   Average confidence: {stats.get('avg_confidence', 0):.4f}")
        logger.info(f"   Recent trend: {stats.get('recent_trend', 'unknown')}")
    
    logger.info("\n✓ Demo 3 completed successfully")
    return prediction


def demo_full_pipeline():
    """Demo 4: Complete ML Pipeline"""
    logger = get_logger()
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 4: Complete ML Pipeline")
    logger.info("=" * 80)
    
    # Full configuration
    config = {
        'feature_engineering': {
            'lookback_periods': [5, 10, 20],
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'atr'],
            'price_features': ['returns', 'volatility', 'momentum'],
            'volume_features': ['volume_ratio', 'volume_velocity'],
            'time_features': ['hour', 'day_of_week', 'is_weekend'],
            'feature_scaling': 'standard',
            'feature_selection': True,
            'n_features': 40
        },
        'model_training': {
            'model_types': ['linear', 'rf', 'gbm'],
            'validation_method': 'timeseries',
            'test_size': 0.2,
            'n_splits': 3,
            'hyperparameter_optimization': False,
            'models_dir': 'ml/models'
        },
        'prediction': {
            'model_types': ['linear', 'rf', 'gbm'],
            'confidence_threshold': 0.6,
            'use_ensemble': True,
            'ensemble_weights': {
                'linear': 0.2,
                'rf': 0.4,
                'gbm': 0.4
            }
        }
    }
    
    # Initialize
    logger.info("\n1. Initializing ML Pipeline...")
    ml_module = MLModule(config)
    
    # Generate data
    logger.info("\n2. Loading historical data...")
    historical_data = generate_sample_data("BTC/USDT", days=180)
    logger.info(f"   Data points: {len(historical_data)}")
    
    # Feature engineering
    logger.info("\n3. Engineering features...")
    features = ml_module.prepare_features(historical_data, fit_scaler=True)
    features_clean = features.dropna(subset=['target'])
    
    summary = ml_module.get_feature_summary(features_clean)
    logger.info(f"   Total features: {summary['total_features']}")
    logger.info(f"   Clean samples: {len(features_clean)}")
    
    # Model training
    logger.info("\n4. Training models...")
    training_results = ml_module.train_models(
        features_clean.drop(columns=['target']),
        features_clean['target']
    )
    logger.info(f"   Best model: {training_results['best_model']}")
    
    # Model comparison
    logger.info("\n5. Model Comparison:")
    print(f"   {'Model':<10} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<10}")
    print(f"   {'-'*50}")
    for model, perf in training_results['performance'].items():
        print(f"   {model.upper():<10} {perf['train_rmse']:<12.4f} {perf['test_rmse']:<12.4f} {perf['test_r2']:<10.4f}")
    
    # Prediction
    logger.info("\n6. Making ensemble prediction...")
    recent_data = historical_data.iloc[-168:]  # Last week
    prediction = ml_module.predict(recent_data)
    
    logger.info(f"   Ensemble Prediction: {prediction['prediction']:.4f}%")
    logger.info(f"   Ensemble Confidence: {prediction['confidence']:.4f}")
    
    # Interpretation
    logger.info("\n7. Prediction Interpretation:")
    if prediction['prediction'] > 1.0:
        logger.info("   📈 BULLISH: Model predicts price increase")
    elif prediction['prediction'] < -1.0:
        logger.info("   📉 BEARISH: Model predicts price decrease")
    else:
        logger.info("   ➡️  NEUTRAL: Model predicts sideways movement")
    
    if prediction['confidence'] > 0.7:
        logger.info("   ✅ HIGH CONFIDENCE: Strong signal")
    elif prediction['confidence'] > 0.5:
        logger.info("   ⚠️  MEDIUM CONFIDENCE: Moderate signal")
    else:
        logger.info("   ❌ LOW CONFIDENCE: Weak signal")
    
    logger.info("\n✓ Demo 4 completed successfully")
    logger.info("=" * 80)


def main():
    """Run all ML demos"""
    logger = get_logger()
    
    try:
        logger.info("Starting ML Pipeline Demos...")
        logger.info("=" * 80)
        
        # Demo 1: Feature Engineering
        features = demo_feature_engineering()
        
        # Demo 2: Model Training
        ml_module = demo_model_training()
        
        # Demo 3: Prediction
        prediction = demo_prediction()
        
        # Demo 4: Full Pipeline
        demo_full_pipeline()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL ML DEMOS COMPLETED SUCCESSFULLY! ✓")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
