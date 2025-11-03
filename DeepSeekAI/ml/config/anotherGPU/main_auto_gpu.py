#!/usr/bin/env python3
"""
Auto GPU Detection Main Entry Point
Automatically detects GPU and loads optimal configuration
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.logger import setup_logging
from core.gpu_detector import GPUDetector
from core.gpu_manager import GPUManager1050Ti

def main():
    """Main function with auto GPU detection"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("main.auto_gpu")
    
    # Detect GPU and load optimal config
    logger.info("Detecting GPU hardware...")
    gpu_detector = GPUDetector()
    recommended_config = gpu_detector.get_recommended_config()
    
    # Print detection results
    gpu_detector.print_gpu_info()
    
    # Load the recommended configuration
    config_path = Path(recommended_config)
    if not config_path.exists():
        logger.error(f"Recommended config not found: {config_path}")
        logger.info("Falling back to 1050Ti config")
        config_path = Path("config/gtx_1050ti_config.yaml")
    
    config = load_config(config_path)
    logger.info(f"Loaded configuration: {config_path}")
    
    # Initialize GPU Manager with detected settings
    gpu_manager = GPUManager1050Ti(config)
    
    # Check system readiness
    is_safe, memory_status = gpu_manager.check_memory_safe()
    logger.info(f"Memory Status: {memory_status}")
    
    if not is_safe:
        logger.warning("System memory not optimal. Consider closing other applications.")
    
    # Import and initialize appropriate components based on GPU vendor
    gpu_vendor = gpu_detector.gpu_info.get('vendor', 'unknown')
    
    if gpu_vendor == 'amd':
        # Use AMD-optimized components
        from ml.optimized_feature_engine import OptimizedFeatureEngine
        from ml.optimized_model_trainer import OptimizedModelTrainer
        feature_engine = OptimizedFeatureEngine(config)
        model_trainer = OptimizedModelTrainer(config)
        logger.info("Using AMD-optimized components")
    else:
        # Use standard components for NVIDIA/Intel
        from ml.feature_engine import FeatureEngine
        from ml.model_trainer import ModelTrainer
        feature_engine = FeatureEngine(config)
        model_trainer = ModelTrainer(config)
        logger.info("Using standard components")
    
    from ml.predictor import Predictor
    from data.collector import DataCollector
    
    predictor = Predictor(config)
    data_collector = DataCollector(config)
    
    logger.info("All components initialized successfully")
    
    # Continue with normal execution...
    try:
        # Collect data based on config
        logger.info("Collecting historical data...")
        symbols = config.get('data', {}).get('symbols', ['BTC/USDT'])
        data_points = config.get('data', {}).get('history_days', 90) * 24  # Approximate
        
        data = data_collector.get_historical_data(
            symbol=symbols[0],
            timeframe=config.get('data', {}).get('timeframe', '1h'),
            limit=min(data_points, 2000)  # Safety limit
        )
        
        # Prepare features
        logger.info("Generating features...")
        features = feature_engine.transform(data, fit_scaler=True)
        
        logger.info(f"Generated {features.shape[1]} features from {len(data)} data points")
        
        # Train models
        logger.info("Training models...")
        training_result = model_trainer.train_models(features, features['target'])
        
        if 'error' in training_result:
            logger.error(f"Training failed: {training_result['error']}")
            return
        
        logger.info(f"Training completed. Best model: {training_result['best_model']}")
        
        # Show performance
        for model, perf in training_result['performance'].items():
            logger.info(f"{model.upper()} - Test RMSE: {perf['test_rmse']:.4f}, R²: {perf['test_r2']:.4f}")
        
        logger.info("Auto GPU Trading Bot is ready!")
        
        # Cleanup
        gpu_manager.clear_gpu_memory()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        gpu_manager.clear_gpu_memory()

if __name__ == "__main__":
    main()