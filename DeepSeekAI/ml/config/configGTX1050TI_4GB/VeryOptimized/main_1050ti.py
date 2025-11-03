#!/usr/bin/env python3
"""
Main Entry Point for GTX 1050 Ti Optimized Trading Bot
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config
from core.logger import setup_logging
from core.gpu_manager import GPUManager1050Ti
from ml.optimized_feature_engine import OptimizedFeatureEngine
from ml.optimized_model_trainer import OptimizedModelTrainer
from ml.predictor import Predictor
from data.collector import DataCollector

def main():
    """Main function untuk 1050 Ti optimized bot"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("main.1050ti")
    
    # Load 1050Ti specific config
    config_path = Path("config/gtx_1050ti_config.yaml")
    if not config_path.exists():
        logger.error(f"1050Ti config not found: {config_path}")
        return
    
    config = load_config(config_path)
    logger.info("GTX 1050 Ti Optimized Trading Bot Starting...")
    
    # Initialize GPU Manager
    gpu_manager = GPUManager1050Ti(config)
    
    # Check system readiness
    is_safe, memory_status = gpu_manager.check_memory_safe()
    logger.info(f"Memory Status: {memory_status}")
    
    if not is_safe:
        logger.warning("System memory not optimal. Consider closing other applications.")
        # Bisa continue tapi dengan warning
    
    # Initialize components dengan optimized versions
    feature_engine = OptimizedFeatureEngine(config)
    model_trainer = OptimizedModelTrainer(config)
    predictor = Predictor(config)
    data_collector = DataCollector(config)
    
    logger.info("All components initialized successfully")
    
    # Example usage
    try:
        # Collect data (limited untuk hemat memory)
        logger.info("Collecting historical data...")
        data = data_collector.get_historical_data(
            symbol='BTC/USDT', 
            timeframe='1h', 
            limit=500  # Limit data points
        )
        
        # Prepare features
        logger.info("Generating optimized features...")
        features = feature_engine.transform(data, fit_scaler=True)
        
        logger.info(f"Generated {features.shape[1]} features from {len(data)} data points")
        
        # Train models
        logger.info("Training optimized models...")
        training_result = model_trainer.train_models(features, features['target'])
        
        if 'error' in training_result:
            logger.error(f"Training failed: {training_result['error']}")
            return
        
        logger.info(f"Training completed. Best model: {training_result['best_model']}")
        
        # Show performance
        for model, perf in training_result['performance'].items():
            logger.info(f"{model.upper()} - Test RMSE: {perf['test_rmse']:.4f}")
        
        # Ready for prediction
        logger.info("1050Ti Trading Bot is ready!")
        
        # Cleanup
        gpu_manager.clear_gpu_memory()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        # Cleanup even on error
        gpu_manager.clear_gpu_memory()

if __name__ == "__main__":
    main()