"""
Optimized Model Trainer for GTX 1050 Ti
Memory-efficient training untuk GPU 4GB
"""

import pandas as pd
import numpy as np
import logging
from .model_trainer import ModelTrainer
from core.gpu_manager import GPUManager1050Ti

class OptimizedModelTrainer(ModelTrainer):
    """Model trainer yang dioptimalkan untuk 1050 Ti"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.gpu_manager = GPUManager1050Ti(config)
        self.logger.info("1050Ti Optimized Model Trainer initialized")
    
    def train_models(self, data: pd.DataFrame, target: pd.Series) -> dict:
        """Training dengan memory monitoring"""
        # Check memory safety sebelum training
        is_safe, memory_status = self.gpu_manager.check_memory_safe()
        
        if not is_safe:
            self.logger.warning(f"Memory not safe for training: {memory_status}")
            self.logger.warning("Consider reducing data size or feature count")
            return {'error': 'Memory not safe for training'}
        
        self.logger.info("Memory status safe, starting training...")
        
        # Continue dengan parent training method
        result = super().train_models(data, target)
        
        # Clear memory setelah training
        self.gpu_manager.clear_gpu_memory()
        
        return result
    
    def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> tuple:
        """XGBoost training dengan GPU optimization"""
        import xgboost as xgb
        
        # Parameters khusus 1050 Ti
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            # GPU acceleration parameters
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0
        }
        
        # Update dengan config
        xgb_config = self.training_config.get('xgboost_params', {})
        params.update(xgb_config)
        
        model = xgb.XGBRegressor(**params)
        
        try:
            model.fit(X_train, y_train)
            performance = self._evaluate_model(model, X_train, X_test, y_train, y_test)
            
            # Feature importance
            self.feature_importance['xgb'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return model, performance
            
        except Exception as e:
            self.logger.error(f"XGBoost GPU training failed: {e}")
            self.logger.info("Falling back to CPU...")
            
            # Fallback ke CPU
            params['tree_method'] = 'hist'
            params['predictor'] = 'cpu_predictor'
            params.pop('gpu_id', None)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            performance = self._evaluate_model(model, X_train, X_test, y_train, y_test)
            
            return model, performance