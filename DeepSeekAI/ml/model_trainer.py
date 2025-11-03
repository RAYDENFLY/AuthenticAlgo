import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# XGBoost
try:
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    logging.warning("XGBoost not installed, skipping XGBoost models")

# TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_INSTALLED = True
except ImportError:
    TF_INSTALLED = False
    logging.warning("TensorFlow not installed, skipping LSTM models")

class ModelTrainer:
    """
    Machine Learning model trainer for trading strategies
    Supports multiple algorithms and hyperparameter optimization
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("ml.model_trainer")
        
        # Training configuration
        self.training_config = config.get('model_training', {})
        self.models_dir = Path(self.training_config.get('models_dir', 'ml/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_types = self.training_config.get('model_types', ['xgb', 'rf', 'linear'])
        self.validation_method = self.training_config.get('validation_method', 'timeseries')
        self.n_splits = self.training_config.get('n_splits', 5)
        self.test_size = self.training_config.get('test_size', 0.2)
        
        # Hyperparameter grids
        self.param_grids = self._get_param_grids()
        
        # Trained models
        self.trained_models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        self.logger.info("Model Trainer initialized")
    
    def train_models(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models and return performance metrics
        """
        self.logger.info("Starting model training...")
        
        # Prepare features and target
        X = data.drop(columns=['target'], errors='ignore')
        y = target
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Train models
        for model_type in self.model_types:
            self.logger.info(f"Training {model_type} model...")
            
            try:
                if model_type == 'xgb' and XGB_INSTALLED:
                    model, performance = self._train_xgboost(X_train, X_test, y_train, y_test)
                elif model_type == 'rf':
                    model, performance = self._train_random_forest(X_train, X_test, y_train, y_test)
                elif model_type == 'gbm':
                    model, performance = self._train_gradient_boosting(X_train, X_test, y_train, y_test)
                elif model_type == 'linear':
                    model, performance = self._train_linear_model(X_train, X_test, y_train, y_test)
                elif model_type == 'lstm' and TF_INSTALLED:
                    model, performance = self._train_lstm(X_train, X_test, y_train, y_test)
                else:
                    self.logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                self.trained_models[model_type] = model
                self.model_performance[model_type] = performance
                
                # Save model
                self._save_model(model, model_type)
                
            except Exception as e:
                self.logger.error(f"Error training {model_type} model: {e}")
                continue
        
        # Select best model
        best_model_type = self._select_best_model()
        self.logger.info(f"Best model: {best_model_type}")
        
        return {
            'best_model': best_model_type,
            'performance': self.model_performance,
            'feature_importance': self.feature_importance
        }
    
    def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict]:
        """Train XGBoost model with hyperparameter optimization"""
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Update with config if provided
        xgb_config = self.training_config.get('xgboost_params', {})
        params.update(xgb_config)
        
        model = xgb.XGBRegressor(**params)
        
        # Hyperparameter optimization
        if self.training_config.get('hyperparameter_optimization', False):
            param_grid = self.param_grids['xgb']
            grid_search = GridSearchCV(
                model, param_grid, cv=TimeSeriesSplit(n_splits=self.n_splits),
                scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.logger.info(f"XGBoost best params: {grid_search.best_params_}")
        
        else:
            model.fit(X_train, y_train)
        
        # Evaluate model
        performance = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Feature importance
        self.feature_importance['xgb'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, performance
    
    def _train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict]:
        """Train Random Forest model"""
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        rf_config = self.training_config.get('random_forest_params', {})
        params.update(rf_config)
        
        model = RandomForestRegressor(**params)
        
        if self.training_config.get('hyperparameter_optimization', False):
            param_grid = self.param_grids['rf']
            grid_search = GridSearchCV(
                model, param_grid, cv=TimeSeriesSplit(n_splits=self.n_splits),
                scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.logger.info(f"Random Forest best params: {grid_search.best_params_}")
        
        else:
            model.fit(X_train, y_train)
        
        performance = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.feature_importance['rf'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, performance
    
    def _train_gradient_boosting(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict]:
        """Train Gradient Boosting Machine"""
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        
        gbm_config = self.training_config.get('gradient_boosting_params', {})
        params.update(gbm_config)
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        performance = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.feature_importance['gbm'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, performance
    
    def _train_linear_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict]:
        """Train Linear Regression model with regularization"""
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        params = {
            'alpha': 1.0,
            'random_state': 42
        }
        
        linear_config = self.training_config.get('linear_params', {})
        params.update(linear_config)
        
        model = Ridge(**params)
        model.fit(X_train_scaled, y_train)
        
        performance = self._evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Feature importance for linear model (coefficients)
        self.feature_importance['linear'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        return model, performance
    
    def _train_lstm(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict]:
        """Train LSTM model for time series prediction"""
        # Reshape data for LSTM [samples, timesteps, features]
        sequence_length = self.training_config.get('sequence_length', 10)
        
        X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test.values, y_test.values, sequence_length)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            batch_size=32,
            epochs=100,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate model
        train_pred = model.predict(X_train_seq).flatten()
        test_pred = model.predict(X_test_seq).flatten()
        
        performance = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_seq, test_pred)),
            'train_mae': mean_absolute_error(y_train_seq, train_pred),
            'test_mae': mean_absolute_error(y_test_seq, test_pred),
            'train_r2': r2_score(y_train_seq, train_pred),
            'test_r2': r2_score(y_test_seq, test_pred),
            'history': history.history
        }
        
        return model, performance
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets using time series split"""
        if self.validation_method == 'timeseries':
            # Time series split (chronological)
            split_index = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
        
        self.logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def _evaluate_model(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
        }
        
        # Cross-validation scores
        if self.validation_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                  scoring='neg_mean_squared_error', n_jobs=-1)
        metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        self.logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}, Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def _select_best_model(self) -> str:
        """Select best model based on test RMSE"""
        if not self.model_performance:
            return None
        
        best_model = None
        best_score = float('inf')
        
        for model_type, performance in self.model_performance.items():
            if performance['test_rmse'] < best_score:
                best_score = performance['test_rmse']
                best_model = model_type
        
        return best_model
    
    def _save_model(self, model: Any, model_type: str):
        """Save trained model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.models_dir / f"{model_type}_model_{timestamp}.pkl"
        
        try:
            if model_type == 'lstm':
                # Save Keras model separately
                model_path = self.models_dir / f"{model_type}_model_{timestamp}.h5"
                model.save(model_path)
            else:
                joblib.dump(model, filename)
            
            self.logger.info(f"Saved {model_type} model to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving {model_type} model: {e}")
    
    def load_model(self, model_type: str, model_path: str = None) -> Any:
        """Load trained model from disk"""
        if model_path is None:
            # Find latest model of specified type
            model_files = list(self.models_dir.glob(f"{model_type}_model_*.pkl"))
            if not model_files:
                self.logger.error(f"No saved model found for {model_type}")
                return None
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            if model_type == 'lstm':
                model = tf.keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)
            
            self.logger.info(f"Loaded {model_type} model from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading {model_type} model: {e}")
            return None
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for different models"""
        return {
            'xgb': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'linear': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'best_model': self._select_best_model(),
            'performance': self.model_performance,
            'feature_importance': {
                model: importance.head(10).to_dict('records') 
                for model, importance in self.feature_importance.items()
            }
        }
        
        return summary