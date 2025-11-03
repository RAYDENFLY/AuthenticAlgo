import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import joblib
from pathlib import Path
from collections import deque

from core.logger import get_logger

class Predictor:
    """
    Real-time prediction engine for trading models
    Handles model loading, feature transformation, and prediction caching
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()
        
        # Prediction configuration
        self.prediction_config = config.get('prediction', {})
        self.models_dir = Path(self.prediction_config.get('models_dir', 'ml/models'))
        self.confidence_threshold = self.prediction_config.get('confidence_threshold', 0.7)
        self.use_ensemble = self.prediction_config.get('use_ensemble', True)
        self.ensemble_weights = self.prediction_config.get('ensemble_weights', {})
        self.prediction_cache_size = self.prediction_config.get('prediction_cache_size', 100)
        
        # Loaded models
        self.models = {}
        self.feature_engine = None
        self.scaler = None
        
        # Prediction cache
        self.prediction_cache = deque(maxlen=self.prediction_cache_size)
        
        # Initialize models
        self._load_models()
        
        self.logger.info("Predictor initialized")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using loaded models
        Returns: dict with predictions, confidence scores, and metadata
        """
        if not self.models:
            self.logger.error("No models loaded for prediction")
            return self._get_default_prediction()
        
        try:
            # Prepare features
            features = self._prepare_features(data)
            if features is None or features.empty:
                return self._get_default_prediction()
            
            # Make predictions with all models
            predictions = {}
            confidence_scores = {}
            
            for model_name, model in self.models.items():
                pred, confidence = self._predict_with_model(model, model_name, features)
                predictions[model_name] = pred
                confidence_scores[model_name] = confidence
            
            # Ensemble prediction
            ensemble_pred, ensemble_confidence = self._ensemble_predictions(predictions, confidence_scores)
            
            # Cache prediction
            self._cache_prediction(ensemble_pred, ensemble_confidence, features)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'prediction': ensemble_pred,
                'confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'individual_confidence': confidence_scores,
                'metadata': {
                    'models_used': list(self.models.keys()),
                    'feature_count': features.shape[1],
                    'cache_size': len(self.prediction_cache)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._get_default_prediction()
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for prediction"""
        try:
            # Use feature engine if available
            if self.feature_engine is not None:
                features = self.feature_engine.transform(data)
            else:
                # Basic feature preparation
                from ml.feature_engine import FeatureEngine
                self.feature_engine = FeatureEngine(self.config)
                features = self.feature_engine.transform(data)
            
            # Ensure we have the same features as during training
            if hasattr(self.feature_engine, 'selected_features') and self.feature_engine.selected_features:
                available_features = [f for f in self.feature_engine.selected_features if f in features.columns]
                features = features[available_features]
            
            # Handle missing values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature preparation error: {e}")
            return None
    
    def _predict_with_model(self, model: Any, model_name: str, features: pd.DataFrame) -> Tuple[float, float]:
        """Make prediction with a single model"""
        try:
            if model_name == 'lstm':
                # LSTM requires sequence data
                prediction = self._predict_lstm(model, features)
            else:
                # Standard sklearn-style models
                prediction = model.predict(features)[0]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(model, model_name, features, prediction)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction error with {model_name}: {e}")
            return 0.0, 0.0
    
    def _predict_lstm(self, model: Any, features: pd.DataFrame) -> float:
        """Make prediction with LSTM model"""
        # LSTM requires sequence data - for real-time prediction, we use the most recent sequence
        sequence_length = self.prediction_config.get('sequence_length', 10)
        
        if len(features) < sequence_length:
            self.logger.warning(f"Insufficient data for LSTM prediction. Need {sequence_length}, got {len(features)}")
            return 0.0
        
        # Use the most recent sequence
        sequence = features.iloc[-sequence_length:].values
        sequence = sequence.reshape(1, sequence_length, features.shape[1])
        
        prediction = model.predict(sequence, verbose=0)[0][0]
        return prediction
    
    def _calculate_confidence(self, model: Any, model_name: str, 
                            features: pd.DataFrame, prediction: float) -> float:
        """Calculate confidence score for prediction"""
        try:
            if model_name == 'lstm':
                # For LSTM, use prediction variance (if we had multiple models)
                return 0.8  # Placeholder
            
            elif hasattr(model, 'predict_proba'):
                # Classification models with probability
                proba = model.predict_proba(features)[0]
                confidence = np.max(proba)
                return confidence
            
            elif hasattr(model, 'decision_function'):
                # Models with decision function
                decision_scores = model.decision_function(features)
                confidence = self._sigmoid(np.abs(decision_scores[0]))
                return confidence
            
            else:
                # For regression models, use prediction stability
                if self.prediction_cache:
                    recent_predictions = [p['prediction'] for p in list(self.prediction_cache)[-5:]]
                    if recent_predictions:
                        std_dev = np.std(recent_predictions)
                        if std_dev > 0:
                            confidence = 1.0 / (1.0 + std_dev)
                        else:
                            confidence = 0.9
                    else:
                        confidence = 0.7
                else:
                    confidence = 0.7
                
                return confidence
                
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed for {model_name}: {e}")
            return 0.5
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for confidence normalization"""
        return 1 / (1 + np.exp(-x))
    
    def _ensemble_predictions(self, predictions: Dict[str, float], 
                            confidence_scores: Dict[str, float]) -> Tuple[float, float]:
        """Combine predictions from multiple models"""
        if not self.use_ensemble or len(predictions) == 0:
            # Return the first model's prediction
            first_model = next(iter(predictions))
            return predictions[first_model], confidence_scores[first_model]
        
        # Weighted average based on confidence and configured weights
        total_weight = 0.0
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        
        for model_name, prediction in predictions.items():
            # Get weight for this model
            if model_name in self.ensemble_weights:
                weight = self.ensemble_weights[model_name]
            else:
                weight = confidence_scores[model_name]
            
            weighted_prediction += prediction * weight
            weighted_confidence += confidence_scores[model_name] * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred = weighted_prediction / total_weight
            ensemble_confidence = weighted_confidence / total_weight
        else:
            ensemble_pred = 0.0
            ensemble_confidence = 0.0
        
        return ensemble_pred, ensemble_confidence
    
    def _cache_prediction(self, prediction: float, confidence: float, features: pd.DataFrame):
        """Cache prediction for future reference"""
        cache_entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'feature_hash': hash(features.to_string())  # Simplified feature hash
        }
        
        self.prediction_cache.append(cache_entry)
    
    def _load_models(self):
        """Load trained models from disk"""
        model_types = self.prediction_config.get('model_types', ['xgb', 'rf', 'linear'])
        
        for model_type in model_types:
            try:
                model_path = self.models_dir / f"{model_type}_model"
                
                # Look for the latest model file
                model_files = list(self.models_dir.glob(f"{model_type}_model_*.pkl"))
                if not model_files and model_type == 'lstm':
                    model_files = list(self.models_dir.glob(f"{model_type}_model_*.h5"))
                
                if not model_files:
                    self.logger.warning(f"No model files found for {model_type}")
                    continue
                
                # Get the most recent model
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                
                # Load model
                if model_type == 'lstm':
                    import tensorflow as tf
                    model = tf.keras.models.load_model(latest_model)
                else:
                    model = joblib.load(latest_model)
                
                self.models[model_type] = model
                self.logger.info(f"Loaded {model_type} model from {latest_model}")
                
            except Exception as e:
                self.logger.error(f"Error loading {model_type} model: {e}")
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when models fail"""
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction': 0.0,
            'confidence': 0.0,
            'individual_predictions': {},
            'individual_confidence': {},
            'metadata': {
                'models_used': [],
                'feature_count': 0,
                'cache_size': len(self.prediction_cache),
                'error': 'Prediction failed'
            }
        }
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about recent predictions"""
        if not self.prediction_cache:
            return {'message': 'No predictions in cache'}
        
        predictions = [p['prediction'] for p in self.prediction_cache]
        confidences = [p['confidence'] for p in self.prediction_cache]
        
        return {
            'total_predictions': len(self.prediction_cache),
            'avg_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'recent_trend': self._calculate_recent_trend(predictions)
        }
    
    def _calculate_recent_trend(self, predictions: List[float]) -> str:
        """Calculate trend of recent predictions"""
        if len(predictions) < 2:
            return "neutral"
        
        recent = predictions[-5:]  # Last 5 predictions
        if len(recent) < 2:
            return "neutral"
        
        # Simple linear trend
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        if slope > 0.001:
            return "upward"
        elif slope < -0.001:
            return "downward"
        else:
            return "neutral"
    
    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights based on recent performance"""
        self.ensemble_weights.update(new_weights)
        self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_model_health(self) -> Dict[str, bool]:
        """Check health of all loaded models"""
        health_status = {}
        
        for model_name, model in self.models.items():
            try:
                # Simple health check - try to get model parameters
                if hasattr(model, 'get_params'):
                    model.get_params()
                    health_status[model_name] = True
                else:
                    health_status[model_name] = True  # Assume healthy if we can't check
            except Exception as e:
                self.logger.warning(f"Health check failed for {model_name}: {e}")
                health_status[model_name] = False
        
        return health_status



