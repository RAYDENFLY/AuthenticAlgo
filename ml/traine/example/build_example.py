import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
from core.logger import get_logger
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scripts.quantum_ml_trainer_v6_0 import QuantumMLTrainerV60

warnings.filterwarnings('ignore')
logger = get_logger()

class QuantumOnlineLearner:
    def create_features_online(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for online phase, consistent with training."""
        return self.trainer.create_v6_features(df)
    """
    Real-time online learning for Quantum V6.0
    """
    
    def __init__(self, symbol: str, initial_model: Dict, config: Dict, collector=None):
        self.symbol = symbol
        self.model = initial_model
        self.config = config
        self.performance_history = []
        self.data_buffer = []
        self.collector = collector
        
        # Online learning parameters
        self.learning_rate = config.get('learning_rate', 0.02)
        self.update_interval = config.get('update_interval', 3600)  # 1 hour
        self.min_samples = config.get('min_samples', 4)  # Min 4 hours data
        
        # If collector is not set, try to get from config['trainer']
        if self.collector is None and 'trainer' in self.config:
            self.collector = getattr(self.config['trainer'], 'collector', None)
        
        # Store feature pipeline info from config for online learning
        self.expected_feature_count = self.config.get('expected_feature_count', 208)
        self.base_feature_cols = self.config.get('base_feature_cols', [])
        self.selector = self.config.get('selector', None)
        self.trainer = self.config.get('trainer', None)
        
        self.final_feature_cols = self.config.get('final_feature_cols', [])
        # Initialize QuantumMLTrainerV60 for feature engineering if not provided
        if self.trainer is None:
            self.trainer = QuantumMLTrainerV60({})
        
        logger.info(f"🎯 Online Learning Started for {symbol}")
        logger.info(f"   Update Interval: {self.update_interval}s")
        logger.info(f"   Learning Rate: {self.learning_rate}")
    
    async def get_recent_market_data(self, hours: int = 2) -> pd.DataFrame:
        """Get recent market data from AsterDEX"""
        try:
            # Get recent klines from AsterDEX
            # Ambil minimal 50 candle agar rolling features tidak gagal
            min_candles = max(hours, 80)
            klines = await self.collector.exchange.get_klines_safe(
                self.symbol, 
                '1h', 
                limit=min_candles
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"   📊 Got {len(df)} recent candles")
            return df
            
        except Exception as e:
            logger.error(f"   ❌ Failed to get market data: {e}")
            return pd.DataFrame()
        self.final_feature_cols = self.config.get('final_feature_cols', [])
        # Initialize QuantumMLTrainerV60 for feature engineering if not provided
        if self.trainer is None:
            self.trainer = QuantumMLTrainerV60({})
        """PAKAI FEATURE ENGINEERING YANG SAMA DENGAN TRAINING (V6.0)"""
        # Gunakan QuantumMLTrainerV60 untuk feature engineering yang konsisten
        trainer = QuantumMLTrainerV60({})
        return trainer.create_v6_features(df)
    
    def calculate_online_target(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate target for online learning"""
        if len(df) < 6:
            return np.array([])
            
        # Future return (6 hours ahead)
        df = df.copy()
        df['future_return'] = df['close'].pct_change(6).shift(-6).fillna(0)
        
        # Binary target based on median
        threshold = df['future_return'].median()
        targets = (df['future_return'] > threshold).astype(int)
        
        # Remove last 6 samples (no future data)
        targets.iloc[-6:] = np.nan
        
        return targets.dropna().values
    
    def add_multiscale_deep_features(self, df: pd.DataFrame, feature_cols: list):
        """Sama persis dengan QuantumMLTrainerV60.add_multiscale_deep_features"""
        import numpy as np
        from ml.temporal_cnn import TCNFeatureExtractor
        from ml.attention import AttentionFeatureExtractor
        X_base = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        if not hasattr(self, 'tcn_extractors'):
            self.tcn_extractors = {}
        if not hasattr(self, 'attention_extractor'):
            self.attention_extractor = None
        tcn_features_list = []
        # Shallow
        if 'shallow' not in self.tcn_extractors:
            self.tcn_extractors['shallow'] = TCNFeatureExtractor(
                input_size=len(feature_cols), sequence_length=15, num_channels=[64, 128], batch_size=32)
        try:
            tcn_shallow = self.tcn_extractors['shallow'].transform(X_base)
            tcn_features_list.append(tcn_shallow)
        except Exception as e:
            tcn_features_list.append(np.zeros((len(X_base), 256)))
        # Medium
        if 'medium' not in self.tcn_extractors:
            self.tcn_extractors['medium'] = TCNFeatureExtractor(
                input_size=len(feature_cols), sequence_length=20, num_channels=[64, 128, 256], batch_size=32)
        try:
            tcn_medium = self.tcn_extractors['medium'].transform(X_base)
            tcn_features_list.append(tcn_medium)
        except Exception as e:
            tcn_features_list.append(np.zeros((len(X_base), 512)))
        # Deep
        if 'deep' not in self.tcn_extractors:
            self.tcn_extractors['deep'] = TCNFeatureExtractor(
                input_size=len(feature_cols), sequence_length=25, num_channels=[64, 128, 256, 512], batch_size=32)
        try:
            tcn_deep = self.tcn_extractors['deep'].transform(X_base)
            tcn_features_list.append(tcn_deep)
        except Exception as e:
            tcn_features_list.append(np.zeros((len(X_base), 1024)))
        tcn_features = np.hstack(tcn_features_list)
        # Attention
        if self.attention_extractor is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.attention_extractor = AttentionFeatureExtractor(
                input_size=len(feature_cols), sequence_length=20, d_model=384, batch_size=32, device=device)
        try:
            attention_features = self.attention_extractor.transform(X_base)
        except Exception as e:
            attention_features = np.zeros((len(X_base), 256))
        return tcn_features, attention_features

    async def online_model_update(self, new_data: pd.DataFrame):
        """Update model with new data using full V6.0 feature pipeline"""
        if len(new_data) < self.min_samples:
            logger.info(f"   ⏳ Not enough data ({len(new_data)} < {self.min_samples})")
            return False
        try:
            # Use trainer's create_v6_features for consistent feature engineering
            feature_df = self.trainer.create_v6_features(new_data)
            X_base = feature_df[self.base_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
            tcn_features, attention_features = self.trainer.add_multiscale_deep_features(feature_df, self.base_feature_cols)
            X_dl = np.hstack([tcn_features, attention_features])
            X_dl_selected = self.selector.transform(X_dl)
            X_combined = np.hstack([X_base, X_dl_selected])
            # Build DataFrame with correct columns for downstream use
            if self.final_feature_cols and X_combined.shape[1] == len(self.final_feature_cols):
                X_combined_df = pd.DataFrame(X_combined, columns=self.final_feature_cols)
            else:
                X_combined_df = pd.DataFrame(X_combined)
            logger.info(f"   ✅ Feature shape: {X_combined_df.shape} (Expected: {self.expected_feature_count})")
            if X_combined_df.shape[1] != self.expected_feature_count:
                logger.error(f"   ❌ Feature mismatch: {X_combined_df.shape[1]} vs {self.expected_feature_count}")
                return False
            targets = self.calculate_online_target(feature_df)
            if len(targets) == 0:
                return False
            for model_name in ['xgb', 'lgbm', 'catboost']:
                if model_name in self.model:
                    model = self.model[model_name]
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(X_combined[:len(targets)], targets, classes=[0, 1])
            logger.info(f"   🔄 Model updated with {len(targets)} samples")
            return True
        except Exception as e:
            logger.error(f"   ❌ Online update failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _mini_batch_update(self, model, X_new: np.ndarray, y_new: np.ndarray):
        """Mini-batch update for models without partial_fit"""
        try:
            # Get current model parameters and create new model
            if hasattr(model, 'get_params'):
                params = model.get_params()
                
                # Create new model with same parameters
                if 'XGB' in str(type(model)):
                    new_model = XGBClassifier(**params)
                elif 'LGBM' in str(type(model)):
                    new_model = LGBMClassifier(**params)
                elif 'CatBoost' in str(type(model)):
                    new_model = CatBoostClassifier(**params)
                else:
                    return
                
                # Fit with combined data (simplified)
                new_model.fit(X_new, y_new)
                self.model = new_model
                
        except Exception as e:
            logger.warning(f"   ⚠️ Mini-batch update failed: {e}")
    
    async def monitor_performance(self):
        """Monitor model performance in real-time"""
        try:
            # Get recent data for validation
            val_data = await self.get_recent_market_data(hours=24)
            if len(val_data) < 10:
                return
                
            # Create features and targets
            feature_df = self.create_features_online(val_data)
            targets = self.calculate_online_target(feature_df)
            
            if len(targets) < 5:
                return
            
            feature_cols = [col for col in feature_df.columns if col not in [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore', 'future_return'
            ]]
            
            X_val = feature_df[feature_cols].iloc[:len(targets)].values
            
            # Get ensemble predictions
            predictions = self.predict_proba(X_val)
            predicted_classes = (predictions >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(targets, predicted_classes)
            auc = roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.5
            
            # Store performance
            performance = {
                'timestamp': datetime.now(),
                'accuracy': accuracy,
                'auc': auc,
                'samples_used': len(X_val),
                'confidence_mean': np.mean(predictions),
                'confidence_std': np.std(predictions)
            }
            
            self.performance_history.append(performance)
            
            logger.info(f"   📈 Performance Update - Acc: {accuracy:.3f}, AUC: {auc:.3f}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Performance monitoring failed: {e}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions"""
        try:
            weights = self.model.get('weights', {'xgb': 0.33, 'lgbm': 0.33, 'catboost': 0.34})
            
            predictions = []
            for model_name, weight in weights.items():
                if model_name in self.model:
                    model = self.model[model_name]
                    pred = model.predict_proba(X)[:, 1]
                    predictions.append(pred * weight)
            
            if predictions:
                return np.sum(predictions, axis=0)
            else:
                return np.zeros(len(X))
                
        except Exception as e:
            logger.warning(f"   ⚠️ Prediction failed: {e}")
            return np.zeros(len(X))
    
    async def run_continuous_learning(self, duration_days: int = None, duration_hours: int = None):
        """Run continuous learning for specified duration (days or hours)"""
        start_time = datetime.now()
        if duration_hours is not None:
            end_time = start_time + timedelta(hours=duration_hours)
            logger.info(f"🚀 Starting {duration_hours}-hour continuous learning")
        else:
            if duration_days is None:
                duration_days = 3
            end_time = start_time + timedelta(days=duration_days)
            logger.info(f"🚀 Starting {duration_days}-day continuous learning")
        logger.info(f"   Start: {start_time}")
        logger.info(f"   End: {end_time}")
        iteration = 0
        while datetime.now() < end_time:
            iteration += 1
            current_time = datetime.now()
            logger.info(f"\n⏰ Iteration {iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                # 1. Get recent market data
                new_data = await self.get_recent_market_data(hours=6)
                if len(new_data) > 0:
                    # 2. Update model with new data
                    update_success = await self.online_model_update(new_data)
                    if update_success:
                        # 3. Monitor performance every 3 iterations
                        if iteration % 3 == 0:
                            await self.monitor_performance()
                        # 4. Save checkpoint every 6 iterations
                        if iteration % 6 == 0:
                            self.save_checkpoint(iteration)
                # 5. Wait for next update
                logger.info(f"   💤 Waiting {self.update_interval}s for next update...")
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"   ❌ Iteration {iteration} failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
        # Final performance report
        await self.generate_final_report()
    
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model': self.model,
                'performance_history': self.performance_history,
                'iteration': iteration,
                'timestamp': datetime.now()
            }
            
            checkpoint_file = f"v60_online_checkpoint_{self.symbol}_iter{iteration}.pkl"
            
            import pickle
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            logger.info(f"   💾 Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Checkpoint save failed: {e}")
    
    async def generate_final_report(self):
        """Generate final performance report"""
        if not self.performance_history:
            logger.info("   No performance data collected")
            return
        
        # Create performance plot
        plt.figure(figsize=(12, 8))
        
        times = [p['timestamp'] for p in self.performance_history]
        accuracies = [p['accuracy'] for p in self.performance_history]
        aucs = [p['auc'] for p in self.performance_history]
        
        plt.subplot(2, 1, 1)
        plt.plot(times, accuracies, 'b-', label='Accuracy', linewidth=2)
        plt.ylabel('Accuracy')
        plt.title(f'Quantum V6.0 Online Learning - {self.symbol}')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(times, aucs, 'r-', label='AUC', linewidth=2)
        plt.ylabel('AUC Score')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        
        plot_file = f"v60_online_learning_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate improvements
        initial_acc = accuracies[0] if accuracies else 0
        final_acc = accuracies[-1] if accuracies else 0
        initial_auc = aucs[0] if aucs else 0
        final_auc = aucs[-1] if aucs else 0
        
        logger.info(f"\n{'='*60}")
        logger.info("🎯 ONLINE LEARNING FINAL REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Duration: {len(self.performance_history)} updates")
        logger.info(f"   Accuracy: {initial_acc:.3f} → {final_acc:.3f} ({final_acc-initial_acc:+.3f})")
        logger.info(f"   AUC: {initial_auc:.3f} → {final_auc:.3f} ({final_auc-initial_auc:+.3f})")
        logger.info(f"   Plot saved: {plot_file}")
        logger.info(f"{'='*60}")