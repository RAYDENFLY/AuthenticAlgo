"""
QUANTUM LEAP V5.1 - ADVANCED FEATURES ON SOLID FOUNDATION
Build on V4.1's balanced approach + Deep Learning features

V4.1 Foundation: AUC 0.699, Accuracy 65.67%
V5.1 Target: AUC 0.730-0.760, Accuracy 68-72%

Key Improvements:
1. Keep SMOTE balancing (proven to work!)
2. Add TCN temporal features (512-dim)
3. Add Attention regime-aware features (256-dim)
4. Calibrated ensemble for better confidence
5. Advanced feature engineering
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
from data.asterdex_collector import AsterDEXDataCollector

# V4.1 proven components
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# V5.1 advanced components
from ml.temporal_cnn import TCNFeatureExtractor
from ml.attention import AttentionFeatureExtractor

logger = setup_logger()


class QuantumMLTrainerV51:
    """
    QUANTUM LEAP V5.1 - ADVANCED ON SOLID BASE
    V4.1 balanced foundation + V5.0 deep learning
    """
    
    def __init__(self, config: Dict):
        self.collector = AsterDEXDataCollector(config)
        self.config = config
        
        # V5.1 Deep Learning Components
        self.tcn_extractor = None
        self.attention_extractor = None
        
        logger.info("🚀 QUANTUM LEAP V5.1 - ADVANCED FEATURES")
        logger.info("   Foundation: V4.1 SMOTE Balancing (AUC 0.699)")
        logger.info("   Enhancement: TCN + Attention + Calibration")
        logger.info("   Target: AUC 0.730-0.760, Accuracy 68-72%")
    
    async def get_data(self, symbol: str) -> pd.DataFrame:
        """Get data (same as V4.1)"""
        logger.info(f"📊 Collecting data for {symbol}...")
        
        klines = await self.collector.exchange.get_klines(symbol, '1h', 1000)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"   ✅ Collected {len(df)} candles")
        return df
    
    def create_advanced_features_v51(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V5.1: Enhanced feature engineering
        Keep V4.1 basics + Add advanced features
        """
        logger.info("🔬 V5.1 Advanced feature engineering...")
        
        df = df.copy()
        
        # === V4.1 CORE FEATURES (PROVEN) ===
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5).fillna(0)
        df['momentum_10'] = df['close'].pct_change(10).fillna(0)
        df['momentum_20'] = df['close'].pct_change(20).fillna(0)
        
        # Volatility
        df['volatility_10'] = df['close'].pct_change().rolling(10).std().fillna(0)
        df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # === V5.1 NEW FEATURES ===
        # 1. Momentum Acceleration (multi-timeframe)
        df['momentum_accel_5_10'] = df['momentum_5'] - df['momentum_10']
        df['momentum_accel_10_20'] = df['momentum_10'] - df['momentum_20']
        
        # 2. RSI Divergence
        df['rsi_momentum'] = df['rsi_14'].diff(3).fillna(0)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(10).mean().fillna(50)
        
        # 3. Volume-Price Confirmation
        df['volume_ma'] = df['volume'].rolling(20).mean().fillna(0)
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        price_change = df['close'].pct_change().fillna(0)
        volume_change = df['volume'].pct_change().fillna(0)
        df['volume_price_confirm'] = (np.sign(price_change) == np.sign(volume_change)).astype(int)
        
        # 4. Market Regime Score
        trend_strength = df['momentum_20'].abs()
        volatility_regime = df['volatility_20'] / df['volatility_20'].rolling(50).mean().replace(0, 1)
        df['regime_score'] = (trend_strength * 0.6 + volatility_regime * 0.4).fillna(0)
        
        # 5. Bollinger Band Position
        df['price_ma20'] = df['close'].rolling(20).mean().fillna(0)
        df['price_std20'] = df['close'].rolling(20).std().fillna(0)
        df['upper_band'] = df['price_ma20'] + 2 * df['price_std20']
        df['lower_band'] = df['price_ma20'] - 2 * df['price_std20']
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band']).replace(0, 1)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['price_ma20'].replace(0, 1)
        
        # 6. ATR-based features
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean().fillna(0)
        df['atr_ratio'] = df['atr_14'] / df['close'].replace(0, 1)
        
        logger.info(f"   ✅ Created 23 advanced features")
        return df
    
    def add_deep_learning_features(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        V5.1: Add TCN + Attention features
        """
        logger.info("🧠 Extracting deep learning features...")
        
        X_base = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        # TCN Features
        logger.info("   🔬 TCN temporal patterns...")
        if self.tcn_extractor is None:
            self.tcn_extractor = TCNFeatureExtractor(
                input_size=len(feature_cols),
                sequence_length=20,
                num_channels=[64, 128, 256],
                batch_size=32
            )
        
        try:
            tcn_features = self.tcn_extractor.transform(X_base)
            logger.info(f"      ✅ TCN: {tcn_features.shape}")
        except Exception as e:
            logger.warning(f"      ⚠️ TCN failed: {e}")
            tcn_features = np.zeros((len(X_base), 512))
        
        # Attention Features (regime-aware)
        logger.info("   🔬 Attention regime-aware...")
        if self.attention_extractor is None:
            self.attention_extractor = AttentionFeatureExtractor(
                input_size=len(feature_cols),
                sequence_length=20,
                d_model=256,
                batch_size=32
            )
        
        try:
            attention_features = self.attention_extractor.transform(X_base)
            logger.info(f"      ✅ Attention: {attention_features.shape}")
        except Exception as e:
            logger.warning(f"      ⚠️ Attention failed: {e}")
            attention_features = np.zeros((len(X_base), 256))
        
        return tcn_features, attention_features
    
    def create_balanced_target(self, df: pd.DataFrame, threshold_percentile: int = 60) -> pd.DataFrame:
        """Create balanced target (V4.1 proven method)"""
        logger.info("🎯 Creating balanced target...")
        
        df = df.copy()
        df['future_return'] = df['close'].pct_change(6).shift(-6).fillna(0)
        
        positive_threshold = df['future_return'].quantile(threshold_percentile / 100)
        logger.info(f"   Threshold (p{threshold_percentile}): {positive_threshold:.4f}")
        
        df['target'] = (df['future_return'] > positive_threshold).astype(int)
        
        class_dist = df['target'].value_counts(normalize=True)
        logger.info(f"   Class 0: {class_dist.get(0, 0)*100:.1f}%")
        logger.info(f"   Class 1: {class_dist.get(1, 0)*100:.1f}%")
        
        return df
    
    def apply_smote_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE (V4.1 proven method)"""
        logger.info("⚖️ Applying SMOTE balancing...")
        
        original_dist = pd.Series(y).value_counts()
        logger.info(f"   Original: Class 0={original_dist.get(0, 0)}, Class 1={original_dist.get(1, 0)}")
        
        over = SMOTE(sampling_strategy=0.8, random_state=42)
        under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        
        pipeline = ImbPipeline([('over', over), ('under', under)])
        
        try:
            X_balanced, y_balanced = pipeline.fit_resample(X, y)
            new_dist = pd.Series(y_balanced).value_counts()
            logger.info(f"   Balanced: Class 0={new_dist.get(0, 0)}, Class 1={new_dist.get(1, 0)}")
            logger.info(f"   ✅ SMOTE applied")
            return X_balanced, y_balanced
        except Exception as e:
            logger.warning(f"   ⚠️ SMOTE failed: {e}")
            return X, y
    
    def train_calibrated_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        V5.1: Calibrated ensemble for better confidence
        """
        logger.info("🏋️ Training calibrated ensemble...")
        
        # Calculate class weights
        class_counts = pd.Series(y_train).value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts.index else 1.0
        logger.info(f"   Scale pos weight: {scale_pos_weight:.2f}")
        
        # Base models
        base_xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )
        
        base_lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        
        # V5.1: Calibrate for better confidence
        logger.info("   📊 Calibrating models (Platt scaling)...")
        cal_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=3)
        cal_lgbm = CalibratedClassifierCV(base_lgbm, method='sigmoid', cv=3)
        
        # Train calibrated models
        logger.info("   Training calibrated XGBoost...")
        cal_xgb.fit(X_train, y_train)
        
        logger.info("   Training calibrated LightGBM...")
        cal_lgbm.fit(X_train, y_train)
        
        # Ensemble
        logger.info("   Creating voting ensemble...")
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', cal_xgb),
                ('lgbm', cal_lgbm)
            ],
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        ensemble_pred = ensemble.predict(X_test)
        ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        logger.info(f"   ✅ Calibrated Ensemble - Acc: {ensemble_acc*100:.1f}%, AUC: {ensemble_auc:.3f}")
        
        # Classification report
        logger.info("\n   📊 Classification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=['SELL', 'BUY']))
        
        return {
            'ensemble': ensemble,
            'accuracy': ensemble_acc,
            'auc': ensemble_auc,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
    
    async def train_v51(self, symbol: str) -> Dict:
        """
        Complete V5.1 training pipeline
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 QUANTUM V5.1 TRAINING: {symbol}")
        logger.info(f"{'='*70}")
        
        # 1. Get data
        df = await self.get_data(symbol)
        
        # 2. Create advanced features
        df = self.create_advanced_features_v51(df)
        
        # 3. Create balanced target
        df = self.create_balanced_target(df, threshold_percentile=60)
        
        # 4. Prepare base features
        feature_cols = [
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_20', 'atr_14', 'atr_ratio',
            'rsi_14', 'rsi_momentum', 'rsi_divergence',
            'volume_ratio', 'volume_price_confirm',
            'bb_position', 'bb_width', 'regime_score',
            'momentum_accel_5_10', 'momentum_accel_10_20'
        ]
        
        X_base = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['target']
        
        # Remove NaN targets
        valid_mask = ~y.isna()
        X_base = X_base[valid_mask]
        df_valid = df[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"   Valid samples: {len(X_base)}")
        
        if len(X_base) < 200:
            logger.error("   ❌ Insufficient data")
            return None
        
        # 5. Add deep learning features
        tcn_features, attention_features = self.add_deep_learning_features(df_valid, feature_cols)
        
        # 6. Combine all features
        logger.info("   🔗 Combining features...")
        logger.info(f"      Base: {X_base.shape}")
        logger.info(f"      TCN: {tcn_features.shape}")
        logger.info(f"      Attention: {attention_features.shape}")
        
        # Select top K from deep learning features to avoid overfitting
        selector = SelectKBest(score_func=f_classif, k=min(100, tcn_features.shape[1] + attention_features.shape[1]))
        X_dl = np.hstack([tcn_features, attention_features])
        selector.fit(X_dl, y)
        X_dl_selected = selector.transform(X_dl)
        
        # Final feature matrix
        X_combined = np.hstack([X_base.values, X_dl_selected])
        logger.info(f"      Combined: {X_combined.shape}")
        
        # 7. Stratified split
        logger.info("   Splitting data (stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, 
            test_size=0.2, 
            stratify=y,
            random_state=42
        )
        
        # 8. Apply SMOTE to training data
        X_train_balanced, y_train_balanced = self.apply_smote_balancing(X_train, y_train)
        
        # 9. Train calibrated ensemble
        results = self.train_calibrated_ensemble(
            X_train_balanced, y_train_balanced,
            X_test, y_test
        )
        
        results['symbol'] = symbol
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ {symbol} V5.1 COMPLETE")
        logger.info(f"   Accuracy: {results['accuracy']*100:.2f}%")
        logger.info(f"   AUC: {results['auc']:.3f}")
        logger.info(f"   Features: {X_combined.shape[1]} (Base={X_base.shape[1]}, DL={X_dl_selected.shape[1]})")
        logger.info(f"{'='*70}")
        
        return results


async def main():
    """Main V5.1 training loop"""
    
    config = {
        'exchange': 'asterdex',
        'api_key': 'dummy',
        'api_secret': 'dummy',
        'testnet': True
    }
    
    trainer = QuantumMLTrainerV51(config)
    
    coins = ['BTCUSDT', 'ETHUSDT', 'TRUMPUSDT']
    all_results = []
    
    for symbol in coins:
        try:
            result = await trainer.train_v51(symbol)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        valid_aucs = [r['auc'] for r in all_results if not np.isnan(r['auc'])]
        avg_auc = np.mean(valid_aucs) if valid_aucs else 0.5
        
        logger.info("\n" + "="*70)
        logger.info("🎯 QUANTUM V5.1 FINAL RESULTS")
        logger.info("="*70)
        logger.info(f"   Average Accuracy: {avg_acc*100:.2f}%")
        logger.info(f"   Average AUC: {avg_auc:.3f}")
        logger.info(f"   V4.1 Baseline: AUC 0.699, Acc 65.67%")
        logger.info(f"   V5.1 Target: AUC 0.730-0.760, Acc 68-72%")
        
        auc_improvement = avg_auc - 0.699
        acc_improvement = avg_acc - 0.6567
        
        logger.info(f"\n   📈 Improvements:")
        logger.info(f"      AUC: {auc_improvement:+.3f} ({auc_improvement/0.699*100:+.1f}%)")
        logger.info(f"      Accuracy: {acc_improvement:+.2%}")
        
        if avg_auc >= 0.730:
            logger.info("\n   🎉🎉🎉 V5.1 TARGET ACHIEVED! 🎉🎉🎉")
        elif avg_auc > 0.699:
            logger.info("\n   ✅ V5.1 improves on V4.1!")
        else:
            logger.info("\n   ⚠️ V5.1 needs more tuning")
        
        logger.info("\n   📊 Individual Results:")
        for r in all_results:
            logger.info(f"      {r['symbol']}: Acc={r['accuracy']*100:.1f}%, AUC={r['auc']:.3f}")
    
    await trainer.collector.exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
