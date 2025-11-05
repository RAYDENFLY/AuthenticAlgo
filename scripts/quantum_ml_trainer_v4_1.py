"""
QUANTUM LEAP V4.1 - BALANCED TRAINING
Fix class imbalance dengan SMOTE + proper validation

Key Fixes:
1. SMOTE untuk balance classes
2. Stratified split untuk maintain distribution
3. Class weights untuk model training
4. Proper evaluation metrics
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
from data.asterdex_collector import AsterDEXDataCollector
from execution.asterdex import AsterDEXFutures

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

logger = setup_logger()


class QuantumMLTrainerV41:
    """
    QUANTUM LEAP V4.1 - BALANCED TRAINING
    Fix class imbalance untuk AUC improvement
    """
    
    def __init__(self, config: Dict):
        self.collector = AsterDEXDataCollector(config)
        self.config = config
        
        logger.info("🚀 QUANTUM LEAP V4.1 - BALANCED TRAINING")
        logger.info("   Fix: SMOTE + Stratified Split + Class Weights")
        logger.info("   Target: AUC > 0.65, Accuracy 75-80%")
    
    async def get_data(self, symbol: str) -> pd.DataFrame:
        """Get and prepare data"""
        logger.info(f"📊 Collecting data for {symbol}...")
        
        klines = await self.collector.exchange.get_klines(symbol, '1h', 1000)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"   ✅ Collected {len(df)} candles")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features (V4.0 style but cleaner)"""
        logger.info("🔬 Feature engineering...")
        
        df = df.copy()
        
        # Core momentum
        df['momentum_5'] = df['close'].pct_change(5).fillna(0)
        df['momentum_10'] = df['close'].pct_change(10).fillna(0)
        df['momentum_20'] = df['close'].pct_change(20).fillna(0)
        
        # Volatility
        df['volatility_10'] = df['close'].pct_change().rolling(10).std().fillna(0)
        df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean().fillna(0)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean().fillna(0)
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        
        # Price features
        df['price_ma20'] = df['close'].rolling(20).mean().fillna(0)
        df['price_std20'] = df['close'].rolling(20).std().fillna(0)
        df['upper_band'] = df['price_ma20'] + 2 * df['price_std20']
        df['lower_band'] = df['price_ma20'] - 2 * df['price_std20']
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band']).replace(0, 1)
        
        # Momentum acceleration
        df['momentum_accel'] = df['momentum_5'] - df['momentum_10']
        df['rsi_momentum'] = df['rsi_14'].diff(3).fillna(0)
        
        logger.info(f"   ✅ Created 15 features")
        return df
    
    def create_balanced_target(self, df: pd.DataFrame, threshold_percentile: int = 60) -> pd.DataFrame:
        """
        Create BALANCED binary target
        
        V4.1 Fix: Use percentile-based thresholds untuk ensure balance
        """
        logger.info("🎯 Creating balanced target...")
        
        df = df.copy()
        
        # Future return
        df['future_return'] = df['close'].pct_change(6).shift(-6).fillna(0)
        
        # Dynamic threshold based on percentile
        positive_threshold = df['future_return'].quantile(threshold_percentile / 100)
        
        logger.info(f"   Threshold (p{threshold_percentile}): {positive_threshold:.4f}")
        
        # Binary target: 1 if return > threshold
        df['target'] = (df['future_return'] > positive_threshold).astype(int)
        
        # Check distribution
        class_dist = df['target'].value_counts(normalize=True)
        logger.info(f"   Class 0: {class_dist.get(0, 0)*100:.1f}%")
        logger.info(f"   Class 1: {class_dist.get(1, 0)*100:.1f}%")
        
        return df
    
    def apply_smote_balancing(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE + RandomUnderSampler for balanced data
        
        V4.1 KEY FIX: This should significantly improve AUC
        """
        logger.info("⚖️ Applying SMOTE balancing...")
        
        # Check original distribution
        original_dist = y.value_counts()
        logger.info(f"   Original: Class 0={original_dist.get(0, 0)}, Class 1={original_dist.get(1, 0)}")
        
        # SMOTE + UnderSampling pipeline
        over = SMOTE(sampling_strategy=0.8, random_state=42)  # Oversample minority to 80% of majority
        under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # Balance to 50/50
        
        pipeline = ImbPipeline([
            ('over', over),
            ('under', under)
        ])
        
        try:
            X_balanced, y_balanced = pipeline.fit_resample(X, y)
            
            # Check new distribution
            new_dist = pd.Series(y_balanced).value_counts()
            logger.info(f"   Balanced: Class 0={new_dist.get(0, 0)}, Class 1={new_dist.get(1, 0)}")
            logger.info(f"   ✅ SMOTE applied successfully")
            
            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
        
        except Exception as e:
            logger.warning(f"   ⚠️ SMOTE failed: {e}")
            logger.info("   Using original data")
            return X, y
    
    def train_with_class_weights(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Train models with proper class weights
        
        V4.1: Use scale_pos_weight untuk handle any remaining imbalance
        """
        logger.info("🏋️ Training with class weights...")
        
        # Calculate class weights
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts.index else 1.0
        
        logger.info(f"   Scale pos weight: {scale_pos_weight:.2f}")
        
        models = {}
        results = {}
        
        # XGBoost with class weights
        logger.info("   Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )
        xgb.fit(X_train, y_train)
        
        xgb_pred = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_auc = roc_auc_score(y_test, xgb_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        models['xgb'] = xgb
        results['xgb'] = {'accuracy': xgb_acc, 'auc': xgb_auc}
        
        logger.info(f"      XGB - Acc: {xgb_acc*100:.1f}%, AUC: {xgb_auc:.3f}")
        
        # LightGBM with class weights
        logger.info("   Training LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        
        lgbm_pred = lgbm.predict(X_test)
        lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
        
        lgbm_acc = accuracy_score(y_test, lgbm_pred)
        lgbm_auc = roc_auc_score(y_test, lgbm_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        models['lgbm'] = lgbm
        results['lgbm'] = {'accuracy': lgbm_acc, 'auc': lgbm_auc}
        
        logger.info(f"      LGBM - Acc: {lgbm_acc*100:.1f}%, AUC: {lgbm_auc:.3f}")
        
        # Ensemble (average probabilities)
        logger.info("   Creating ensemble...")
        ensemble_proba = (xgb_proba + lgbm_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        results['ensemble'] = {'accuracy': ensemble_acc, 'auc': ensemble_auc}
        
        logger.info(f"      Ensemble - Acc: {ensemble_acc*100:.1f}%, AUC: {ensemble_auc:.3f}")
        
        # Classification report
        logger.info("\n   📊 Classification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=['SELL', 'BUY']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        logger.info(f"   Confusion Matrix:")
        logger.info(f"      TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"      FN={cm[1,0]}, TP={cm[1,1]}")
        
        return {
            'models': models,
            'results': results,
            'best_accuracy': ensemble_acc,
            'best_auc': ensemble_auc,
            'confusion_matrix': cm
        }
    
    async def train_v41(self, symbol: str) -> Dict:
        """
        Complete V4.1 training pipeline
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 QUANTUM V4.1 TRAINING: {symbol}")
        logger.info(f"{'='*70}")
        
        # 1. Get data
        df = await self.get_data(symbol)
        
        # 2. Create features
        df = self.create_features(df)
        
        # 3. Create BALANCED target
        df = self.create_balanced_target(df, threshold_percentile=60)
        
        # 4. Prepare X, y
        feature_cols = [
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_20', 'atr_14',
            'rsi_14', 'volume_ratio', 'bb_position',
            'momentum_accel', 'rsi_momentum'
        ]
        
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['target']
        
        # Remove NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"   Valid samples: {len(X)}")
        
        if len(X) < 200:
            logger.error("   ❌ Insufficient data")
            return None
        
        # 5. Stratified split (maintain class distribution)
        logger.info("   Splitting data (stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y,  # V4.1 KEY: Stratified split
            random_state=42
        )
        
        logger.info(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 6. Apply SMOTE balancing to training data only
        X_train_balanced, y_train_balanced = self.apply_smote_balancing(X_train, y_train)
        
        # 7. Train with class weights
        results = self.train_with_class_weights(
            X_train_balanced, y_train_balanced,
            X_test, y_test
        )
        
        results['symbol'] = symbol
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ {symbol} COMPLETE")
        logger.info(f"   Best Accuracy: {results['best_accuracy']*100:.2f}%")
        logger.info(f"   Best AUC: {results['best_auc']:.3f}")
        logger.info(f"{'='*70}")
        
        return results


async def main():
    """Main training loop"""
    
    config = {
        'exchange': 'asterdex',
        'api_key': 'dummy',
        'api_secret': 'dummy',
        'testnet': True
    }
    
    trainer = QuantumMLTrainerV41(config)
    
    coins = ['BTCUSDT', 'ETHUSDT', 'TRUMPUSDT']
    all_results = []
    
    for symbol in coins:
        try:
            result = await trainer.train_v41(symbol)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if all_results:
        avg_acc = np.mean([r['best_accuracy'] for r in all_results])
        
        # Fix AUC calculation - filter nan
        valid_aucs = [r['best_auc'] for r in all_results if not np.isnan(r['best_auc'])]
        avg_auc = np.mean(valid_aucs) if valid_aucs else 0.5
        
        logger.info("\n" + "="*70)
        logger.info("🎯 QUANTUM V4.1 FINAL RESULTS")
        logger.info("="*70)
        logger.info(f"   Average Accuracy: {avg_acc*100:.2f}%")
        logger.info(f"   Average AUC: {avg_auc:.3f}")
        logger.info(f"   Target: AUC > 0.65")
        
        if avg_auc > 0.65:
            logger.info("\n   🎉🎉🎉 V4.1 FIX SUCCESSFUL! 🎉🎉🎉")
        elif avg_auc > 0.60:
            logger.info("\n   ✅ Improvement achieved, but more work needed")
        else:
            logger.warning("\n   ⚠️ AUC still low, deeper investigation required")
        
        logger.info("\n   📊 Individual Results:")
        for r in all_results:
            logger.info(f"      {r['symbol']}: Acc={r['best_accuracy']*100:.1f}%, AUC={r['best_auc']:.3f}")
    
    await trainer.collector.exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
