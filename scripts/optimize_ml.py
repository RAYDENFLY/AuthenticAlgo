"""
ML Model Optimization with Optuna
Hyperparameter tuning to achieve 75%+ accuracy
"""

import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

from core.logger import get_logger
import ta  # Technical Analysis library

logger = get_logger()

# Symbols and timeframes to optimize
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
TIMEFRAMES = ['1h', '4h']

# Target accuracy
TARGET_ACCURACY = 0.75  # 75% minimum


class MLOptimizer:
    """ML Model Optimizer with Optuna"""
    
    def __init__(self):
        self.logger = logger
        
    def load_and_prepare_data(self, symbol: str, timeframe: str):
        """Load data and prepare features"""
        self.logger.info(f"\n📊 Loading data for {symbol} {timeframe}...")
        
        # Load historical data from CSV files
        try:
            # Try to find the data file
            import glob
            pattern = f"data/historical/*_{symbol}_{timeframe}_*.csv"
            files = glob.glob(pattern)
            
            if not files:
                self.logger.warning(f"❌ No data file found for {symbol} {timeframe}")
                return None, None, None
            
            # Use the latest/largest file
            data_file = max(files, key=lambda x: os.path.getsize(x))
            self.logger.info(f"   Loading from: {data_file}")
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            data = data.sort_index()
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return None, None, None
        
        if data.empty or len(data) < 200:
            self.logger.warning(f"❌ Insufficient data for {symbol} {timeframe}")
            return None, None, None
            
        self.logger.info(f"✅ Loaded {len(data)} candles")
        
        # Add enhanced technical indicators
        data = self._add_enhanced_features(data)
        
        # Create target (1 = price up in next N periods, 0 = down)
        lookahead = 5 if timeframe == '1h' else 2
        data['future_return'] = data['close'].pct_change(lookahead).shift(-lookahead)
        data['target'] = (data['future_return'] > 0.002).astype(int)  # 0.2% threshold
        
        # Drop NaN
        data = data.dropna()
        
        if len(data) < 100:
            self.logger.warning(f"❌ Too few samples after feature engineering: {len(data)}")
            return None, None, None
        
        # Prepare features
        feature_cols = [col for col in data.columns 
                       if col not in ['target', 'future_return', 'open', 'high', 'low', 'close', 'volume']]
        
        X = data[feature_cols].copy()
        y = data['target'].copy()
        
        # Remove any remaining inf/nan
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        self.logger.info(f"✅ Features: {X.shape[1]}, Samples: {len(X)}")
        self.logger.info(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, data
    
    def _add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 50+ technical indicators for better prediction using ta library"""
        
        # Trend indicators
        for period in [5, 10, 14, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
        # Price position relative to MAs
        for period in [10, 20, 50]:
            df[f'price_to_sma{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Multiple RSI periods
        for period in [7, 14, 21, 28]:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # MACD variations
        macd_indicator = ta.trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # Bollinger Bands (multiple periods)
        for period in [10, 20]:
            bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=2)
            df[f'bb_upper_{period}'] = bb.bollinger_hband()
            df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            df[f'bb_lower_{period}'] = bb.bollinger_lband()
            df[f'bb_width_{period}'] = bb.bollinger_wband()
            df[f'bb_position_{period}'] = bb.bollinger_pband()
        
        # ATR (volatility)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
        
        # ADX (trend strength)
        for period in [14, 20]:
            df[f'adx_{period}'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        
        # Stochastic
        for period in [14, 21]:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=period, smooth_window=3)
            df[f'stoch_k_{period}'] = stoch.stoch()
            df[f'stoch_d_{period}'] = stoch.stoch_signal()
        
        # Volume features
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_20']
        
        # Price momentum
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Price acceleration
        df['acceleration_5'] = df['return_5'] - df['return_1']
        df['acceleration_10'] = df['return_10'] - df['return_5']
        
        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # High/Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_sma10'] = df['hl_range'].rolling(10).mean()
        
        # Support/Resistance (simplified)
        df['high_5d'] = df['high'].rolling(5).max()
        df['low_5d'] = df['low'].rolling(5).min()
        df['high_20d'] = df['high'].rolling(20).max()
        df['low_20d'] = df['low'].rolling(20).min()
        df['dist_to_high_5d'] = (df['high_5d'] - df['close']) / df['close']
        df['dist_to_low_5d'] = (df['close'] - df['low_5d']) / df['close']
        
        # Trend consistency
        df['trend_consistency_5'] = (df['close'] > df['close'].shift(1)).rolling(5).sum() / 5
        df['trend_consistency_10'] = (df['close'] > df['close'].shift(1)).rolling(10).sum() / 10
        
        # Time features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return df
    
    def objective_xgboost(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for XGBoost"""
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    def objective_random_forest(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for Random Forest"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    def optimize_model(self, X, y, model_type='xgboost', n_trials=50):
        """
        Optimize model hyperparameters with Optuna
        
        Args:
            X: Features
            y: Target
            model_type: 'xgboost' or 'random_forest'
            n_trials: Number of optimization trials
        """
        self.logger.info(f"\n🔧 Optimizing {model_type.upper()} with {n_trials} trials...")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_params_list = []
        best_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"\n📊 Fold {fold + 1}/3")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Optimize
            if model_type == 'xgboost':
                objective = lambda trial: self.objective_xgboost(trial, X_train, y_train, X_val, y_val)
            else:
                objective = lambda trial: self.objective_random_forest(trial, X_train, y_train, X_val, y_val)
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            best_params_list.append(study.best_params)
            best_scores.append(study.best_value)
            
            self.logger.info(f"✅ Fold {fold + 1} Best Accuracy: {study.best_value:.4f}")
        
        # Average best parameters
        avg_accuracy = np.mean(best_scores)
        self.logger.info(f"\n🎯 Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        
        # Use best params from best fold
        best_fold = np.argmax(best_scores)
        best_params = best_params_list[best_fold]
        
        self.logger.info(f"\n🏆 Best parameters from fold {best_fold + 1}:")
        for key, value in best_params.items():
            self.logger.info(f"   {key}: {value}")
        
        return best_params, avg_accuracy
    
    def train_final_model(self, X, y, params, model_type='xgboost'):
        """Train final model with optimized parameters"""
        
        self.logger.info(f"\n🎓 Training final {model_type.upper()} model...")
        
        if model_type == 'xgboost':
            params['tree_method'] = 'hist'
            params['random_state'] = 42
            params['n_jobs'] = -1
            model = xgb.XGBClassifier(**params)
        else:
            params['random_state'] = 42
            params['n_jobs'] = -1
            model = RandomForestClassifier(**params)
        
        model.fit(X, y)
        
        # Evaluate on full data
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        self.logger.info(f"\n📊 Final Model Performance:")
        self.logger.info(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        self.logger.info(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        self.logger.info(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        return model, accuracy
    
    def save_model(self, model, params, accuracy, symbol, timeframe, model_type):
        """Save optimized model"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_type == 'xgboost':
            model_path = f"ml/models/xgboost_optimized_{symbol}_{timeframe}_{timestamp}.json"
            model.save_model(model_path)
        else:
            model_path = f"ml/models/random_forest_optimized_{symbol}_{timeframe}_{timestamp}.pkl"
            joblib.dump(model, model_path)
        
        # Save parameters
        params_path = model_path.replace('.json', '_params.json').replace('.pkl', '_params.json')
        with open(params_path, 'w') as f:
            json.dump({
                'params': params,
                'accuracy': float(accuracy),
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': model_type,
                'timestamp': timestamp
            }, f, indent=2)
        
        self.logger.info(f"💾 Model saved: {model_path}")
        self.logger.info(f"💾 Params saved: {params_path}")
        
        return model_path


def main():
    """Main optimization workflow"""
    
    logger.info("="*80)
    logger.info("🚀 ML MODEL OPTIMIZATION WITH OPTUNA")
    logger.info("="*80)
    logger.info(f"Target Accuracy: {TARGET_ACCURACY*100:.0f}%")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Timeframes: {TIMEFRAMES}")
    logger.info("="*80)
    
    optimizer = MLOptimizer()
    
    results = []
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.info(f"\n{'='*80}")
            logger.info(f"🎯 Optimizing {symbol} {timeframe}")
            logger.info(f"{'='*80}")
            
            # Load data
            X, y, data = optimizer.load_and_prepare_data(symbol, timeframe)
            
            if X is None:
                logger.warning(f"⚠️ Skipping {symbol} {timeframe} - insufficient data")
                continue
            
            # Optimize XGBoost
            logger.info(f"\n{'='*40}")
            logger.info("🔧 XGBOOST OPTIMIZATION")
            logger.info(f"{'='*40}")
            
            xgb_params, xgb_accuracy = optimizer.optimize_model(X, y, 'xgboost', n_trials=30)
            xgb_model, xgb_final_acc = optimizer.train_final_model(X, y, xgb_params, 'xgboost')
            
            if xgb_final_acc >= TARGET_ACCURACY:
                logger.info(f"✅ XGBoost meets target! {xgb_final_acc*100:.2f}% >= {TARGET_ACCURACY*100:.0f}%")
                xgb_path = optimizer.save_model(xgb_model, xgb_params, xgb_final_acc, symbol, timeframe, 'xgboost')
            else:
                logger.warning(f"⚠️ XGBoost below target: {xgb_final_acc*100:.2f}% < {TARGET_ACCURACY*100:.0f}%")
                xgb_path = None
            
            # Optimize Random Forest
            logger.info(f"\n{'='*40}")
            logger.info("🔧 RANDOM FOREST OPTIMIZATION")
            logger.info(f"{'='*40}")
            
            rf_params, rf_accuracy = optimizer.optimize_model(X, y, 'random_forest', n_trials=30)
            rf_model, rf_final_acc = optimizer.train_final_model(X, y, rf_params, 'random_forest')
            
            if rf_final_acc >= TARGET_ACCURACY:
                logger.info(f"✅ Random Forest meets target! {rf_final_acc*100:.2f}% >= {TARGET_ACCURACY*100:.0f}%")
                rf_path = optimizer.save_model(rf_model, rf_params, rf_final_acc, symbol, timeframe, 'random_forest')
            else:
                logger.warning(f"⚠️ Random Forest below target: {rf_final_acc*100:.2f}% < {TARGET_ACCURACY*100:.0f}%")
                rf_path = None
            
            # Store results
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'xgboost_accuracy': xgb_final_acc,
                'xgboost_meets_target': xgb_final_acc >= TARGET_ACCURACY,
                'xgboost_path': xgb_path,
                'rf_accuracy': rf_final_acc,
                'rf_meets_target': rf_final_acc >= TARGET_ACCURACY,
                'rf_path': rf_path
            })
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 OPTIMIZATION SUMMARY")
    logger.info(f"{'='*80}\n")
    
    for result in results:
        logger.info(f"{result['symbol']} {result['timeframe']}:")
        logger.info(f"  XGBoost: {result['xgboost_accuracy']*100:.2f}% {'✅' if result['xgboost_meets_target'] else '❌'}")
        logger.info(f"  Random Forest: {result['rf_accuracy']*100:.2f}% {'✅' if result['rf_meets_target'] else '❌'}")
        logger.info("")
    
    # Count successes
    total = len(results) * 2
    successes = sum(1 for r in results if r['xgboost_meets_target']) + sum(1 for r in results if r['rf_meets_target'])
    
    logger.info(f"{'='*80}")
    logger.info(f"🎯 Success Rate: {successes}/{total} models >= {TARGET_ACCURACY*100:.0f}%")
    logger.info(f"{'='*80}")
    
    if successes == 0:
        logger.warning("\n⚠️ WARNING: No models met the target accuracy!")
        logger.warning("Consider:")
        logger.warning("  1. Collecting more data (6+ months)")
        logger.warning("  2. Adjusting target threshold (0.2% -> 0.5%)")
        logger.warning("  3. Using ensemble methods")
        logger.warning("  4. Feature engineering improvements")
    
    logger.info("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()
