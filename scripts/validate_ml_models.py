"""
Backtest Validation for Optimized ML Models
Tests models on recent unseen data (Oct-Nov 2025)
"""

import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib
import xgboost as xgb

from core.logger import get_logger
import ta

logger = get_logger()

# Test on recent unseen data
TEST_START = "2025-10-15"  # Data after training period
TEST_END = "2025-11-03"    # Today


class ModelValidator:
    """Validate optimized models on unseen data"""
    
    def __init__(self):
        self.logger = logger
        self.results = []
        
    def load_model(self, model_path: str, params_path: str):
        """Load model and parameters"""
        
        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Load model
        if model_path.endswith('.json'):
            # XGBoost
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            model_type = 'xgboost'
        else:
            # Random Forest
            model = joblib.load(model_path)
            model_type = 'random_forest'
        
        return model, params, model_type
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add same features as training"""
        
        # Trend indicators
        for period in [5, 10, 14, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
        # Price position
        for period in [10, 20, 50]:
            df[f'price_to_sma{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # RSI
        for period in [7, 14, 21, 28]:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # MACD
        macd_indicator = ta.trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        for period in [10, 20]:
            bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=2)
            df[f'bb_upper_{period}'] = bb.bollinger_hband()
            df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            df[f'bb_lower_{period}'] = bb.bollinger_lband()
            df[f'bb_width_{period}'] = bb.bollinger_wband()
            df[f'bb_position_{period}'] = bb.bollinger_pband()
        
        # ATR
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
        
        # ADX
        for period in [14, 20]:
            df[f'adx_{period}'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        
        # Stochastic
        for period in [14, 21]:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=period, smooth_window=3)
            df[f'stoch_k_{period}'] = stoch.stoch()
            df[f'stoch_d_{period}'] = stoch.stoch_signal()
        
        # Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_20']
        
        # Momentum
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Acceleration
        df['acceleration_5'] = df['return_5'] - df['return_1']
        df['acceleration_10'] = df['return_10'] - df['return_5']
        
        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # Range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_sma10'] = df['hl_range'].rolling(10).mean()
        
        # Support/Resistance
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
    
    def validate_model(self, model_path: str, params_path: str):
        """Validate single model on unseen data"""
        
        # Load model
        model, params, model_type = self.load_model(model_path, params_path)
        symbol = params['symbol']
        timeframe = params['timeframe']
        training_accuracy = params['accuracy']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Validating {model_type.upper()} {symbol} {timeframe}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Training Accuracy: {training_accuracy*100:.2f}%")
        
        # Load recent data
        pattern = f"data/historical/*_{symbol}_{timeframe}_*.csv"
        files = glob.glob(pattern)
        
        if not files:
            self.logger.warning(f"❌ No data found for {symbol} {timeframe}")
            return None
        
        data_file = max(files, key=lambda x: os.path.getsize(x))
        data = pd.read_csv(data_file)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data = data.sort_index()
        
        # Filter to test period
        test_data = data[(data.index >= TEST_START) & (data.index <= TEST_END)].copy()
        
        if len(test_data) < 50:
            self.logger.warning(f"❌ Insufficient test data: {len(test_data)} rows")
            return None
        
        self.logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        self.logger.info(f"Test samples: {len(test_data)}")
        
        # Prepare features
        test_data = self.prepare_features(test_data)
        
        # Create target (actual price movement)
        lookahead = 5 if timeframe == '1h' else 2
        test_data['future_return'] = test_data['close'].pct_change(lookahead).shift(-lookahead)
        test_data['actual_direction'] = (test_data['future_return'] > 0.002).astype(int)
        
        test_data = test_data.dropna()
        
        if len(test_data) < 20:
            self.logger.warning(f"❌ Too few samples after prep: {len(test_data)}")
            return None
        
        # Prepare features for model
        feature_cols = [col for col in test_data.columns 
                       if col not in ['actual_direction', 'future_return', 'open', 'high', 'low', 'close', 'volume']]
        
        X_test = test_data[feature_cols].copy()
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.mean())
        
        y_test = test_data['actual_direction'].copy()
        
        # Predict
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        except Exception as e:
            self.logger.error(f"❌ Prediction error: {e}")
            return None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate backtest metrics
        test_data['prediction'] = y_pred
        test_data['confidence'] = y_proba
        
        # Simulate trades
        trades = 0
        wins = 0
        total_return = 0.0
        
        for i in range(len(test_data) - lookahead):
            if test_data['prediction'].iloc[i] == 1 and test_data['confidence'].iloc[i] > 0.6:
                trades += 1
                actual_ret = test_data['future_return'].iloc[i]
                if actual_ret > 0:
                    wins += 1
                total_return += actual_ret
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        avg_return = (total_return / trades * 100) if trades > 0 else 0
        
        # Results
        self.logger.info(f"\n📊 Validation Results:")
        self.logger.info(f"   Accuracy:     {accuracy*100:.2f}% (training: {training_accuracy*100:.2f}%)")
        self.logger.info(f"   Precision:    {precision*100:.2f}%")
        self.logger.info(f"   Recall:       {recall*100:.2f}%")
        self.logger.info(f"   F1 Score:     {f1*100:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"📈 Trading Simulation:")
        self.logger.info(f"   Total Trades: {trades}")
        self.logger.info(f"   Wins:         {wins}")
        self.logger.info(f"   Win Rate:     {win_rate:.2f}%")
        self.logger.info(f"   Avg Return:   {avg_return:.3f}%")
        self.logger.info(f"   Total Return: {total_return*100:.2f}%")
        
        # Verdict
        accuracy_diff = abs(accuracy - training_accuracy)
        
        if accuracy >= 0.70 and accuracy_diff < 0.15:
            verdict = "✅ PASSED - Ready for deployment"
            status = "PASS"
        elif accuracy >= 0.60:
            verdict = "⚠️ MARGINAL - Paper trade first"
            status = "MARGINAL"
        else:
            verdict = "❌ FAILED - Do not deploy"
            status = "FAIL"
        
        self.logger.info(f"\n{verdict}")
        self.logger.info(f"   Accuracy drop: {accuracy_diff*100:.2f}% {'✅' if accuracy_diff < 0.15 else '❌'}")
        
        # Store result
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'model_type': model_type,
            'training_accuracy': training_accuracy,
            'test_accuracy': accuracy,
            'accuracy_diff': accuracy_diff,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'trades': trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'status': status,
            'verdict': verdict,
            'model_path': model_path
        }
        
        self.results.append(result)
        return result


def main():
    """Main validation workflow"""
    
    logger.info("="*80)
    logger.info("🔍 ML MODEL VALIDATION ON UNSEEN DATA")
    logger.info("="*80)
    logger.info(f"Test Period: {TEST_START} to {TEST_END}")
    logger.info(f"Minimum Accuracy: 70% (live trading standard)")
    logger.info("="*80)
    
    validator = ModelValidator()
    
    # Find all optimized models
    model_files = glob.glob("ml/models/*optimized*.json") + glob.glob("ml/models/*optimized*.pkl")
    param_files = glob.glob("ml/models/*optimized*_params.json")
    
    # Match models with params
    model_pairs = []
    for model_file in model_files:
        if '_params' in model_file:
            continue
        
        # Find corresponding params file
        if model_file.endswith('.json'):
            params_file = model_file.replace('.json', '_params.json')
        else:
            params_file = model_file.replace('.pkl', '_params.json')
        
        if os.path.exists(params_file):
            model_pairs.append((model_file, params_file))
    
    logger.info(f"\nFound {len(model_pairs)} models to validate\n")
    
    # Validate each model
    for model_path, params_path in model_pairs:
        validator.validate_model(model_path, params_path)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 VALIDATION SUMMARY")
    logger.info(f"{'='*80}\n")
    
    if not validator.results:
        logger.warning("No models validated successfully")
        return
    
    # Sort by test accuracy
    results_sorted = sorted(validator.results, key=lambda x: x['test_accuracy'], reverse=True)
    
    # Print table
    logger.info(f"{'Symbol':<10} {'TF':<4} {'Model':<12} {'Train%':<8} {'Test%':<8} {'Diff%':<7} {'WinRate%':<10} {'Status':<10}")
    logger.info("-" * 80)
    
    for r in results_sorted:
        logger.info(
            f"{r['symbol']:<10} "
            f"{r['timeframe']:<4} "
            f"{r['model_type']:<12} "
            f"{r['training_accuracy']*100:>6.2f}% "
            f"{r['test_accuracy']*100:>6.2f}% "
            f"{r['accuracy_diff']*100:>5.2f}% "
            f"{r['win_rate']:>8.2f}% "
            f"{r['status']:<10}"
        )
    
    # Count by status
    passed = sum(1 for r in validator.results if r['status'] == 'PASS')
    marginal = sum(1 for r in validator.results if r['status'] == 'MARGINAL')
    failed = sum(1 for r in validator.results if r['status'] == 'FAIL')
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Results: {passed} PASSED ✅ | {marginal} MARGINAL ⚠️ | {failed} FAILED ❌")
    logger.info(f"{'='*80}\n")
    
    # Recommendations
    if passed > 0:
        logger.info("✅ READY FOR DEPLOYMENT:")
        for r in results_sorted:
            if r['status'] == 'PASS':
                logger.info(f"   • {r['model_type'].upper()} {r['symbol']} {r['timeframe']}: {r['test_accuracy']*100:.2f}% accuracy, {r['win_rate']:.1f}% win rate")
    
    if marginal > 0:
        logger.info("\n⚠️ PAPER TRADE FIRST:")
        for r in results_sorted:
            if r['status'] == 'MARGINAL':
                logger.info(f"   • {r['model_type'].upper()} {r['symbol']} {r['timeframe']}: {r['test_accuracy']*100:.2f}% accuracy")
    
    if failed > 0:
        logger.info("\n❌ DO NOT DEPLOY:")
        for r in results_sorted:
            if r['status'] == 'FAIL':
                logger.info(f"   • {r['model_type'].upper()} {r['symbol']} {r['timeframe']}: {r['test_accuracy']*100:.2f}% accuracy (too low)")
    
    logger.info("\n✅ Validation complete!")
    
    # Save results
    results_df = pd.DataFrame(validator.results)
    results_df.to_csv('ml/models/validation_results.csv', index=False)
    logger.info(f"📁 Results saved to: ml/models/validation_results.csv")


if __name__ == "__main__":
    main()
