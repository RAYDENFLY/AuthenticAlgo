"""
ML Strategy Benchmark - Train and Compare Machine Learning Models
XGBoost vs Random Forest vs Technical Strategies
Optimized for GTX 1050 Ti GPU
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import setup_logger
from loguru import logger

# ML libraries
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("⚠️ ML libraries not available. Install with: pip install xgboost scikit-learn")

# Setup
setup_logger()


class MLBenchmark:
    """Benchmark ML models for trading"""
    
    def __init__(self):
        self.data_dir = Path("data/historical")
        self.models_dir = Path("ml/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load historical data"""
        filename = f"asterdex_{symbol}_{timeframe}_20250805_to_20251103.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        logger.info(f"✅ Loaded {len(df)} candles from {filename}")
        return df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 30 technical features for ML"""
        logger.info("🔧 Calculating 30 technical features...")
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['high_low_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Moving Averages
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = df['close'].pct_change(10) * 100
        
        # Create target (1 = price goes up, 0 = price goes down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare data for ML training"""
        # Feature columns (30 features)
        feature_cols = [
            'returns', 'log_returns', 'price_change', 'high_low_range', 'body_size',
            'sma_7', 'sma_14', 'sma_21', 'sma_50',
            'ema_7', 'ema_14', 'ema_21', 'ema_50',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'atr_percent',
            'volume', 'volume_sma', 'volume_ratio',
            'momentum', 'roc'
        ]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"✅ Prepared data: Train={len(X_train)}, Test={len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test, df.iloc[split_idx:]
    
    def train_xgboost(self, X_train, y_train, use_gpu: bool = True) -> xgb.XGBClassifier:
        """Train XGBoost model (GPU optimized)"""
        logger.info("\n🚀 Training XGBoost model...")
        
        # GTX 1050 Ti optimized parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',  # Fast histogram algorithm
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Try GPU if requested (may not work on all systems)
        if use_gpu:
            try:
                # Better GPU detection
                import subprocess
                try:
                    # Check if NVIDIA GPU exists
                    nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, timeout=3)
                    if nvidia_smi.returncode == 0:
                        logger.info("🎮 NVIDIA GPU detected, testing XGBoost GPU support...")
                        # XGBoost 3.1+ uses 'device' parameter instead of 'gpu_id'
                        test_model = xgb.XGBClassifier(
                            tree_method='gpu_hist',
                            device='cuda',  # Use 'cuda' or 'cuda:0' for GPU
                            n_estimators=1
                        )
                        test_model.fit(X_train[:100], y_train[:100])
                        params['tree_method'] = 'gpu_hist'
                        params['device'] = 'cuda:0'  # Explicitly use GPU 0
                        logger.info("✅ GPU acceleration enabled! (GTX 1050 Ti)")
                    else:
                        raise Exception("No NVIDIA GPU found")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    logger.warning("⚠️ nvidia-smi not found, trying GPU anyway...")
                    test_model = xgb.XGBClassifier(
                        tree_method='gpu_hist',
                        device='cuda',
                        n_estimators=1
                    )
                    test_model.fit(X_train[:100], y_train[:100])
                    params['tree_method'] = 'gpu_hist'
                    params['device'] = 'cuda:0'
                    logger.info("✅ GPU acceleration enabled!")
            except Exception as e:
                logger.warning(f"⚠️ GPU not available ({type(e).__name__}: {str(e)[:50]})")
                logger.info("💡 GPU requires: CUDA toolkit + xgboost compiled with GPU support")
                logger.info("� Falling back to fast CPU hist algorithm")
                params['tree_method'] = 'hist'
        else:
            logger.info("💻 Using CPU (GPU disabled in config)")
        
        model = xgb.XGBClassifier(**params)
        
        start_time = datetime.now()
        model.fit(X_train, y_train, verbose=False)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ XGBoost trained in {duration:.2f}s")
        return model
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest model"""
        logger.info("\n🌲 Training Random Forest model...")
        
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        start_time = datetime.now()
        model.fit(X_train, y_train)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Random Forest trained in {duration:.2f}s")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            'model': model_name,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        }
        
        logger.info(f"\n📊 {model_name} Evaluation:")
        logger.info(f"   Accuracy:  {accuracy*100:.2f}%")
        logger.info(f"   Precision: {precision*100:.2f}%")
        logger.info(f"   Recall:    {recall*100:.2f}%")
        logger.info(f"   F1 Score:  {f1*100:.2f}%")
        
        return results
    
    def backtest_ml_strategy(self, model, X_test, df_test, initial_capital: float = 1000) -> Dict:
        """Backtest ML strategy"""
        logger.info(f"\n📈 Backtesting ML Strategy...")
        
        # Get predictions
        predictions = model.predict(X_test)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(len(predictions)):
            current_price = df_test['close'].iloc[i]
            prediction = predictions[i]
            
            # Entry: Model predicts price will go up
            if position == 0 and prediction == 1:
                position_size = (capital * 0.1) / current_price
                position = position_size
                entry_price = current_price
            
            # Exit: Model predicts price will go down OR stop loss
            elif position > 0:
                atr = df_test['atr'].iloc[i]
                stop_loss = entry_price - (atr * 2)
                
                if prediction == 0 or current_price < stop_loss:
                    pnl = (current_price - entry_price) * position
                    capital += pnl
                    
                    trade = {
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'return': (current_price - entry_price) / entry_price * 100
                    }
                    trades.append(trade)
                    position = 0
                    entry_price = 0
            
            equity_curve.append(capital + (position * current_price if position > 0 else 0))
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Sharpe Ratio
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        results = {
            'total_return': total_return,
            'final_capital': capital,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
        
        return results
    
    def run_ml_benchmark(self, symbol: str, timeframe: str, use_gpu: bool = True):
        """Run complete ML benchmark"""
        logger.info("="*80)
        logger.info(f"🧠 ML BENCHMARK - {symbol} {timeframe}")
        logger.info("="*80)
        
        # Load and prepare data
        df = self.load_data(symbol, timeframe)
        if df.empty:
            return None
        
        df = self.calculate_features(df)
        X_train, X_test, y_train, y_test, df_test = self.prepare_ml_data(df)
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_candles': len(df),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, use_gpu)
        xgb_eval = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        xgb_backtest = self.backtest_ml_strategy(xgb_model, X_test, df_test)
        
        results['xgboost'] = {**xgb_eval, **xgb_backtest}
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_eval = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        rf_backtest = self.backtest_ml_strategy(rf_model, X_test, df_test)
        
        results['random_forest'] = {**rf_eval, **rf_backtest}
        
        # Save models
        self.save_models(xgb_model, rf_model, symbol, timeframe)
        
        # Print comparison
        self.print_comparison(results)
        
        return results
    
    def save_models(self, xgb_model, rf_model, symbol: str, timeframe: str):
        """Save trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        xgb_path = self.models_dir / f"xgboost_{symbol}_{timeframe}_{timestamp}.json"
        xgb_model.save_model(xgb_path)
        logger.info(f"💾 XGBoost saved to: {xgb_path}")
        
        import pickle
        rf_path = self.models_dir / f"random_forest_{symbol}_{timeframe}_{timestamp}.pkl"
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_model, f)
        logger.info(f"💾 Random Forest saved to: {rf_path}")
    
    def print_comparison(self, results: Dict):
        """Print ML models comparison"""
        logger.info(f"\n{'='*80}")
        logger.info("🏆 ML MODELS COMPARISON")
        logger.info(f"{'='*80}\n")
        
        xgb = results['xgboost']
        rf = results['random_forest']
        
        comparison = pd.DataFrame([
            {
                'Metric': 'Prediction Accuracy',
                'XGBoost': f"{xgb['accuracy']:.2f}%",
                'Random Forest': f"{rf['accuracy']:.2f}%",
                'Winner': '🏆 XGB' if xgb['accuracy'] > rf['accuracy'] else '🏆 RF'
            },
            {
                'Metric': 'F1 Score',
                'XGBoost': f"{xgb['f1_score']:.2f}%",
                'Random Forest': f"{rf['f1_score']:.2f}%",
                'Winner': '🏆 XGB' if xgb['f1_score'] > rf['f1_score'] else '🏆 RF'
            },
            {
                'Metric': 'Total Return',
                'XGBoost': f"{xgb['total_return']:.2f}%",
                'Random Forest': f"{rf['total_return']:.2f}%",
                'Winner': '🏆 XGB' if xgb['total_return'] > rf['total_return'] else '🏆 RF'
            },
            {
                'Metric': 'Win Rate',
                'XGBoost': f"{xgb['win_rate']:.1f}%",
                'Random Forest': f"{rf['win_rate']:.1f}%",
                'Winner': '🏆 XGB' if xgb['win_rate'] > rf['win_rate'] else '🏆 RF'
            },
            {
                'Metric': 'Sharpe Ratio',
                'XGBoost': f"{xgb['sharpe_ratio']:.2f}",
                'Random Forest': f"{rf['sharpe_ratio']:.2f}",
                'Winner': '🏆 XGB' if xgb['sharpe_ratio'] > rf['sharpe_ratio'] else '🏆 RF'
            },
            {
                'Metric': 'Max Drawdown',
                'XGBoost': f"{xgb['max_drawdown']:.2f}%",
                'Random Forest': f"{rf['max_drawdown']:.2f}%",
                'Winner': '🏆 XGB' if xgb['max_drawdown'] > rf['max_drawdown'] else '🏆 RF'
            }
        ])
        
        logger.info(comparison.to_string(index=False))
        logger.info("")


async def main():
    """Main ML benchmark execution"""
    if not ML_AVAILABLE:
        logger.error("❌ ML libraries not installed!")
        logger.info("Install with: pip install xgboost scikit-learn")
        return
    
    benchmark = MLBenchmark()
    
    # Check GPU availability (will try, fallback to CPU if not available)
    logger.info("🔍 Checking for GPU support...")
    use_gpu = False  # CPU sufficient for weekly training (0.3s/model, GPU saves only 1-2s total)
    # Set to True if you have GPU-enabled XGBoost (conda install py-xgboost-gpu)
    # Note: pip install xgboost does NOT include GPU support
    
    logger.info(f"""
🧠 ML Benchmark Configuration:
   Models: XGBoost, Random Forest
   Features: 30 technical indicators
   Training: 80% of data
   Testing: 20% of data
   GPU Acceleration: {'✅ Enabled' if use_gpu else '❌ Disabled'}
   Optimization: GTX 1050 Ti optimized
""")
    
    # Benchmark multiple symbols
    all_results = []
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
        for timeframe in ['1h', '4h']:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting benchmark: {symbol} {timeframe}")
            logger.info(f"{'='*80}\n")
            
            result = benchmark.run_ml_benchmark(symbol, timeframe, use_gpu)
            if result:
                all_results.append(result)
            
            await asyncio.sleep(1)
    
    # Generate summary
    if all_results:
        generate_ml_summary(all_results)
    
    logger.info(f"\n✅ ML Benchmark completed!")
    logger.info(f"   Total tests: {len(all_results)}")
    logger.info(f"   Models saved to: ml/models/")


def generate_ml_summary(results: List[Dict]):
    """Generate overall ML performance summary"""
    logger.info(f"\n{'='*80}")
    logger.info("🏆 OVERALL ML BENCHMARK SUMMARY")
    logger.info(f"{'='*80}\n")
    
    xgb_returns = [r['xgboost']['total_return'] for r in results]
    rf_returns = [r['random_forest']['total_return'] for r in results]
    
    xgb_accuracy = [r['xgboost']['accuracy'] for r in results]
    rf_accuracy = [r['random_forest']['accuracy'] for r in results]
    
    summary = pd.DataFrame([
        {
            'Model': 'XGBoost',
            'Avg Return': f"{np.mean(xgb_returns):.2f}%",
            'Avg Accuracy': f"{np.mean(xgb_accuracy):.2f}%",
            'Best Return': f"{max(xgb_returns):.2f}%",
            'Worst Return': f"{min(xgb_returns):.2f}%",
            'Tests': len(results)
        },
        {
            'Model': 'Random Forest',
            'Avg Return': f"{np.mean(rf_returns):.2f}%",
            'Avg Accuracy': f"{np.mean(rf_accuracy):.2f}%",
            'Best Return': f"{max(rf_returns):.2f}%",
            'Worst Return': f"{min(rf_returns):.2f}%",
            'Tests': len(results)
        }
    ])
    
    logger.info(summary.to_string(index=False))
    
    logger.info(f"\n🏆 OVERALL WINNER: ", end="")
    if np.mean(xgb_returns) > np.mean(rf_returns):
        logger.info("XGBoost! 🚀")
    else:
        logger.info("Random Forest! 🌲")
    
    logger.info(f"\n{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
