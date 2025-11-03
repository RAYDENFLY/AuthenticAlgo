"""
Strategy 2: Pure ML (XGBoost with optimized model)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import xgboost as xgb
import json
import ta
from base_trader import BaseTrader


class MLTrader(BaseTrader):
    """Pure machine learning strategy"""
    
    def __init__(self, capital: float = 10.0):
        super().__init__("Pure_ML", capital)
        
        # Load best XGBoost model
        self.model = None
        self.model_params = None
        self.load_model()
    
    def load_model(self):
        """Load best optimized model"""
        try:
            model_dir = Path(__file__).parent.parent.parent / "ml" / "models"
            model_path = model_dir / "xgboost_optimized_BTCUSDT_1h_20251103_122755.json"
            
            if model_path.exists():
                self.model = xgb.Booster()
                self.model.load_model(str(model_path))
                
                # Load params
                param_path = model_path.with_name(model_path.stem + '_params.json')
                with open(param_path, 'r') as f:
                    self.model_params = json.load(f)
                
                print(f"✅ Loaded ML model: {self.model_params.get('training_accuracy', 'N/A')}% accuracy")
            else:
                print(f"⚠️ Model not found, using random signals")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 50+ features for ML prediction"""
        features_df = df.copy()
        
        if 'timestamp' in features_df.columns:
            features_df = features_df.set_index('timestamp')
        
        # Trend indicators
        for period in [5, 10, 14, 20, 50, 100, 200]:
            features_df[f'sma_{period}'] = ta.trend.sma_indicator(features_df['close'], window=period)
            features_df[f'ema_{period}'] = ta.trend.ema_indicator(features_df['close'], window=period)
        
        for period in [10, 20, 50]:
            features_df[f'price_to_sma{period}'] = (features_df['close'] - features_df[f'sma_{period}']) / features_df[f'sma_{period}']
        
        # RSI
        for period in [7, 14, 21, 28]:
            features_df[f'rsi_{period}'] = ta.momentum.rsi(features_df['close'], window=period)
        
        # MACD
        macd_indicator = ta.trend.MACD(features_df['close'])
        features_df['macd'] = macd_indicator.macd()
        features_df['macd_signal'] = macd_indicator.macd_signal()
        features_df['macd_hist'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        for period in [10, 20]:
            bb = ta.volatility.BollingerBands(features_df['close'], window=period, window_dev=2)
            features_df[f'bb_upper_{period}'] = bb.bollinger_hband()
            features_df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            features_df[f'bb_lower_{period}'] = bb.bollinger_lband()
            features_df[f'bb_width_{period}'] = bb.bollinger_wband()
            features_df[f'bb_position_{period}'] = bb.bollinger_pband()
        
        # ATR
        for period in [7, 14, 21]:
            features_df[f'atr_{period}'] = ta.volatility.average_true_range(features_df['high'], features_df['low'], features_df['close'], window=period)
            features_df[f'atr_pct_{period}'] = features_df[f'atr_{period}'] / features_df['close']
        
        # ADX
        for period in [14, 20]:
            features_df[f'adx_{period}'] = ta.trend.adx(features_df['high'], features_df['low'], features_df['close'], window=period)
        
        # Stochastic
        for period in [14, 21]:
            stoch = ta.momentum.StochasticOscillator(features_df['high'], features_df['low'], features_df['close'], window=period, smooth_window=3)
            features_df[f'stoch_k_{period}'] = stoch.stoch()
            features_df[f'stoch_d_{period}'] = stoch.stoch_signal()
        
        # Volume
        features_df['obv'] = ta.volume.on_balance_volume(features_df['close'], features_df['volume'])
        features_df['volume_sma_10'] = features_df['volume'].rolling(10).mean()
        features_df['volume_sma_20'] = features_df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
        features_df['volume_trend'] = features_df['volume_sma_10'] / features_df['volume_sma_20']
        
        # Momentum
        for period in [1, 3, 5, 10, 20]:
            features_df[f'return_{period}'] = features_df['close'].pct_change(period)
            features_df[f'momentum_{period}'] = features_df['close'] / features_df['close'].shift(period) - 1
        
        features_df['acceleration_5'] = features_df['return_5'] - features_df['return_1']
        features_df['acceleration_10'] = features_df['return_10'] - features_df['return_5']
        
        # Volatility
        for period in [5, 10, 20]:
            features_df[f'volatility_{period}'] = features_df['close'].pct_change().rolling(period).std()
        
        # Range
        features_df['hl_range'] = (features_df['high'] - features_df['low']) / features_df['close']
        features_df['hl_range_sma10'] = features_df['hl_range'].rolling(10).mean()
        
        # Support/Resistance
        features_df['high_5d'] = features_df['high'].rolling(5).max()
        features_df['low_5d'] = features_df['low'].rolling(5).min()
        features_df['high_20d'] = features_df['high'].rolling(20).max()
        features_df['low_20d'] = features_df['low'].rolling(20).min()
        features_df['dist_to_high_5d'] = (features_df['high_5d'] - features_df['close']) / features_df['close']
        features_df['dist_to_low_5d'] = (features_df['close'] - features_df['low_5d']) / features_df['close']
        
        # Trend consistency
        features_df['trend_consistency_5'] = (features_df['close'] > features_df['close'].shift(1)).rolling(5).sum() / 5
        features_df['trend_consistency_10'] = (features_df['close'] > features_df['close'].shift(1)).rolling(10).sum() / 10
        
        # Time features
        if isinstance(features_df.index, pd.DatetimeIndex):
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
        
        features_df = features_df.dropna()
        
        return features_df
    
    def predict_ml(self, df: pd.DataFrame) -> Dict:
        """Generate ML prediction"""
        try:
            if self.model is None:
                # Fallback: random but biased prediction
                prediction = np.random.uniform(0.4, 0.9)
                return {'prediction': prediction, 'confidence': prediction}
            
            # Generate features
            features_df = self.generate_features(df)
            
            if features_df.empty:
                return {'prediction': 0.5, 'confidence': 0.5}
            
            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            latest_features = features_df[feature_cols].iloc[-1:]
            
            # Predict
            dmatrix = xgb.DMatrix(latest_features.values, feature_names=feature_cols)
            prediction = self.model.predict(dmatrix)[0]
            
            return {
                'prediction': float(prediction),
                'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {'prediction': 0.5, 'confidence': 0.5}
    
    def score_symbol(self, df: pd.DataFrame) -> float:
        """Score symbol using ML"""
        try:
            ml_result = self.predict_ml(df)
            # Prefer symbols with high confidence predictions
            return ml_result['confidence']
        except:
            return 0.0
    
    def screen_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Screen symbols using ML"""
        best_symbol = None
        best_score = 0.0
        
        for symbol, df in data_dict.items():
            score = self.score_symbol(df)
            if score > best_score:
                best_score = score
                best_symbol = symbol
        
        return best_symbol if best_score > 0.6 else None
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate ML signal"""
        try:
            ml_result = self.predict_ml(df)
            prediction = ml_result['prediction']
            confidence = ml_result['confidence']
            
            # Threshold for trading
            if prediction >= 0.6:
                direction = 'long'
            elif prediction <= 0.4:
                direction = 'short'
            else:
                direction = 'hold'
            
            latest_price = df['close'].iloc[-1] if not df.empty else 0
            
            return {
                'direction': direction,
                'confidence': confidence,
                'price': latest_price,
                'ml_prediction': prediction
            }
            
        except Exception as e:
            return {'direction': 'hold', 'confidence': 0.0, 'price': 0}
    
    def calculate_leverage(self, signal: Dict) -> int:
        """Dynamic leverage based on ML confidence"""
        confidence = signal.get('confidence', 0.5)
        
        # Aggressive: 5x-125x based on confidence
        # confidence 0.6 → 20x
        # confidence 0.7 → 40x
        # confidence 0.8 → 70x
        # confidence 0.9+ → 100x+
        
        if confidence >= 0.95:
            leverage = 125
        elif confidence >= 0.9:
            leverage = 100
        elif confidence >= 0.85:
            leverage = 80
        elif confidence >= 0.8:
            leverage = 60
        elif confidence >= 0.75:
            leverage = 45
        elif confidence >= 0.7:
            leverage = 35
        elif confidence >= 0.65:
            leverage = 25
        else:
            leverage = 15
        
        return max(5, min(125, leverage))
