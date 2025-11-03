"""
Ensemble Strategy - Combines Multiple Strategies
Voting system & weighted approach for better accuracy
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from strategies.base_strategy import BaseStrategy
from core.logger import setup_logger
from loguru import logger

setup_logger()


@dataclass
class StrategySignal:
    """Individual strategy signal with confidence"""
    strategy_name: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    reason: str


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble Strategy combining multiple approaches:
    1. RSI + MACD (Technical)
    2. Bollinger Bands (Technical)
    3. XGBoost ML (if available)
    4. Random Forest ML (if available)
    
    Modes:
    - 'voting': Simple majority voting
    - 'weighted': Weighted by strategy performance
    - 'unanimous': All must agree
    - 'confidence': Based on confidence scores
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Ensemble_Strategy", config)
        
        # Ensemble configuration
        self.mode = config.get('ensemble_mode', 'weighted')  # voting, weighted, unanimous, confidence
        self.min_agreement = config.get('min_agreement', 0.5)  # For voting mode
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Strategy weights based on benchmark results
        self.weights = config.get('strategy_weights', {
            'rsi_macd': 0.35,        # Best avg return (+0.13%)
            'random_forest': 0.30,   # Best single result (+0.76%)
            'xgboost': 0.20,         # Best win rate (59%)
            'bollinger': 0.15        # Volatile but good on BNB
        })
        
        # Performance tracking for adaptive weights
        self.strategy_performance = {
            'rsi_macd': {'wins': 0, 'losses': 0},
            'random_forest': {'wins': 0, 'losses': 0},
            'xgboost': {'wins': 0, 'losses': 0},
            'bollinger': {'wins': 0, 'losses': 0}
        }
        
        # ML models (loaded if available)
        self.ml_available = False
        self.xgb_model = None
        self.rf_model = None
        self.scaler = None
        
        # Load ML models if enabled
        if config.get('use_ml', True):
            self._load_ml_models(config)
        
        logger.info(f"Ensemble Strategy initialized: mode={self.mode}, weights={self.weights}")
    
    def _load_ml_models(self, config: Dict[str, Any]):
        """Load pre-trained ML models"""
        try:
            import pickle
            import xgboost as xgb
            from pathlib import Path
            
            models_dir = Path("ml/models")
            
            # Get latest models for current symbol
            symbol = config.get('symbol', 'ETHUSDT')
            timeframe = config.get('timeframe', '1h')
            
            # Find latest XGBoost model
            xgb_files = list(models_dir.glob(f"xgboost_{symbol}_{timeframe}_*.json"))
            if xgb_files:
                latest_xgb = max(xgb_files, key=lambda p: p.stat().st_mtime)
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(latest_xgb)
                logger.info(f"✅ Loaded XGBoost model: {latest_xgb.name}")
            
            # Find latest Random Forest model
            rf_files = list(models_dir.glob(f"random_forest_{symbol}_{timeframe}_*.pkl"))
            if rf_files:
                latest_rf = max(rf_files, key=lambda p: p.stat().st_mtime)
                with open(latest_rf, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info(f"✅ Loaded Random Forest model: {latest_rf.name}")
            
            if self.xgb_model or self.rf_model:
                self.ml_available = True
                # Load scaler
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                logger.info("✅ ML models available for ensemble")
            
        except Exception as e:
            logger.warning(f"⚠️ ML models not available: {e}")
            self.ml_available = False
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for ensemble"""
        
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
        
        # ATR for stop-loss
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Additional features for ML
        if self.ml_available:
            df = self._calculate_ml_features(df)
        
        return df
    
    def _calculate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 30 features for ML models"""
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['high_low_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Moving averages
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Bollinger Band features
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR features
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = df['close'].pct_change(10) * 100
        
        return df
    
    def get_rsi_macd_signal(self, row: pd.Series) -> StrategySignal:
        """Get signal from RSI + MACD strategy"""
        signal = 'HOLD'
        confidence = 0.0
        reason = ""
        
        rsi = row['rsi']
        macd = row['macd']
        macd_signal = row['macd_signal']
        macd_hist = row['macd_hist']
        prev_macd_hist = row.get('prev_macd_hist', 0)
        
        # BUY signal: RSI oversold + MACD bullish crossover
        if rsi < 30 and macd_hist > 0 and prev_macd_hist <= 0:
            signal = 'BUY'
            confidence = min(1.0, (30 - rsi) / 30 + 0.3)  # Higher confidence when more oversold
            reason = f"RSI oversold ({rsi:.1f}) + MACD bullish crossover"
        
        # SELL signal: RSI overbought OR MACD bearish crossover
        elif rsi > 70 or (macd_hist < 0 and prev_macd_hist >= 0):
            signal = 'SELL'
            if rsi > 70:
                confidence = min(1.0, (rsi - 70) / 30 + 0.3)
                reason = f"RSI overbought ({rsi:.1f})"
            else:
                confidence = 0.6
                reason = "MACD bearish crossover"
        
        return StrategySignal('rsi_macd', signal, confidence, reason)
    
    def get_bollinger_signal(self, row: pd.Series) -> StrategySignal:
        """Get signal from Bollinger Bands strategy"""
        signal = 'HOLD'
        confidence = 0.0
        reason = ""
        
        close = row['close']
        bb_lower = row['bb_lower']
        bb_middle = row['bb_middle']
        bb_upper = row['bb_upper']
        volume_ratio = row.get('volume_ratio', 1.0)
        
        # BUY signal: Price at/below lower band + volume spike
        if close <= bb_lower and volume_ratio > 1.5:
            signal = 'BUY'
            distance = (bb_lower - close) / bb_lower
            confidence = min(1.0, distance * 10 + 0.5)
            reason = f"Price at lower BB ({close:.2f} <= {bb_lower:.2f}) + volume spike"
        
        # SELL signal: Price at/above middle or upper band
        elif close >= bb_middle:
            signal = 'SELL'
            if close >= bb_upper:
                confidence = 0.8
                reason = f"Price at upper BB ({close:.2f} >= {bb_upper:.2f})"
            else:
                confidence = 0.5
                reason = f"Price at middle BB ({close:.2f} >= {bb_middle:.2f})"
        
        return StrategySignal('bollinger', signal, confidence, reason)
    
    def get_ml_signal(self, row: pd.Series, model, model_name: str) -> Optional[StrategySignal]:
        """Get signal from ML model"""
        if not model or not self.ml_available:
            return None
        
        try:
            # Prepare features (30 indicators)
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
            
            # Check if all features exist
            missing = [col for col in feature_cols if col not in row.index]
            if missing:
                return None
            
            X = row[feature_cols].values.reshape(1, -1)
            
            # Scale features
            if self.scaler:
                # Fit scaler on first call (use recent data)
                if not hasattr(self.scaler, 'mean_'):
                    # Use dummy fit for now - in production, fit on training data
                    self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Get prediction
            prediction = model.predict(X_scaled)[0]
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.7  # Default confidence for models without proba
            
            signal = 'BUY' if prediction == 1 else 'SELL' if prediction == 0 else 'HOLD'
            reason = f"ML prediction: {prediction} (confidence: {confidence:.2f})"
            
            return StrategySignal(model_name, signal, confidence, reason)
        
        except Exception as e:
            logger.debug(f"ML signal error for {model_name}: {e}")
            return None
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate ensemble signal combining all strategies
        """
        if len(data) < 50:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data',
                'details': {}
            }
        
        # Calculate indicators
        df = self.calculate_indicators(data.copy())
        
        # Get latest row
        current = df.iloc[-1]
        
        # Store previous MACD hist for crossover detection
        if len(df) > 1:
            current['prev_macd_hist'] = df.iloc[-2]['macd_hist']
        else:
            current['prev_macd_hist'] = 0
        
        # Collect signals from all strategies
        signals: List[StrategySignal] = []
        
        # 1. RSI + MACD
        signals.append(self.get_rsi_macd_signal(current))
        
        # 2. Bollinger Bands
        signals.append(self.get_bollinger_signal(current))
        
        # 3. XGBoost ML
        if self.xgb_model:
            xgb_signal = self.get_ml_signal(current, self.xgb_model, 'xgboost')
            if xgb_signal:
                signals.append(xgb_signal)
        
        # 4. Random Forest ML
        if self.rf_model:
            rf_signal = self.get_ml_signal(current, self.rf_model, 'random_forest')
            if rf_signal:
                signals.append(rf_signal)
        
        # Combine signals based on mode
        final_signal = self._combine_signals(signals)
        
        return final_signal
    
    def _combine_signals(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Combine multiple signals into final decision"""
        
        if not signals:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No signals available',
                'details': {}
            }
        
        if self.mode == 'voting':
            return self._voting_mode(signals)
        elif self.mode == 'weighted':
            return self._weighted_mode(signals)
        elif self.mode == 'unanimous':
            return self._unanimous_mode(signals)
        elif self.mode == 'confidence':
            return self._confidence_mode(signals)
        else:
            return self._weighted_mode(signals)  # Default
    
    def _voting_mode(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Simple majority voting"""
        buy_votes = sum(1 for s in signals if s.signal == 'BUY')
        sell_votes = sum(1 for s in signals if s.signal == 'SELL')
        total_votes = len(signals)
        
        buy_ratio = buy_votes / total_votes
        sell_ratio = sell_votes / total_votes
        
        if buy_ratio >= self.min_agreement:
            signal = 'BUY'
            confidence = buy_ratio
            reason = f"{buy_votes}/{total_votes} strategies vote BUY"
        elif sell_ratio >= self.min_agreement:
            signal = 'SELL'
            confidence = sell_ratio
            reason = f"{sell_votes}/{total_votes} strategies vote SELL"
        else:
            signal = 'HOLD'
            confidence = 0.0
            reason = f"No consensus (BUY:{buy_votes}, SELL:{sell_votes})"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'details': {
                'mode': 'voting',
                'signals': [{'strategy': s.strategy_name, 'signal': s.signal, 'confidence': s.confidence} for s in signals]
            }
        }
    
    def _weighted_mode(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Weighted combination based on strategy performance"""
        buy_weight = 0.0
        sell_weight = 0.0
        
        for s in signals:
            weight = self.weights.get(s.strategy_name, 0.1)
            if s.signal == 'BUY':
                buy_weight += weight * s.confidence
            elif s.signal == 'SELL':
                sell_weight += weight * s.confidence
        
        total_weight = sum(self.weights.get(s.strategy_name, 0.1) for s in signals)
        
        # Normalize
        if total_weight > 0:
            buy_score = buy_weight / total_weight
            sell_score = sell_weight / total_weight
        else:
            buy_score = sell_score = 0.0
        
        # Decision threshold
        threshold = self.confidence_threshold
        
        if buy_score > threshold and buy_score > sell_score:
            signal = 'BUY'
            confidence = buy_score
            reason = f"Weighted score: BUY={buy_score:.2f} > SELL={sell_score:.2f}"
        elif sell_score > threshold and sell_score > buy_score:
            signal = 'SELL'
            confidence = sell_score
            reason = f"Weighted score: SELL={sell_score:.2f} > BUY={buy_score:.2f}"
        else:
            signal = 'HOLD'
            confidence = max(buy_score, sell_score)
            reason = f"Below threshold (BUY:{buy_score:.2f}, SELL:{sell_score:.2f})"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'details': {
                'mode': 'weighted',
                'buy_score': buy_score,
                'sell_score': sell_score,
                'threshold': threshold,
                'signals': [{'strategy': s.strategy_name, 'signal': s.signal, 'confidence': s.confidence, 'weight': self.weights.get(s.strategy_name, 0.1)} for s in signals]
            }
        }
    
    def _unanimous_mode(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """All strategies must agree"""
        buy_signals = [s for s in signals if s.signal == 'BUY']
        sell_signals = [s for s in signals if s.signal == 'SELL']
        
        if len(buy_signals) == len(signals):
            signal = 'BUY'
            confidence = np.mean([s.confidence for s in buy_signals])
            reason = "All strategies agree: BUY"
        elif len(sell_signals) == len(signals):
            signal = 'SELL'
            confidence = np.mean([s.confidence for s in sell_signals])
            reason = "All strategies agree: SELL"
        else:
            signal = 'HOLD'
            confidence = 0.0
            reason = f"No unanimous agreement (BUY:{len(buy_signals)}, SELL:{len(sell_signals)})"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'details': {
                'mode': 'unanimous',
                'signals': [{'strategy': s.strategy_name, 'signal': s.signal} for s in signals]
            }
        }
    
    def _confidence_mode(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Use signal with highest confidence"""
        if not signals:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No signals',
                'details': {}
            }
        
        # Find signal with highest confidence
        best_signal = max(signals, key=lambda s: s.confidence)
        
        if best_signal.confidence >= self.confidence_threshold:
            return {
                'signal': best_signal.signal,
                'confidence': best_signal.confidence,
                'reason': f"Best: {best_signal.strategy_name} - {best_signal.reason}",
                'details': {
                    'mode': 'confidence',
                    'best_strategy': best_signal.strategy_name,
                    'signals': [{'strategy': s.strategy_name, 'signal': s.signal, 'confidence': s.confidence} for s in signals]
                }
            }
        else:
            return {
                'signal': 'HOLD',
                'confidence': best_signal.confidence,
                'reason': f"Best confidence ({best_signal.confidence:.2f}) below threshold ({self.confidence_threshold})",
                'details': {
                    'mode': 'confidence',
                    'signals': [{'strategy': s.strategy_name, 'signal': s.signal, 'confidence': s.confidence} for s in signals]
                }
            }
    
    def update_performance(self, strategy_name: str, win: bool):
        """Update strategy performance for adaptive weights"""
        if strategy_name in self.strategy_performance:
            if win:
                self.strategy_performance[strategy_name]['wins'] += 1
            else:
                self.strategy_performance[strategy_name]['losses'] += 1
    
    def get_stop_loss(self, entry_price: float, data: pd.DataFrame) -> float:
        """Calculate stop loss using ATR"""
        df = self.calculate_indicators(data.copy())
        atr = df.iloc[-1]['atr']
        return entry_price - (atr * 2)  # 2 ATR stop loss
    
    def get_take_profit(self, entry_price: float, data: pd.DataFrame) -> float:
        """Calculate take profit using ATR"""
        df = self.calculate_indicators(data.copy())
        atr = df.iloc[-1]['atr']
        return entry_price + (atr * 3)  # 3 ATR take profit
    
    def should_enter(self, data: pd.DataFrame, position_type: str = "long") -> bool:
        """Check if should enter position"""
        signal_data = self.generate_signal(data)
        if position_type == "long":
            return signal_data['signal'] == 'BUY' and signal_data['confidence'] >= self.confidence_threshold
        else:
            return signal_data['signal'] == 'SELL' and signal_data['confidence'] >= self.confidence_threshold
    
    def should_exit(self, data: pd.DataFrame, position_type: str = "long", entry_price: float = 0) -> bool:
        """Check if should exit position"""
        signal_data = self.generate_signal(data)
        
        # Stop loss check
        if entry_price > 0:
            current_price = data.iloc[-1]['close']
            stop_loss = self.get_stop_loss(entry_price, data)
            if current_price < stop_loss:
                return True
        
        # Signal-based exit
        if position_type == "long":
            return signal_data['signal'] == 'SELL'
        else:
            return signal_data['signal'] == 'BUY'
