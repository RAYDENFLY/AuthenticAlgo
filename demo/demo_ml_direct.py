"""
Demo: Direct ML Trading Bot (No Complex Imports)
Loads XGBoost model directly and generates features manually
Bypasses MLPredictor/FeatureEngine to avoid import issues
"""

import asyncio
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
from execution.asterdex import AsterDEXFutures

logger = setup_logger()


class DirectMLBot:
    """Trading bot that loads XGBoost model directly"""
    
    def __init__(self, model_path: str, capital: float = 10.0, leverage: int = 10):
        self.model_path = Path(model_path)
        self.capital = capital
        self.leverage = leverage
        self.exchange = None
        self.model = None
        self.params = None
        
        # Risk management
        self.confidence_threshold = 0.6
        self.stop_loss_atr_mult = 2.0
        self.take_profit_atr_mult = 3.0
        
    def load_model(self):
        """Load XGBoost model and parameters"""
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            
            # Load parameters
            param_path = self.model_path.with_name(
                self.model_path.stem + '_params.json'
            )
            with open(param_path, 'r') as f:
                self.params = json.load(f)
            
            logger.info(f"✅ Model loaded: {self.params.get('training_accuracy', 'N/A')}% accuracy")
            logger.info(f"   Symbol: {self.params.get('symbol', 'N/A')}")
            logger.info(f"   Timeframe: {self.params.get('timeframe', 'N/A')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 50+ features EXACTLY matching optimize_ml.py
        Must match feature names expected by model
        """
        import ta
        
        # Make a copy and ensure datetime index
        features_df = df.copy()
        if 'timestamp' in features_df.columns:
            features_df = features_df.set_index('timestamp')
        
        # === Trend indicators - Moving Averages ===
        for period in [5, 10, 14, 20, 50, 100, 200]:
            features_df[f'sma_{period}'] = ta.trend.sma_indicator(features_df['close'], window=period)
            features_df[f'ema_{period}'] = ta.trend.ema_indicator(features_df['close'], window=period)
            
        # Price position relative to MAs
        for period in [10, 20, 50]:
            features_df[f'price_to_sma{period}'] = (features_df['close'] - features_df[f'sma_{period}']) / features_df[f'sma_{period}']
        
        # === Multiple RSI periods ===
        for period in [7, 14, 21, 28]:
            features_df[f'rsi_{period}'] = ta.momentum.rsi(features_df['close'], window=period)
        
        # === MACD variations ===
        macd_indicator = ta.trend.MACD(features_df['close'])
        features_df['macd'] = macd_indicator.macd()
        features_df['macd_signal'] = macd_indicator.macd_signal()
        features_df['macd_hist'] = macd_indicator.macd_diff()
        
        # === Bollinger Bands (multiple periods) ===
        for period in [10, 20]:
            bb = ta.volatility.BollingerBands(features_df['close'], window=period, window_dev=2)
            features_df[f'bb_upper_{period}'] = bb.bollinger_hband()
            features_df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            features_df[f'bb_lower_{period}'] = bb.bollinger_lband()
            features_df[f'bb_width_{period}'] = bb.bollinger_wband()
            features_df[f'bb_position_{period}'] = bb.bollinger_pband()
        
        # === ATR (volatility) ===
        for period in [7, 14, 21]:
            features_df[f'atr_{period}'] = ta.volatility.average_true_range(features_df['high'], features_df['low'], features_df['close'], window=period)
            features_df[f'atr_pct_{period}'] = features_df[f'atr_{period}'] / features_df['close']
        
        # === ADX (trend strength) ===
        for period in [14, 20]:
            features_df[f'adx_{period}'] = ta.trend.adx(features_df['high'], features_df['low'], features_df['close'], window=period)
        
        # === Stochastic ===
        for period in [14, 21]:
            stoch = ta.momentum.StochasticOscillator(features_df['high'], features_df['low'], features_df['close'], window=period, smooth_window=3)
            features_df[f'stoch_k_{period}'] = stoch.stoch()
            features_df[f'stoch_d_{period}'] = stoch.stoch_signal()
        
        # === Volume features ===
        features_df['obv'] = ta.volume.on_balance_volume(features_df['close'], features_df['volume'])
        features_df['volume_sma_10'] = features_df['volume'].rolling(10).mean()
        features_df['volume_sma_20'] = features_df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
        features_df['volume_trend'] = features_df['volume_sma_10'] / features_df['volume_sma_20']
        
        # === Price momentum ===
        for period in [1, 3, 5, 10, 20]:
            features_df[f'return_{period}'] = features_df['close'].pct_change(period)
            features_df[f'momentum_{period}'] = features_df['close'] / features_df['close'].shift(period) - 1
        
        # === Price acceleration ===
        features_df['acceleration_5'] = features_df['return_5'] - features_df['return_1']
        features_df['acceleration_10'] = features_df['return_10'] - features_df['return_5']
        
        # === Volatility ===
        for period in [5, 10, 20]:
            features_df[f'volatility_{period}'] = features_df['close'].pct_change().rolling(period).std()
        
        # === High/Low range ===
        features_df['hl_range'] = (features_df['high'] - features_df['low']) / features_df['close']
        features_df['hl_range_sma10'] = features_df['hl_range'].rolling(10).mean()
        
        # === Support/Resistance (simplified) ===
        features_df['high_5d'] = features_df['high'].rolling(5).max()
        features_df['low_5d'] = features_df['low'].rolling(5).min()
        features_df['high_20d'] = features_df['high'].rolling(20).max()
        features_df['low_20d'] = features_df['low'].rolling(20).min()
        features_df['dist_to_high_5d'] = (features_df['high_5d'] - features_df['close']) / features_df['close']
        features_df['dist_to_low_5d'] = (features_df['close'] - features_df['low_5d']) / features_df['close']
        
        # === Trend consistency ===
        features_df['trend_consistency_5'] = (features_df['close'] > features_df['close'].shift(1)).rolling(5).sum() / 5
        features_df['trend_consistency_10'] = (features_df['close'] > features_df['close'].shift(1)).rolling(10).sum() / 10
        
        # === Time features ===
        if isinstance(features_df.index, pd.DatetimeIndex):
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
        
        # Drop NaN rows
        features_df = features_df.dropna()
        
        return features_df
    
    def predict_signal(self, df: pd.DataFrame) -> dict:
        """Generate trading signal from recent data"""
        try:
            # Generate features
            features_df = self.generate_features(df)
            
            if len(features_df) == 0:
                logger.warning("No valid features generated")
                return {'signal': 'hold', 'confidence': 0.0}
            
            # Get latest features (exclude OHLCV columns)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            latest_features = features_df[feature_cols].iloc[-1:]
            
            # CRITICAL: Convert to DMatrix WITH feature names
            dmatrix = xgb.DMatrix(latest_features.values, feature_names=feature_cols)
            
            # Predict
            prediction = self.model.predict(dmatrix)[0]
            
            # Determine signal
            if prediction >= self.confidence_threshold:
                signal = 'long'
                confidence = float(prediction)
            elif prediction <= (1 - self.confidence_threshold):
                signal = 'short'
                confidence = float(1 - prediction)
            else:
                signal = 'hold'
                confidence = float(0.5)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': float(prediction)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'signal': 'hold', 'confidence': 0.0}
    
    async def run_demo(self, duration_minutes: int = 10):
        """Run demo for specified duration"""
        logger.info(f"🚀 Starting Direct ML Bot Demo ({duration_minutes} minutes)")
        logger.info(f"   Capital: ${self.capital}")
        logger.info(f"   Leverage: {self.leverage}x")
        logger.info(f"   Model: {self.model_path.name}")
        logger.info("=" * 80)
        
        # Load model
        if not self.load_model():
            return
        
        # Connect to exchange
        try:
            logger.info("Connecting to AsterDEX...")
            self.exchange = AsterDEXFutures()
            await self.exchange.connect()
            logger.info("✅ Connected to AsterDEX")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to AsterDEX: {e}")
            logger.info("Running in simulation mode (no real trades)")
        
        # Get symbol from params
        symbol = self.params.get('symbol', 'BTCUSDT')
        timeframe = self.params.get('timeframe', '1h')
        
        logger.info(f"\n📊 Fetching {symbol} {timeframe} data...")
        
        try:
            # Try to get real data
            if self.exchange:
                ticker = await self.exchange.fetch_ticker(symbol)
                logger.info(f"Current {symbol} price: ${ticker['last']:.2f}")
            
            # For demo, we'll simulate with dummy data
            # In production, fetch real data from exchange
            logger.info("\n⚠️ Using simulation data for demo")
            logger.info("In production, would fetch real market data")
            
            # Simulate data (in reality, get from exchange)
            dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
            dummy_df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(200).cumsum() + 50000,
                'high': np.random.randn(200).cumsum() + 50100,
                'low': np.random.randn(200).cumsum() + 49900,
                'close': np.random.randn(200).cumsum() + 50000,
                'volume': np.random.randint(100, 1000, 200)
            })
            
            # Generate signal
            logger.info("\n🤖 Generating ML prediction...")
            signal = self.predict_signal(dummy_df)
            
            logger.info(f"\n📡 SIGNAL:")
            logger.info(f"   Direction: {signal['signal'].upper()}")
            logger.info(f"   Confidence: {signal['confidence']*100:.2f}%")
            logger.info(f"   Raw prediction: {signal.get('prediction', 0):.4f}")
            
            # Analyze signal
            if signal['signal'] == 'long':
                logger.info(f"\n✅ LONG signal detected!")
                logger.info(f"   Would open LONG position with ${self.capital} at {self.leverage}x")
                logger.info(f"   Position size: ${self.capital * self.leverage}")
            elif signal['signal'] == 'short':
                logger.info(f"\n✅ SHORT signal detected!")
                logger.info(f"   Would open SHORT position with ${self.capital} at {self.leverage}x")
                logger.info(f"   Position size: ${self.capital * self.leverage}")
            else:
                logger.info(f"\n⏸️ No clear signal (confidence too low)")
                logger.info(f"   Waiting for better opportunity...")
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ Demo complete!")
            logger.info("\nNext steps:")
            logger.info("1. Verify signal generation works")
            logger.info("2. Connect to real AsterDEX")
            logger.info("3. Enable real trading")
            logger.info("4. Monitor first trades")
            
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            if self.exchange:
                await self.exchange.close()


async def main():
    """Run the demo"""
    # Path to best model
    model_path = Path(__file__).parent.parent / "ml" / "models" / "xgboost_optimized_BTCUSDT_1h_20251103_122755.json"
    
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Run optimize_ml.py first to generate models")
        return
    
    # Create bot
    bot = DirectMLBot(
        model_path=str(model_path),
        capital=10.0,
        leverage=10
    )
    
    # Run demo
    await bot.run_demo(duration_minutes=10)


if __name__ == "__main__":
    asyncio.run(main())
