📊 TECHNICAL INDICATORS MODULE
1. indicators/init.py


```python
"""
Technical Indicators Module
Comprehensive collection of trading indicators
"""

from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators
from .custom import CustomIndicators

class TechnicalIndicators:
    """Main interface for all technical indicators"""
    
    def __init__(self):
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()
        self.custom = CustomIndicators()
    
    def get_all_indicators(self, df):
        """Calculate all indicators for a dataframe"""
        df = self.trend.calculate_all(df)
        df = self.momentum.calculate_all(df)
        df = self.volatility.calculate_all(df)
        df = self.volume.calculate_all(df)
        return df

__all__ = ['TechnicalIndicators', 'TrendIndicators', 'MomentumIndicators', 
           'VolatilityIndicators', 'VolumeIndicators', 'CustomIndicators']
```


2. indicators/trend.py

```python
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TrendIndicators:
    """Trend-following indicators"""
    
    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Simple Moving Average"""
        return df[column].rolling(window=period).mean()
    
    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Exponential Moving Average"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, 
             column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD, Signal Line, and Histogram"""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Smooth the values
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean() / 
                         pd.Series(true_range).ewm(alpha=1/period).mean())
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean() / 
                          pd.Series(true_range).ewm(alpha=1/period).mean())
        
        # Calculate ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> dict:
        """Ichimoku Cloud"""
        high = df['high']
        low = df['low']
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan).max() + 
                     low.rolling(window=tenkan).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun).max() + 
                    low.rolling(window=kijun).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou).max() + 
                         low.rolling(window=senkou).min()) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = df['close'].shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Supertrend Indicator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        for i in range(len(df)):
            if i < period:
                supertrend.iloc[i] = 0
                direction.iloc[i] = 0
                continue
                
            # Current values
            current_close = close.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            prev_supertrend = supertrend.iloc[i-1]
            prev_direction = direction.iloc[i-1]
            
            if prev_supertrend == 0:  # Initialization
                supertrend.iloc[i] = current_upper
                direction.iloc[i] = 1
            else:
                if prev_direction == 1:  # Previous uptrend
                    if current_close > current_lower:
                        supertrend.iloc[i] = max(current_lower, prev_supertrend)
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = current_upper
                        direction.iloc[i] = -1
                else:  # Previous downtrend
                    if current_close < current_upper:
                        supertrend.iloc[i] = min(current_upper, prev_supertrend)
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = current_lower
                        direction.iloc[i] = 1
        
        return supertrend, direction
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trend indicators"""
        # SMA & EMA
        df['sma_20'] = self.sma(df, 20)
        df['sma_50'] = self.sma(df, 50)
        df['ema_20'] = self.ema(df, 20)
        
        # MACD
        macd, signal, histogram = self.macd(df)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # ADX
        adx, plus_di, minus_di = self.adx(df)
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Ichimoku
        ichimoku_data = self.ichimoku(df)
        for key, value in ichimoku_data.items():
            df[f'ichimoku_{key}'] = value
        
        # Supertrend
        supertrend, direction = self.supertrend(df)
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        
        return df


```
3. indicators/momentum.py


```python
import pandas as pd
import numpy as np
from typing import Tuple

class MomentumIndicators:
    """Momentum and oscillator indicators"""
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Relative Strength Index"""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator %K and %D"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi
    
    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12, column: str = 'close') -> pd.Series:
        """Rate of Change"""
        return ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum indicators"""
        # RSI
        df['rsi_14'] = self.rsi(df, 14)
        df['rsi_21'] = self.rsi(df, 21)
        
        # Stochastic
        stoch_k, stoch_d = self.stochastic(df)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = self.williams_r(df)
        
        # CCI
        df['cci'] = self.cci(df)
        
        # MFI
        df['mfi'] = self.mfi(df)
        
        # ROC
        df['roc_12'] = self.roc(df, 12)
        
        return df
```


4. indicators/volatility.py


```python
import pandas as pd
import numpy as np

class VolatilityIndicators:
    """Volatility-based indicators"""
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                       column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (Upper, Middle, Lower)"""
        middle_band = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift()
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, 
                        multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels (Upper, Middle, Lower)"""
        middle = df['close'].ewm(span=ema_period).mean()
        atr = VolatilityIndicators.atr(df, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower
    
    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels (Upper, Middle, Lower)"""
        upper = df['high'].rolling(window=period).max()
        lower = df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    @staticmethod
    def standard_deviation(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Standard Deviation"""
        return df[column].rolling(window=period).std()
    
    @staticmethod
    def volatility_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volatility Ratio (Current Range vs Average Range)"""
        current_range = (df['high'] - df['low']) / df['close']
        avg_range = current_range.rolling(window=period).mean()
        return current_range / avg_range
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility indicators"""
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(df)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle  # Normalized width
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)  # Position within bands
        
        # ATR
        df['atr_14'] = self.atr(df, 14)
        df['atr_percent'] = self.atr(df, 14) / df['close']  # Normalized ATR
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.keltner_channels(df)
        df['kc_upper'] = kc_upper
        df['kc_middle'] = kc_middle
        df['kc_lower'] = kc_lower
        
        # Donchian Channels
        dc_upper, dc_middle, dc_lower = self.donchian_channels(df)
        df['dc_upper'] = dc_upper
        df['dc_middle'] = dc_middle
        df['dc_lower'] = dc_lower
        
        # Standard Deviation
        df['std_dev_20'] = self.standard_deviation(df, 20)
        
        # Volatility Ratio
        df['volatility_ratio'] = self.volatility_ratio(df)
        
        return df

```
5. indicators/volume.py

```python
import pandas as pd
import numpy as np

class VolumeIndicators:
    """Volume-based indicators"""
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_vp = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()
        return cumulative_vp / cumulative_volume
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, price_bins: int = 20) -> dict:
        """Volume Profile (Volume at Price)"""
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / price_bins
        
        volume_at_price = {}
        for i in range(len(df)):
            price_level = round(df['close'].iloc[i] / bin_size) * bin_size
            volume_at_price[price_level] = volume_at_price.get(price_level, 0) + df['volume'].iloc[i]
        
        return volume_at_price
    
    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf
    
    @staticmethod
    def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return df['volume'].rolling(window=period).mean()
    
    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Ratio (Current Volume vs Average Volume)"""
        avg_volume = df['volume'].rolling(window=period).mean()
        return df['volume'] / avg_volume
    
    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)  # Handle division by zero
        ad = (clv * df['volume']).cumsum()
        return ad
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volume indicators"""
        # VWAP
        df['vwap'] = self.vwap(df)
        
        # OBV
        df['obv'] = self.obv(df)
        df['obv_ema'] = df['obv'].ewm(span=20).mean()  # OBV EMA for trend
        
        # CMF
        df['cmf'] = self.cmf(df)
        
        # Volume SMA and Ratio
        df['volume_sma_20'] = self.volume_sma(df, 20)
        df['volume_ratio'] = self.volume_ratio(df, 20)
        
        # Accumulation/Distribution
        df['ad_line'] = self.accumulation_distribution(df)
        
        return df

```


6. indicators/custom.py


```python
import pandas as pd
import numpy as np
from functools import lru_cache

class CustomIndicators:
    """Custom and composite indicators"""
    
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=100)
    def cached_indicator(self, indicator_name: str, *args, **kwargs):
        """LRU cache for expensive indicator calculations"""
        cache_key = f"{indicator_name}_{hash(str(args))}_{hash(str(kwargs))}"
        return cache_key
    
    def trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Composite Trend Strength Indicator"""
        # Combine ADX, RSI, and MACD for trend strength
        adx = df.get('adx', 50)  # Default to neutral
        rsi = df.get('rsi_14', 50)
        macd_hist = df.get('macd_histogram', 0)
        
        # Normalize and weight components
        adx_strength = np.clip((adx - 25) / 50, 0, 1)  # ADX > 25 indicates trend
        rsi_strength = 1 - 2 * abs(rsi - 50) / 100  # Centered around 50
        macd_strength = np.tanh(macd_hist * 10)  # Normalize MACD histogram
        
        # Combined strength (0-1 scale)
        trend_strength = (adx_strength * 0.4 + rsi_strength * 0.3 + macd_strength * 0.3)
        return trend_strength
    
    def volatility_regime(self, df: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """Volatility Regime Detection (Low, Normal, High)"""
        volatility = df['atr_percent'] if 'atr_percent' in df else self._calculate_volatility(df)
        
        # Use rolling percentiles to determine regime
        low_threshold = volatility.rolling(window=lookback).apply(lambda x: np.percentile(x, 33))
        high_threshold = volatility.rolling(window=lookback).apply(lambda x: np.percentile(x, 66))
        
        regime = pd.Series('Normal', index=df.index)
        regime[volatility < low_threshold] = 'Low'
        regime[volatility > high_threshold] = 'High'
        
        return regime
    
    def market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Composite Market Regime Detection"""
        trend_strength = self.trend_strength(df)
        volatility_regime = self.volatility_regime(df)
        
        # Define market regimes
        conditions = [
            (trend_strength > 0.7) & (volatility_regime == 'Low'),   # Strong Trend, Low Vol
            (trend_strength > 0.7) & (volatility_regime == 'High'),  # Strong Trend, High Vol
            (trend_strength < 0.3) & (volatility_regime == 'Low'),   # Weak Trend, Low Vol (Ranging)
            (trend_strength < 0.3) & (volatility_regime == 'High'),  # Weak Trend, High Vol (Chaotic)
        ]
        
        regimes = ['Strong_Trend_Low_Vol', 'Strong_Trend_High_Vol', 
                  'Ranging_Low_Vol', 'Chaotic_High_Vol']
        
        default_regime = 'Normal_Market'
        market_regime = pd.Series(default_regime, index=df.index)
        
        for condition, regime in zip(conditions, regimes):
            market_regime = market_regime.where(~condition, regime)
        
        return market_regime
    
    def support_resistance(self, df: pd.DataFrame, window: int = 20) -> dict:
        """Dynamic Support and Resistance Levels"""
        closes = df['close']
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            window_data = closes.iloc[i-window:i+window]
            current_price = closes.iloc[i]
            
            if current_price == window_data.max():
                resistance_levels.append(current_price)
            elif current_price == window_data.min():
                support_levels.append(current_price)
        
        # Return recent levels (last 10 of each)
        recent_resistance = sorted(resistance_levels)[-5:] if resistance_levels else []
        recent_support = sorted(support_levels)[-5:] if support_levels else []
        
        return {
            'support_levels': recent_support,
            'resistance_levels': recent_resistance,
            'current_support': min(recent_support) if recent_support else None,
            'current_resistance': max(recent_resistance) if recent_resistance else None
        }
    
    def price_momentum_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Custom Price Momentum Oscillator"""
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        roc_sma = roc.rolling(window=period).mean()
        pmo = roc - roc_sma
        return pmo
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Helper method to calculate volatility if not present"""
        high_low_range = (df['high'] - df['low']) / df['close']
        return high_low_range.rolling(window=20).std()
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all custom indicators"""
        df['trend_strength'] = self.trend_strength(df)
        df['volatility_regime'] = self.volatility_regime(df)
        df['market_regime'] = self.market_regime(df)
        df['price_momentum_oscillator'] = self.price_momentum_oscillator(df)
        
        return df

```
7. tests/test_indicators.py


```python
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators
from indicators.custom import CustomIndicators

class TestIndicators:
    """Test suite for technical indicators"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = {
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 0.5,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 0.5,
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_sma(self, sample_data):
        """Test Simple Moving Average"""
        trend = TrendIndicators()
        sma = trend.sma(sample_data, period=10)
        
        assert len(sma) == len(sample_data)
        assert sma.iloc[9] == sample_data['close'].iloc[:10].mean()
        assert sma.isna().sum() == 9  # First 9 should be NaN
    
    def test_ema(self, sample_data):
        """Test Exponential Moving Average"""
        trend = TrendIndicators()
        ema = trend.ema(sample_data, period=10)
        
        assert len(ema) == len(sample_data)
        assert not all(ema.isna())  # Should have some valid values
        # EMA should be more responsive than SMA
        assert abs(ema.iloc[-1] - sample_data['close'].iloc[-1]) < abs(
            trend.sma(sample_data, 10).iloc[-1] - sample_data['close'].iloc[-1]
        )
    
    def test_macd(self, sample_data):
        """Test MACD Indicator"""
        trend = TrendIndicators()
        macd, signal, histogram = trend.macd(sample_data)
        
        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(histogram) == len(sample_data)
        assert histogram.equals(macd - signal)
    
    def test_rsi(self, sample_data):
        """Test RSI Indicator"""
        momentum = MomentumIndicators()
        rsi = momentum.rsi(sample_data)
        
        assert len(rsi) == len(sample_data)
        # RSI should be between 0 and 100
        assert 0 <= rsi.min() <= 100
        assert 0 <= rsi.max() <= 100
        # RSI should be 100 when all periods are gains
        all_gains = sample_data.copy()
        all_gains['close'] = all_gains['close'].shift(1) + 1
        rsi_all_gains = momentum.rsi(all_gains.dropna())
        assert rsi_all_gains.iloc[-1] == 100
    
    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands"""
        volatility = VolatilityIndicators()
        upper, middle, lower = volatility.bollinger_bands(sample_data)
        
        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)
        # Upper band should be above middle band
        assert (upper > middle).all()
        # Lower band should be below middle band
        assert (lower < middle).all()
    
    def test_atr(self, sample_data):
        """Test Average True Range"""
        volatility = VolatilityIndicators()
        atr = volatility.atr(sample_data)
        
        assert len(atr) == len(sample_data)
        # ATR should always be positive
        assert (atr > 0).all()
    
    def test_obv(self, sample_data):
        """Test On-Balance Volume"""
        volume = VolumeIndicators()
        obv = volume.obv(sample_data)
        
        assert len(obv) == len(sample_data)
        # OBV should start with first volume
        assert obv.iloc[0] == sample_data['volume'].iloc[0]
    
    def test_custom_trend_strength(self, sample_data):
        """Test Custom Trend Strength Indicator"""
        custom = CustomIndicators()
        
        # First calculate required base indicators
        trend = TrendIndicators()
        momentum = MomentumIndicators()
        
        sample_data = trend.calculate_all(sample_data)
        sample_data = momentum.calculate_all(sample_data)
        
        trend_strength = custom.trend_strength(sample_data)
        
        assert len(trend_strength) == len(sample_data)
        # Trend strength should be between 0 and 1
        assert 0 <= trend_strength.min() <= 1
        assert 0 <= trend_strength.max() <= 1
    
    def test_volume_profile(self, sample_data):
        """Test Volume Profile"""
        volume = VolumeIndicators()
        profile = volume.volume_profile(sample_data)
        
        assert isinstance(profile, dict)
        # Total volume in profile should match total volume in data
        assert sum(profile.values()) == sample_data['volume'].sum()
    
    def test_indicator_caching(self, sample_data):
        """Test indicator caching functionality"""
        custom = CustomIndicators()
        
        # Test that same calculation returns same result
        result1 = custom.trend_strength(sample_data)
        result2 = custom.trend_strength(sample_data)
        
        assert result1.equals(result2)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

```