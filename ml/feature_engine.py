"""
Feature Engineering Module
Advanced feature extraction for ML trading models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from dataclasses import dataclass

from core.logger import get_logger
from indicators.momentum import RSI, MACD, StochasticOscillator
from indicators.trend import SMA, EMA, ADX
from indicators.volatility import BollingerBands, ATR
from indicators.volume import OBV, VWAP


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_periods: List[int] = None
    technical_indicators: List[str] = None
    price_features: List[str] = None
    volume_features: List[str] = None
    time_features: List[str] = None
    feature_scaling: str = 'standard'  # 'standard', 'minmax', or None
    feature_selection: bool = True
    n_features: int = 50


class FeatureEngine:
    """
    Advanced feature engineering for trading ML models
    Extracts technical, price, volume, and time-based features
    """
    
    def __init__(self, config: dict):
        """Initialize feature engine"""
        self.config = config
        self.logger = get_logger()
        
        # Feature configuration
        self.feature_config = config.get('feature_engineering', {})
        
        # GTX 1050 Ti Optimization: Support reduced feature sets
        self.lookback_periods = self.feature_config.get('lookback_periods', [5, 10, 20, 50])
        self.technical_indicators = self.feature_config.get('technical_indicators', [
            'rsi', 'macd', 'bollinger_bands', 'atr', 'obv', 'adx'
        ])
        self.price_features = self.feature_config.get('price_features', [
            'returns', 'volatility', 'momentum', 'price_acceleration'
        ])
        self.volume_features = self.feature_config.get('volume_features', [
            'volume_ratio', 'volume_velocity', 'volume_volatility'
        ])
        self.time_features = self.feature_config.get('time_features', [
            'hour', 'day_of_week', 'month', 'is_weekend'
        ])
        
        # Feature selection configuration
        self.feature_selection_enabled = self.feature_config.get('feature_selection', True)
        self.n_features = self.feature_config.get('n_features', 50)
        self.feature_scaling_method = self.feature_config.get('feature_scaling', 'standard')
        
        # GTX 1050 Ti: Log optimization mode
        if self.n_features <= 30:
            self.logger.info(f"🎯 GTX 1050 Ti Optimization Mode: {self.n_features} features max")
        
        # Feature engineering tools
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
        self.logger.info("Feature Engine initialized")
    
    def transform(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Transform raw data into features for ML
        
        Args:
            data: Raw OHLCV data
            fit_scaler: Whether to fit scaler (True for training data)
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature transformation...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        
        # 1. Technical indicator features
        df = self._add_technical_features(df)
        
        # 2. Price-based features
        df = self._add_price_features(df)
        
        # 3. Volume-based features
        df = self._add_volume_features(df)
        
        # 4. Time-based features
        df = self._add_time_features(df)
        
        # 5. Lag features
        df = self._add_lag_features(df)
        
        # 6. Rolling window features
        df = self._add_rolling_features(df)
        
        # 7. Target variable (if not present)
        if 'target' not in df.columns:
            df = self._create_target_variable(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature scaling
        if self.feature_config.get('feature_scaling'):
            df = self._scale_features(df, fit_scaler)
        
        # Feature selection
        if self.feature_config.get('feature_selection', False) and fit_scaler and 'target' in df.columns:
            df = self._select_features(df, df['target'])
        
        self.logger.info(f"Feature transformation complete. Final shape: {df.shape}")
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        self.logger.debug("Adding technical indicator features...")
        
        # RSI
        if 'rsi' in self.technical_indicators:
            for period in [14, 21, 28]:
                rsi_calculator = RSI(period=period)
                df[f'rsi_{period}'] = rsi_calculator.calculate(df['close'])
        
        # MACD
        if 'macd' in self.technical_indicators:
            macd_calculator = MACD(fast_period=12, slow_period=26, signal_period=9)
            macd_data = macd_calculator.calculate(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        if 'bollinger_bands' in self.technical_indicators:
            bb_calculator = BollingerBands(period=20, std_dev=2.0)
            bb_data = bb_calculator.calculate(df['close'])
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
            df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # ATR
        if 'atr' in self.technical_indicators:
            for period in [14, 21]:
                atr_calculator = ATR(period=period)
                df[f'atr_{period}'] = atr_calculator.calculate(df['high'], df['low'], df['close'])
        
        # ADX
        if 'adx' in self.technical_indicators:
            adx_calculator = ADX(period=14)
            adx_data = adx_calculator.calculate(df['high'], df['low'], df['close'])
            df['adx'] = adx_data['adx']
            df['plus_di'] = adx_data['plus_di']
            df['minus_di'] = adx_data['minus_di']
        
        # OBV
        if 'obv' in self.technical_indicators and 'volume' in df.columns:
            obv_calculator = OBV()
            df['obv'] = obv_calculator.calculate(df['close'], df['volume'])
        
        # Stochastic
        if 'stochastic' in self.technical_indicators:
            stoch_calculator = StochasticOscillator(k_period=14, d_period=3)
            stoch_data = stoch_calculator.calculate(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_data['k']
            df['stoch_d'] = stoch_data['d']
        
        # Moving Averages
        for period in [10, 20, 50, 200]:
            sma_calculator = SMA(period=period)
            df[f'sma_{period}'] = sma_calculator.calculate(df['close'])
            
            ema_calculator = EMA(period=period)
            df[f'ema_{period}'] = ema_calculator.calculate(df['close'])
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        self.logger.debug("Adding price-based features...")
        
        # Returns at different periods
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Price acceleration (rate of change of momentum)
        df['price_acceleration'] = df['momentum_5'] - df['momentum_10']
        
        # High-Low range
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close range
        df['open_close_range'] = (df['close'] - df['open']) / df['open']
        
        # Price position relative to recent range
        df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                                  (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                                  (df['high'].rolling(50).max() - df['low'].rolling(50).min())
        
        # Distance from moving averages
        for period in [20, 50, 200]:
            if f'sma_{period}' in df.columns:
                df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            return df
        
        self.logger.debug("Adding volume-based features...")
        
        # Volume ratio (current volume vs average)
        for window in [5, 10, 20]:
            df[f'volume_ratio_{window}'] = df['volume'] / df['volume'].rolling(window).mean()
        
        # Volume velocity (change in volume)
        df['volume_velocity'] = df['volume'].pct_change()
        
        # Volume volatility
        df['volume_volatility'] = df['volume'].rolling(20).std()
        
        # Volume-price correlation
        df['volume_price_corr'] = df['volume'].rolling(20).corr(df['close'])
        
        # On-Balance Volume features
        if 'obv' in df.columns:
            df['obv_velocity'] = df['obv'].pct_change()
            df['obv_momentum'] = df['obv'] / df['obv'].shift(10) - 1
        
        # Volume-weighted features
        if 'vwap' not in df.columns and 'volume' in df.columns:
            vwap_calculator = VWAP()
            df['vwap'] = vwap_calculator.calculate(df['high'], df['low'], df['close'], df['volume'])
        
        if 'vwap' in df.columns:
            df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not DatetimeIndex, skipping time features")
            return df
        
        self.logger.debug("Adding time-based features...")
        
        # Time of day features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df.index.day == 1).astype(int)
        df['is_month_end'] = (df.index.day == df.index.days_in_month).astype(int)
        
        # Market session features (simplified)
        df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 16) | (df['hour'] < 24)).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        self.logger.debug("Adding lag features...")
        
        features_to_lag = ['close', 'volume', 'return_1', 'volatility_5']
        
        # Add RSI and MACD if available
        if 'rsi_14' in df.columns:
            features_to_lag.append('rsi_14')
        if 'macd' in df.columns:
            features_to_lag.append('macd')
        
        available_features = [f for f in features_to_lag if f in df.columns]
        
        for feature in available_features:
            for lag in [1, 2, 3, 5, 10]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        self.logger.debug("Adding rolling features...")
        
        # Select key columns for rolling features
        key_columns = []
        if 'close' in df.columns:
            key_columns.append('close')
        if 'volume' in df.columns:
            key_columns.append('volume')
        if 'return_1' in df.columns:
            key_columns.append('return_1')
        if 'rsi_14' in df.columns:
            key_columns.append('rsi_14')
        
        for column in key_columns:
            for window in [5, 10, 20]:
                # Rolling mean
                df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
                
                # Rolling standard deviation
                df[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
                
                # Rolling min/max
                df[f'{column}_rolling_min_{window}'] = df[column].rolling(window).min()
                df[f'{column}_rolling_max_{window}'] = df[column].rolling(window).max()
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Create target variable for prediction
        Target: Future price movement in the next 'horizon' periods
        
        Args:
            df: DataFrame with price data
            horizon: Number of periods to look ahead
            
        Returns:
            DataFrame with 'target' column
        """
        # Future return
        df['target'] = (df['close'].shift(-horizon) / df['close'] - 1) * 100
        
        # Alternatively, classification target
        # threshold = 0.02  # 2%
        # df['target_class'] = np.where(df['target'] > threshold, 1,
        #                              np.where(df['target'] < -threshold, -1, 0))
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature set"""
        self.logger.debug("Handling missing values...")
        
        # Drop columns with too many missing values
        missing_threshold = 0.3
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        
        if len(columns_to_drop) > 0:
            df = df.drop(columns=columns_to_drop)
            self.logger.info(f"Dropped {len(columns_to_drop)} columns with >{missing_threshold:.0%} missing values")
        
        # Fill remaining missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill for time series
        df[numeric_columns] = df[numeric_columns].ffill()
        
        # Backward fill any remaining missing values
        df[numeric_columns] = df[numeric_columns].bfill()
        
        # Fill any remaining with 0
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Scale features using StandardScaler or MinMaxScaler"""
        scaling_method = self.feature_config.get('feature_scaling', 'standard')
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Don't scale target variable
        if 'target' in numeric_columns:
            numeric_columns.remove('target')
        if 'target_class' in numeric_columns:
            numeric_columns.remove('target_class')
        
        if scaling_method == 'standard':
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
                self.logger.info("Fitted StandardScaler")
            else:
                df[numeric_columns] = self.scaler.transform(df[numeric_columns])
        
        elif scaling_method == 'minmax':
            if fit_scaler or self.scaler is None:
                self.scaler = MinMaxScaler()
                df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
                self.logger.info("Fitted MinMaxScaler")
            else:
                df[numeric_columns] = self.scaler.transform(df[numeric_columns])
        
        return df
    
    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features using statistical tests"""
        n_features = self.feature_config.get('n_features', 50)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Don't include target in feature selection
        feature_columns = [col for col in numeric_columns if col not in ['target', 'target_class']]
        
        if len(feature_columns) <= n_features:
            self.selected_features = feature_columns
            self.logger.info(f"Using all {len(feature_columns)} features (less than n_features={n_features})")
            return df
        
        X = df[feature_columns].fillna(0)
        y = target.fillna(0)
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        # Use mutual information for feature selection
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(feature_columns)))
        self.feature_selector.fit(X, y)
        
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        self.logger.info(f"Selected {len(self.selected_features)} features from {len(feature_columns)}")
        
        # Keep only selected features and target
        columns_to_keep = self.selected_features.copy()
        if 'target' in df.columns:
            columns_to_keep.append('target')
        if 'target_class' in df.columns:
            columns_to_keep.append('target_class')
        
        return df[columns_to_keep]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selector"""
        if self.feature_selector is None or self.selected_features is None:
            return pd.DataFrame()
        
        scores = self.feature_selector.scores_[self.feature_selector.get_support()]
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance_score': scores
        }).sort_values('importance_score', ascending=False)
        
        return importance_df
    
    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Get summary of features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'total_features': len(numeric_columns),
            'feature_categories': {
                'technical_indicators': len([col for col in numeric_columns if any(ind in col for ind in ['rsi', 'macd', 'bb', 'atr', 'obv', 'adx', 'stoch'])]),
                'price_features': len([col for col in numeric_columns if any(term in col for term in ['return', 'volatility', 'momentum', 'acceleration', 'distance'])]),
                'volume_features': len([col for col in numeric_columns if 'volume' in col or 'vwap' in col]),
                'time_features': len([col for col in numeric_columns if col in ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']]),
                'lag_features': len([col for col in numeric_columns if 'lag' in col]),
                'rolling_features': len([col for col in numeric_columns if 'rolling' in col])
            },
            'missing_values': df.isnull().sum().sum(),
            'feature_dtypes': df.dtypes.value_counts().to_dict()
        }
        
        return summary
