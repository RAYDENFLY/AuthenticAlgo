import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
from dataclasses import dataclass

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
        self.config = config
        self.logger = logging.getLogger("ml.feature_engine")
        
        # Feature configuration
        self.feature_config = config.get('feature_engineering', {})
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
        
        # Feature engineering tools
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
        self.logger.info("Feature Engine initialized")
    
    def transform(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Transform raw data into features for ML
        """
        self.logger.info("Starting feature transformation...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
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
        if self.feature_config.get('feature_selection', False) and fit_scaler:
            df = self._select_features(df, df['target'])
        
        self.logger.info(f"Feature transformation complete. Final shape: {df.shape}")
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        from indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Calculate all technical indicators
        df = indicators.get_all_indicators(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns at different periods
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()
        
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
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                              (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            return df
        
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
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not DatetimeIndex, skipping time features")
            return df
        
        # Time of day features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
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
        features_to_lag = [
            'close', 'volume', 'return_1', 'volatility_5', 'rsi_14', 'macd'
        ]
        
        available_features = [f for f in features_to_lag if f in df.columns]
        
        for feature in available_features:
            for lag in [1, 2, 3, 5, 10]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns[:10]:  # Limit to first 10 columns to avoid explosion
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
        """
        # Future return
        df['target'] = (df['close'].shift(-horizon) / df['close'] - 1) * 100
        
        # Alternatively, we can create a classification target
        # threshold = 0.02  # 2%
        # df['target_class'] = np.where(df['target'] > threshold, 1,
        #                              np.where(df['target'] < -threshold, -1, 0))
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature set"""
        # Drop columns with too many missing values
        missing_threshold = 0.3
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        df = df.drop(columns=columns_to_drop)
        
        # Fill remaining missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill for time series
        df[numeric_columns] = df[numeric_columns].ffill()
        
        # Backward fill any remaining missing values
        df[numeric_columns] = df[numeric_columns].bfill()
        
        # Fill any remaining with 0
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        self.logger.info(f"Dropped {len(columns_to_drop)} columns with >{missing_threshold:.0%} missing values")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Scale features using StandardScaler or MinMaxScaler"""
        scaling_method = self.feature_config.get('feature_scaling', 'standard')
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Don't scale target variable
        if 'target' in numeric_columns:
            numeric_columns = numeric_columns.drop('target')
        if 'target_class' in numeric_columns:
            numeric_columns = numeric_columns.drop('target_class')
        
        if scaling_method == 'standard':
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(df[numeric_columns])
            df[numeric_columns] = self.scaler.transform(df[numeric_columns])
        
        elif scaling_method == 'minmax':
            if fit_scaler or self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(df[numeric_columns])
            df[numeric_columns] = self.scaler.transform(df[numeric_columns])
        
        return df
    
    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features using statistical tests"""
        n_features = self.feature_config.get('n_features', 50)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Don't include target in feature selection
        feature_columns = [col for col in numeric_columns if col not in ['target', 'target_class']]
        
        if len(feature_columns) <= n_features:
            self.selected_features = feature_columns
            return df
        
        X = df[feature_columns].fillna(0)
        y = target.fillna(0)
        
        # Use mutual information for feature selection
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        self.feature_selector.fit(X, y)
        
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        self.logger.info(f"Selected {len(self.selected_features)} features from {len(feature_columns)}")
        
        # Keep only selected features and target
        columns_to_keep = self.selected_features + ['target'] if 'target' in df.columns else self.selected_features
        if 'target_class' in df.columns:
            columns_to_keep.append('target_class')
        
        return df[columns_to_keep]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selector"""
        if self.feature_selector is None or self.selected_features is None:
            return pd.DataFrame()
        
        scores = self.feature_selector.scores_
        features = self.selected_features
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance_score': scores
        }).sort_values('importance_score', ascending=False)
        
        return importance_df
    
    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Get summary of features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'total_features': len(numeric_columns),
            'feature_categories': {
                'technical_indicators': len([col for col in numeric_columns if any(ind in col for ind in ['rsi', 'macd', 'bb', 'atr', 'obv', 'adx'])]),
                'price_features': len([col for col in numeric_columns if any(term in col for term in ['return', 'volatility', 'momentum', 'acceleration'])]),
                'volume_features': len([col for col in numeric_columns if 'volume' in col]),
                'time_features': len([col for col in numeric_columns if col in ['hour', 'day_of_week', 'month', 'is_weekend']]),
                'lag_features': len([col for col in numeric_columns if 'lag' in col]),
                'rolling_features': len([col for col in numeric_columns if 'rolling' in col])
            },
            'missing_values': df.isnull().sum().sum(),
            'feature_dtypes': df.dtypes.value_counts().to_dict()
        }
        
        return summary