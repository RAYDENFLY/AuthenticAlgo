"""
Utility functions for Bot Trading V2
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np


def get_timestamp() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert timestamp to datetime object
    
    Args:
        timestamp: Unix timestamp (seconds or milliseconds)
        
    Returns:
        datetime object in UTC
    """
    # Handle milliseconds
    if timestamp > 10**10:
        timestamp = timestamp / 1000
    
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert datetime to timestamp in milliseconds
    
    Args:
        dt: datetime object
        
    Returns:
        Unix timestamp in milliseconds
    """
    return int(dt.timestamp() * 1000)


def round_decimal(value: float, decimals: int = 8) -> float:
    """
    Round decimal to specified places
    
    Args:
        value: Value to round
        decimals: Number of decimal places
        
    Returns:
        Rounded value
    """
    return round(value, decimals)


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to dictionary
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary representation
    """
    return df.to_dict(orient='records')


def dict_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert dictionary to DataFrame
    
    Args:
        data: Dictionary data
        
    Returns:
        Pandas DataFrame
    """
    return pd.DataFrame(data)


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe format
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
        
    Returns:
        True if valid, False otherwise
    """
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
    return timeframe in valid_timeframes


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h')
        
    Returns:
        Number of seconds
    """
    units = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
    }
    
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    
    if unit not in units:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    return value * units[unit]


def calculate_lot_size(
    balance: float,
    risk_percentage: float,
    stop_loss_percentage: float,
    price: float,
) -> float:
    """
    Calculate position size based on risk parameters
    
    Args:
        balance: Account balance
        risk_percentage: Risk percentage per trade (e.g., 1 for 1%)
        stop_loss_percentage: Stop loss percentage (e.g., 2 for 2%)
        price: Current price
        
    Returns:
        Position size
    """
    risk_amount = balance * (risk_percentage / 100)
    position_size = risk_amount / (stop_loss_percentage / 100)
    lot_size = position_size / price
    
    return lot_size


def is_valid_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        
    Returns:
        True if valid, False otherwise
    """
    return '/' in symbol and len(symbol.split('/')) == 2


def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
    """
    Retry function on exception
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
            continue
    
    raise last_exception


def moving_average(data: pd.Series, period: int) -> pd.Series:
    """Calculate simple moving average"""
    return data.rolling(window=period).mean()


def exponential_moving_average(data: pd.Series, period: int) -> pd.Series:
    """Calculate exponential moving average"""
    return data.ewm(span=period, adjust=False).mean()
