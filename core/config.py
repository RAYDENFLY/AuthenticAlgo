"""
Configuration management for Bot Trading V2
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Config:
    """Configuration loader and manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            base_path = Path(__file__).parent.parent
            config_path = base_path / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Args:
            key: Configuration key (e.g., 'general.name' or 'exchanges.binance.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    def get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean"""
        value = self.get_env(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int_env(self, key: str, default: int = 0) -> int:
        """Get environment variable as integer"""
        value = self.get_env(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_float_env(self, key: str, default: float = 0.0) -> float:
        """Get environment variable as float"""
        value = self.get_env(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.get_env('ENV', 'development').lower() == 'production'
    
    @property
    def is_paper_trading(self) -> bool:
        """Check if in paper trading mode"""
        return self.get_env('TRADING_MODE', 'paper').lower() == 'paper'
    
    @property
    def trading_mode(self) -> str:
        """Get current trading mode"""
        return self.get_env('TRADING_MODE', 'paper').lower()
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config() -> None:
    """Reload global config instance"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()
