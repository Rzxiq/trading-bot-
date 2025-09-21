"""
Configuration management for the trading bot.
"""
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the trading bot."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_config(self) -> Dict[str, str]:
        """Get API configuration."""
        return self.get('api.binance', {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.get('trading', {})
    
    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration."""
        return self.get('websocket', {})
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.get('strategy', {})
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.get('risk', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.get('monitoring', {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.get('backtesting', {})
    
    def is_testnet(self) -> bool:
        """Check if running on testnet."""
        return self.get('api.binance.testnet', True)
    
    def get_symbols(self) -> list:
        """Get trading symbols."""
        return self.get('trading.symbols', ['BTCUSDT'])
    
    def get_leverage(self) -> int:
        """Get leverage setting."""
        return self.get('trading.leverage', 10)
    
    def get_max_position_size(self) -> float:
        """Get maximum position size."""
        return self.get('trading.max_position_size', 0.1)
    
    def get_max_daily_loss(self) -> float:
        """Get maximum daily loss percentage."""
        return self.get('trading.max_daily_loss', 0.05)
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown percentage."""
        return self.get('trading.max_drawdown', 0.15)