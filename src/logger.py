"""
Advanced logging system for the trading bot.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import asyncio
from datetime import datetime


class TradingLogger:
    """Advanced logging system with multiple outputs and alerting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logger()
    
    def setup_logger(self) -> None:
        """Setup logger with multiple handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler
        logger.add(
            sys.stdout,
            level=self.config.get('level', 'INFO'),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # File handler
        log_file = self.config.get('file', 'logs/trading_bot.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=self.config.get('level', 'INFO'),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=self.config.get('max_file_size', '10MB'),
            retention=self.config.get('backup_count', 5),
            compression="zip"
        )
        
        # Error file handler
        error_file = str(Path(log_file).parent / "errors.log")
        logger.add(
            error_file,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days"
        )
    
    async def send_telegram_alert(self, message: str, level: str = "INFO") -> None:
        """Send alert to Telegram."""
        telegram_config = self.config.get('telegram', {})
        if not telegram_config.get('enabled', False):
            return
        
        try:
            import aiohttp
            
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                logger.warning("Telegram credentials not configured")
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': f"[{level}] {message}",
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.debug("Telegram alert sent successfully")
                    else:
                        logger.error(f"Failed to send Telegram alert: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def send_discord_alert(self, message: str, level: str = "INFO") -> None:
        """Send alert to Discord."""
        discord_config = self.config.get('discord', {})
        if not discord_config.get('enabled', False):
            return
        
        try:
            import aiohttp
            
            webhook_url = discord_config.get('webhook_url')
            if not webhook_url:
                logger.warning("Discord webhook URL not configured")
                return
            
            # Color coding based on level
            color_map = {
                'INFO': 0x00ff00,      # Green
                'WARNING': 0xffff00,   # Yellow
                'ERROR': 0xff0000,     # Red
                'CRITICAL': 0x800080   # Purple
            }
            
            embed = {
                'title': f'Trading Bot Alert - {level}',
                'description': message,
                'color': color_map.get(level, 0x00ff00),
                'timestamp': datetime.utcnow().isoformat(),
                'footer': {
                    'text': 'Binance Futures Trading Bot'
                }
            }
            
            data = {'embeds': [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=data) as response:
                    if response.status == 204:
                        logger.debug("Discord alert sent successfully")
                    else:
                        logger.error(f"Failed to send Discord alert: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
    
    async def alert(self, message: str, level: str = "INFO") -> None:
        """Send alert to all configured channels."""
        logger.log(level, message)
        
        # Send to external channels for important messages
        if level in ['WARNING', 'ERROR', 'CRITICAL']:
            await asyncio.gather(
                self.send_telegram_alert(message, level),
                self.send_discord_alert(message, level),
                return_exceptions=True
            )
    
    def info(self, message: str) -> None:
        """Log info message."""
        logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        logger.critical(message)
    
    def trade_log(self, symbol: str, side: str, quantity: float, price: float, 
                  order_id: str, status: str) -> None:
        """Log trade execution details."""
        message = (f"TRADE | {symbol} | {side} | {quantity} @ {price} | "
                  f"OrderID: {order_id} | Status: {status}")
        logger.info(message)
    
    def performance_log(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        message = f"PERFORMANCE | {metrics}"
        logger.info(message)
    
    def risk_log(self, symbol: str, risk_type: str, value: float, threshold: float) -> None:
        """Log risk management events."""
        message = f"RISK | {symbol} | {risk_type} | Value: {value} | Threshold: {threshold}"
        logger.warning(message)