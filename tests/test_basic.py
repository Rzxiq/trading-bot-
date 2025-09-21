"""
Basic tests for the trading bot.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import Config
from src.logger import TradingLogger
from src.data_models import Kline, OrderBook, OrderBookLevel, TradingSignal, OrderSide, OrderType
from src.indicators import TechnicalIndicators
from src.strategy import RSIBollingerStrategy


class TestTechnicalIndicators:
    """Test technical indicators."""
    
    def test_sma(self):
        """Test Simple Moving Average."""
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sma = TechnicalIndicators.sma(prices, 5)
        assert sma == 8.0  # Average of last 5 values
    
    def test_rsi(self):
        """Test RSI calculation."""
        prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.25, 47.25]
        rsi = TechnicalIndicators.rsi(prices, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        bb = TechnicalIndicators.bollinger_bands(prices, 20, 2)
        assert bb is not None
        upper, middle, lower = bb
        assert upper > middle > lower


class TestDataModels:
    """Test data models."""
    
    def test_kline_creation(self):
        """Test Kline object creation."""
        kline = Kline(
            symbol="BTCUSDT",
            open_time=datetime.utcnow(),
            close_time=datetime.utcnow(),
            open_price=50000.0,
            high_price=51000.0,
            low_price=49000.0,
            close_price=50500.0,
            volume=100.0,
            quote_volume=5050000.0,
            trades_count=1000,
            taker_buy_base_volume=50.0,
            taker_buy_quote_volume=2525000.0,
            interval="1m"
        )
        
        assert kline.symbol == "BTCUSDT"
        assert kline.is_bullish() == True
        assert kline.get_body_size() == 500.0
    
    def test_order_book_creation(self):
        """Test OrderBook object creation."""
        bids = [OrderBookLevel(50000.0, 1.0), OrderBookLevel(49999.0, 2.0)]
        asks = [OrderBookLevel(50001.0, 1.5), OrderBookLevel(50002.0, 2.5)]
        
        order_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            last_update_id=12345
        )
        
        assert order_book.get_spread() == 1.0
        assert order_book.get_mid_price() == 50000.5
        assert order_book.get_imbalance(2) == 0.0  # Equal volume on both sides


class TestStrategy:
    """Test trading strategy."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2,
            'order_book_levels': 20,
            'min_volume': 1000000,
            'position_size': 0.1,
            'max_positions': 3,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'min_confidence': 0.6,
            'require_order_book_confirmation': True
        }
        
        self.logger = Mock(spec=TradingLogger)
        self.strategy = RSIBollingerStrategy(self.config, self.logger)
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.name == "RSIBollingerStrategy"
        assert self.strategy.rsi_period == 14
        assert self.strategy.bb_period == 20
    
    def test_required_indicators(self):
        """Test required indicators list."""
        indicators = self.strategy.get_required_indicators()
        expected = ['rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'order_book', 'volume_profile']
        assert set(indicators) == set(expected)
    
    @pytest.mark.asyncio
    async def test_strategy_analysis_no_data(self):
        """Test strategy analysis with no data."""
        market_data = {
            'symbol': 'BTCUSDT',
            'indicators': {},
            'signals': {}
        }
        
        signal = await self.strategy.analyze(market_data)
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_strategy_analysis_oversold(self):
        """Test strategy analysis with oversold conditions."""
        market_data = {
            'symbol': 'BTCUSDT',
            'indicators': {
                'rsi': 25,  # Oversold
                'bb_upper': 52000,
                'bb_middle': 50000,
                'bb_lower': 48000,
                'order_book': {
                    'imbalance': 0.3  # Bullish imbalance
                },
                'volume_profile': {
                    'total_volume': 2000000  # Above minimum
                },
                'price_action': {
                    'close_price': 47500  # Below lower band
                }
            },
            'signals': {}
        }
        
        signal = await self.strategy.analyze(market_data)
        assert signal is not None
        assert signal.side == OrderSide.BUY
        assert signal.symbol == 'BTCUSDT'


class TestConfig:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Create a temporary config file
        config_content = """
api:
  binance:
    api_key: "test_key"
    secret_key: "test_secret"
    testnet: true

trading:
  symbols: ["BTCUSDT"]
  leverage: 10
  max_position_size: 0.1
"""
        
        with open('test_config.yaml', 'w') as f:
            f.write(config_content)
        
        try:
            config = Config('test_config.yaml')
            assert config.get('api.binance.api_key') == 'test_key'
            assert config.get_symbols() == ['BTCUSDT']
            assert config.get_leverage() == 10
        finally:
            import os
            os.remove('test_config.yaml')


if __name__ == '__main__':
    pytest.main([__file__])