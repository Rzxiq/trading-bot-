"""
Technical indicators for real-time market analysis.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass
from .data_models import Kline, OrderBook


@dataclass
class IndicatorValue:
    """Indicator value with metadata."""
    value: float
    timestamp: float
    symbol: str
    indicator_name: str


class TechnicalIndicators:
    """Real-time technical indicators calculator."""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        """Simple Moving Average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    @staticmethod
    def ema(prices: List[float], period: int, previous_ema: Optional[float] = None) -> Optional[float]:
        """Exponential Moving Average."""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        
        if previous_ema is None:
            # Calculate SMA for first EMA
            return sum(prices[-period:]) / period
        else:
            return (prices[-1] * multiplier) + (previous_ema * (1 - multiplier))
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Optional[Tuple[float, float, float]]:
        """Bollinger Bands (upper, middle, lower)."""
        if len(prices) < period:
            return None
        
        sma = sum(prices[-period:]) / period
        variance = sum((price - sma) ** 2 for price in prices[-period:]) / period
        std = np.sqrt(variance)
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    @staticmethod
    def macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Tuple[float, float, float]]:
        """MACD (macd_line, signal_line, histogram)."""
        if len(prices) < slow_period:
            return None
        
        # Calculate EMAs
        fast_ema = TechnicalIndicators._calculate_ema_series(prices, fast_period)
        slow_ema = TechnicalIndicators._calculate_ema_series(prices, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        macd_values = [macd_line]  # This should be a series in real implementation
        signal_line = TechnicalIndicators.ema(macd_values, signal_period)
        
        if signal_line is None:
            return None
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def _calculate_ema_series(prices: List[float], period: int) -> Optional[float]:
        """Calculate EMA for a series of prices."""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]  # Start with first price
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def stochastic(klines: List[Kline], k_period: int = 14, d_period: int = 3) -> Optional[Tuple[float, float]]:
        """Stochastic Oscillator (%K, %D)."""
        if len(klines) < k_period:
            return None
        
        recent_klines = klines[-k_period:]
        current_close = recent_klines[-1].close_price
        lowest_low = min(kline.low_price for kline in recent_klines)
        highest_high = max(kline.high_price for kline in recent_klines)
        
        if highest_high == lowest_low:
            return 50.0, 50.0  # Neutral when no range
        
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (SMA of %K)
        k_values = []
        for i in range(len(klines) - k_period + 1, len(klines)):
            period_klines = klines[i-k_period+1:i+1]
            close = period_klines[-1].close_price
            low = min(kline.low_price for kline in period_klines)
            high = max(kline.high_price for kline in period_klines)
            
            if high != low:
                k_val = ((close - low) / (high - low)) * 100
                k_values.append(k_val)
        
        if len(k_values) < d_period:
            return k_percent, k_percent
        
        d_percent = sum(k_values[-d_period:]) / d_period
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(klines: List[Kline], period: int = 14) -> Optional[float]:
        """Williams %R."""
        if len(klines) < period:
            return None
        
        recent_klines = klines[-period:]
        current_close = recent_klines[-1].close_price
        highest_high = max(kline.high_price for kline in recent_klines)
        lowest_low = min(kline.low_price for kline in recent_klines)
        
        if highest_high == lowest_low:
            return -50.0  # Neutral when no range
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    @staticmethod
    def atr(klines: List[Kline], period: int = 14) -> Optional[float]:
        """Average True Range."""
        if len(klines) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(klines)):
            current = klines[i]
            previous = klines[i-1]
            
            tr1 = current.high_price - current.low_price
            tr2 = abs(current.high_price - previous.close_price)
            tr3 = abs(current.low_price - previous.close_price)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        return sum(true_ranges[-period:]) / period
    
    @staticmethod
    def adx(klines: List[Kline], period: int = 14) -> Optional[float]:
        """Average Directional Index."""
        if len(klines) < period + 1:
            return None
        
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(klines)):
            current = klines[i]
            previous = klines[i-1]
            
            high_diff = current.high_price - previous.high_price
            low_diff = previous.low_price - current.low_price
            
            plus_dm_val = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm_val = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            plus_dm.append(plus_dm_val)
            minus_dm.append(minus_dm_val)
        
        if len(plus_dm) < period:
            return None
        
        # Calculate smoothed values
        plus_dm_smooth = sum(plus_dm[-period:]) / period
        minus_dm_smooth = sum(minus_dm[-period:]) / period
        
        # Calculate True Range
        atr_val = TechnicalIndicators.atr(klines, period)
        if atr_val is None or atr_val == 0:
            return None
        
        # Calculate DI+ and DI-
        di_plus = (plus_dm_smooth / atr_val) * 100
        di_minus = (minus_dm_smooth / atr_val) * 100
        
        # Calculate DX
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
        
        return dx
    
    @staticmethod
    def volume_profile(klines: List[Kline], period: int = 20) -> Optional[Dict[str, float]]:
        """Volume profile analysis."""
        if len(klines) < period:
            return None
        
        recent_klines = klines[-period:]
        
        total_volume = sum(kline.volume for kline in recent_klines)
        total_quote_volume = sum(kline.quote_volume for kline in recent_klines)
        
        # Volume-weighted average price
        vwap = total_quote_volume / total_volume if total_volume > 0 else 0
        
        # Volume trend
        first_half_volume = sum(kline.volume for kline in recent_klines[:period//2])
        second_half_volume = sum(kline.volume for kline in recent_klines[period//2:])
        volume_trend = (second_half_volume - first_half_volume) / first_half_volume if first_half_volume > 0 else 0
        
        return {
            'total_volume': total_volume,
            'average_volume': total_volume / period,
            'vwap': vwap,
            'volume_trend': volume_trend
        }
    
    @staticmethod
    def order_book_imbalance(order_book: OrderBook, levels: int = 5) -> float:
        """Calculate order book imbalance."""
        if not order_book or len(order_book.bids) < levels or len(order_book.asks) < levels:
            return 0.0
        
        bid_volume = sum(level.quantity for level in order_book.bids[:levels])
        ask_volume = sum(level.quantity for level in order_book.asks[:levels])
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    @staticmethod
    def order_book_pressure(order_book: OrderBook, levels: int = 10) -> Dict[str, float]:
        """Calculate order book pressure metrics."""
        if not order_book or len(order_book.bids) < levels or len(order_book.asks) < levels:
            return {'bid_pressure': 0.0, 'ask_pressure': 0.0, 'net_pressure': 0.0}
        
        bid_volume = sum(level.quantity for level in order_book.bids[:levels])
        ask_volume = sum(level.quantity for level in order_book.asks[:levels])
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return {'bid_pressure': 0.0, 'ask_pressure': 0.0, 'net_pressure': 0.0}
        
        bid_pressure = bid_volume / total_volume
        ask_pressure = ask_volume / total_volume
        net_pressure = bid_pressure - ask_pressure
        
        return {
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'net_pressure': net_pressure
        }
    
    @staticmethod
    def volatility(klines: List[Kline], period: int = 20) -> Optional[float]:
        """Calculate price volatility."""
        if len(klines) < period:
            return None
        
        recent_klines = klines[-period:]
        returns = []
        
        for i in range(1, len(recent_klines)):
            current = recent_klines[i]
            previous = recent_klines[i-1]
            
            if previous.close_price > 0:
                ret = (current.close_price - previous.close_price) / previous.close_price
                returns.append(ret)
        
        if len(returns) < 2:
            return None
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((ret - mean_return) ** 2 for ret in returns) / len(returns)
        volatility = np.sqrt(variance)
        
        return volatility * 100  # Return as percentage


class IndicatorBuffer:
    """Buffer for storing and managing indicator values."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.indicators: Dict[str, List[IndicatorValue]] = {}
    
    def add_indicator(self, symbol: str, indicator_name: str, value: float, timestamp: float) -> None:
        """Add indicator value to buffer."""
        key = f"{symbol}_{indicator_name}"
        
        if key not in self.indicators:
            self.indicators[key] = []
        
        indicator_value = IndicatorValue(
            value=value,
            timestamp=timestamp,
            symbol=symbol,
            indicator_name=indicator_name
        )
        
        self.indicators[key].append(indicator_value)
        
        # Keep only recent values
        if len(self.indicators[key]) > self.max_size:
            self.indicators[key] = self.indicators[key][-self.max_size:]
    
    def get_latest(self, symbol: str, indicator_name: str) -> Optional[IndicatorValue]:
        """Get latest indicator value."""
        key = f"{symbol}_{indicator_name}"
        if key not in self.indicators or not self.indicators[key]:
            return None
        return self.indicators[key][-1]
    
    def get_values(self, symbol: str, indicator_name: str, count: int = 100) -> List[IndicatorValue]:
        """Get recent indicator values."""
        key = f"{symbol}_{indicator_name}"
        if key not in self.indicators:
            return []
        return self.indicators[key][-count:]
    
    def get_all_indicators(self, symbol: str) -> Dict[str, List[IndicatorValue]]:
        """Get all indicators for a symbol."""
        result = {}
        for key, values in self.indicators.items():
            if key.startswith(f"{symbol}_"):
                indicator_name = key.split("_", 1)[1]
                result[indicator_name] = values
        return result