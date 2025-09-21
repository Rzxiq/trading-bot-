"""
Data models for market data and trading operations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    STOP = "STOP"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    OCO = "OCO"


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class OrderBookLevel:
    """Order book level data."""
    price: float
    quantity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"price": self.price, "quantity": self.quantity}


@dataclass
class OrderBook:
    """Order book data structure."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_update_id: int
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid price."""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask price."""
        return self.asks[0] if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None
    
    def get_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance."""
        bid_volume = sum(level.quantity for level in self.bids[:levels])
        ask_volume = sum(level.quantity for level in self.asks[:levels])
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        return (bid_volume - ask_volume) / total_volume


@dataclass
class Kline:
    """Candlestick data structure."""
    symbol: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    trades_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float
    interval: str
    
    def get_body_size(self) -> float:
        """Get candlestick body size."""
        return abs(self.close_price - self.open_price)
    
    def get_upper_shadow(self) -> float:
        """Get upper shadow size."""
        return self.high_price - max(self.open_price, self.close_price)
    
    def get_lower_shadow(self) -> float:
        """Get lower shadow size."""
        return min(self.open_price, self.close_price) - self.low_price
    
    def is_bullish(self) -> bool:
        """Check if candlestick is bullish."""
        return self.close_price > self.open_price
    
    def is_bearish(self) -> bool:
        """Check if candlestick is bearish."""
        return self.close_price < self.open_price


@dataclass
class Trade:
    """Trade data structure."""
    symbol: str
    trade_id: int
    price: float
    quantity: float
    buyer_order_id: int
    seller_order_id: int
    trade_time: datetime
    is_buyer_maker: bool
    
    def is_buy(self) -> bool:
        """Check if trade is a buy."""
        return not self.is_buyer_maker


@dataclass
class MarkPrice:
    """Mark price data structure."""
    symbol: str
    mark_price: float
    index_price: float
    estimated_settle_price: float
    funding_rate: float
    next_funding_time: datetime
    timestamp: datetime


@dataclass
class Liquidation:
    """Liquidation data structure."""
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    quantity: float
    price: float
    average_price: float
    order_status: str
    last_filled_quantity: float
    filled_accumulated_quantity: float
    trade_time: datetime


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    position_side: PositionSide
    position_amount: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    notional: float
    isolated: bool
    isolated_margin: float
    leverage: int
    initial_margin: float
    maint_margin: float
    timestamp: datetime
    
    def get_pnl_percentage(self) -> float:
        """Get PnL percentage."""
        if self.entry_price == 0:
            return 0.0
        return ((self.mark_price - self.entry_price) / self.entry_price) * 100


@dataclass
class Order:
    """Order data structure."""
    symbol: str
    order_id: int
    client_order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = "USDT"
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
    
    def get_remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity


@dataclass
class MarketData:
    """Combined market data structure."""
    symbol: str
    timestamp: datetime
    order_book: Optional[OrderBook] = None
    kline: Optional[Kline] = None
    trade: Optional[Trade] = None
    mark_price: Optional[MarkPrice] = None
    liquidation: Optional[Liquidation] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.order_book:
            data["order_book"] = {
                "bids": [level.to_dict() for level in self.order_book.bids],
                "asks": [level.to_dict() for level in self.order_book.asks],
                "last_update_id": self.order_book.last_update_id
            }
        
        if self.kline:
            data["kline"] = {
                "open_time": self.kline.open_time.isoformat(),
                "close_time": self.kline.close_time.isoformat(),
                "open_price": self.kline.open_price,
                "high_price": self.kline.high_price,
                "low_price": self.kline.low_price,
                "close_price": self.kline.close_price,
                "volume": self.kline.volume,
                "interval": self.kline.interval
            }
        
        if self.trade:
            data["trade"] = {
                "trade_id": self.trade.trade_id,
                "price": self.trade.price,
                "quantity": self.trade.quantity,
                "is_buy": self.trade.is_buy(),
                "trade_time": self.trade.trade_time.isoformat()
            }
        
        if self.mark_price:
            data["mark_price"] = {
                "mark_price": self.mark_price.mark_price,
                "index_price": self.mark_price.index_price,
                "funding_rate": self.mark_price.funding_rate,
                "next_funding_time": self.mark_price.next_funding_time.isoformat()
            }
        
        return data


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type.value,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    average_win: float
    average_loss: float
    profit_factor: float
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "latency_ms": self.latency_ms
        }