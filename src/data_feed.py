"""
WebSocket data feed module for real-time market data.
"""
import asyncio
import json
import websockets
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from websockets.exceptions import ConnectionClosed, WebSocketException

from .data_models import (
    OrderBook, OrderBookLevel, Kline, Trade, MarkPrice, Liquidation, MarketData
)
from .logger import TradingLogger


class BinanceWebSocketFeed:
    """WebSocket data feed for Binance Futures."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.reconnect_interval = config.get('reconnect_interval', 5)
        self.ping_interval = config.get('ping_interval', 30)
        self.ping_timeout = config.get('ping_timeout', 10)
        
        # Data callbacks
        self.data_callbacks: List[Callable[[MarketData], None]] = []
        self.order_book_callbacks: List[Callable[[OrderBook], None]] = []
        self.kline_callbacks: List[Callable[[Kline], None]] = []
        self.trade_callbacks: List[Callable[[Trade], None]] = []
        self.mark_price_callbacks: List[Callable[[MarkPrice], None]] = []
        self.liquidation_callbacks: List[Callable[[Liquidation], None]] = []
        
        # Data buffers
        self.order_books: Dict[str, OrderBook] = {}
        self.klines: Dict[str, List[Kline]] = {}
        self.mark_prices: Dict[str, MarkPrice] = {}
        
        # Stream subscriptions
        self.subscribed_streams: List[str] = []
        self.symbols: List[str] = []
        
    def add_data_callback(self, callback: Callable[[MarketData], None]) -> None:
        """Add callback for all market data."""
        self.data_callbacks.append(callback)
    
    def add_order_book_callback(self, callback: Callable[[OrderBook], None]) -> None:
        """Add callback for order book updates."""
        self.order_book_callbacks.append(callback)
    
    def add_kline_callback(self, callback: Callable[[Kline], None]) -> None:
        """Add callback for kline updates."""
        self.kline_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add callback for trade updates."""
        self.trade_callbacks.append(callback)
    
    def add_mark_price_callback(self, callback: Callable[[MarkPrice], None]) -> None:
        """Add callback for mark price updates."""
        self.mark_price_callbacks.append(callback)
    
    def add_liquidation_callback(self, callback: Callable[[Liquidation], None]) -> None:
        """Add callback for liquidation updates."""
        self.liquidation_callbacks.append(callback)
    
    def _build_stream_url(self, streams: List[str]) -> str:
        """Build WebSocket stream URL."""
        base_url = self.config.get('base_url', 'wss://fstream.binance.com/ws')
        stream_params = '/'.join(streams)
        return f"{base_url}/{stream_params}"
    
    def _create_streams(self, symbols: List[str]) -> List[str]:
        """Create stream names for given symbols."""
        streams = []
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            
            # Order book depth stream (20 levels, 100ms updates)
            streams.append(f"{symbol_lower}@depth20@100ms")
            
            # Kline streams (1m and 5m)
            streams.append(f"{symbol_lower}@kline_1m")
            streams.append(f"{symbol_lower}@kline_5m")
            
            # Aggregate trades
            streams.append(f"{symbol_lower}@aggTrade")
            
            # Mark price stream (1s updates)
            streams.append(f"{symbol_lower}@markPrice@1s")
            
            # Force order stream (liquidations)
            streams.append(f"{symbol_lower}@forceOrder")
        
        return streams
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionClosed, WebSocketException))
    )
    async def connect(self, symbols: List[str]) -> None:
        """Connect to WebSocket streams."""
        self.symbols = symbols
        streams = self._create_streams(symbols)
        self.subscribed_streams = streams
        
        url = self._build_stream_url(streams)
        self.logger.info(f"Connecting to WebSocket: {url}")
        
        try:
            self.websocket = await websockets.connect(
                url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=10
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            self.logger.info("WebSocket connected successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.logger.info("WebSocket disconnected")
    
    async def listen(self) -> None:
        """Listen for WebSocket messages."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            async for message in self.websocket:
                await self._handle_message(message)
                
        except ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
            await self._handle_reconnection()
            
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {e}")
            self.is_connected = False
            await self._handle_reconnection()
    
    async def _handle_reconnection(self) -> None:
        """Handle WebSocket reconnection."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        self.logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        await asyncio.sleep(self.reconnect_interval)
        
        try:
            await self.connect(self.symbols)
            await self.listen()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            await self._handle_reconnection()
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle stream data
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                stream_data = data['data']
                
                if '@depth' in stream:
                    await self._handle_order_book(stream_data)
                elif '@kline' in stream:
                    await self._handle_kline(stream_data)
                elif '@aggTrade' in stream:
                    await self._handle_trade(stream_data)
                elif '@markPrice' in stream:
                    await self._handle_mark_price(stream_data)
                elif '@forceOrder' in stream:
                    await self._handle_liquidation(stream_data)
            
            # Handle ping/pong
            elif 'ping' in data:
                await self._send_pong(data['ping'])
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    async def _send_pong(self, ping_id: int) -> None:
        """Send pong response to ping."""
        if self.websocket:
            await self.websocket.send(json.dumps({"pong": ping_id}))
    
    async def _handle_order_book(self, data: Dict[str, Any]) -> None:
        """Handle order book data."""
        try:
            symbol = data['s']
            timestamp = datetime.fromtimestamp(data['E'] / 1000)
            last_update_id = data['lastUpdateId']
            
            # Parse bids and asks
            bids = [OrderBookLevel(float(price), float(qty)) for price, qty in data['b']]
            asks = [OrderBookLevel(float(price), float(qty)) for price, qty in data['a']]
            
            order_book = OrderBook(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                last_update_id=last_update_id
            )
            
            # Update buffer
            self.order_books[symbol] = order_book
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                order_book=order_book
            )
            
            # Trigger callbacks
            for callback in self.order_book_callbacks:
                try:
                    callback(order_book)
                except Exception as e:
                    self.logger.error(f"Error in order book callback: {e}")
            
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling order book data: {e}")
    
    async def _handle_kline(self, data: Dict[str, Any]) -> None:
        """Handle kline data."""
        try:
            symbol = data['s']
            kline_data = data['k']
            
            kline = Kline(
                symbol=symbol,
                open_time=datetime.fromtimestamp(kline_data['t'] / 1000),
                close_time=datetime.fromtimestamp(kline_data['T'] / 1000),
                open_price=float(kline_data['o']),
                high_price=float(kline_data['h']),
                low_price=float(kline_data['l']),
                close_price=float(kline_data['c']),
                volume=float(kline_data['v']),
                quote_volume=float(kline_data['q']),
                trades_count=int(kline_data['n']),
                taker_buy_base_volume=float(kline_data['V']),
                taker_buy_quote_volume=float(kline_data['Q']),
                interval=kline_data['i']
            )
            
            # Update buffer
            if symbol not in self.klines:
                self.klines[symbol] = []
            self.klines[symbol].append(kline)
            
            # Keep only last 1000 klines per symbol
            if len(self.klines[symbol]) > 1000:
                self.klines[symbol] = self.klines[symbol][-1000:]
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                timestamp=kline.close_time,
                kline=kline
            )
            
            # Trigger callbacks
            for callback in self.kline_callbacks:
                try:
                    callback(kline)
                except Exception as e:
                    self.logger.error(f"Error in kline callback: {e}")
            
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling kline data: {e}")
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle trade data."""
        try:
            symbol = data['s']
            
            trade = Trade(
                symbol=symbol,
                trade_id=int(data['a']),
                price=float(data['p']),
                quantity=float(data['q']),
                buyer_order_id=int(data['b']),
                seller_order_id=int(data['a']),
                trade_time=datetime.fromtimestamp(data['T'] / 1000),
                is_buyer_maker=data['m']
            )
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                timestamp=trade.trade_time,
                trade=trade
            )
            
            # Trigger callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(trade)
                except Exception as e:
                    self.logger.error(f"Error in trade callback: {e}")
            
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling trade data: {e}")
    
    async def _handle_mark_price(self, data: Dict[str, Any]) -> None:
        """Handle mark price data."""
        try:
            symbol = data['s']
            
            mark_price = MarkPrice(
                symbol=symbol,
                mark_price=float(data['p']),
                index_price=float(data['i']),
                estimated_settle_price=float(data['P']),
                funding_rate=float(data['r']),
                next_funding_time=datetime.fromtimestamp(data['T'] / 1000),
                timestamp=datetime.fromtimestamp(data['E'] / 1000)
            )
            
            # Update buffer
            self.mark_prices[symbol] = mark_price
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                timestamp=mark_price.timestamp,
                mark_price=mark_price
            )
            
            # Trigger callbacks
            for callback in self.mark_price_callbacks:
                try:
                    callback(mark_price)
                except Exception as e:
                    self.logger.error(f"Error in mark price callback: {e}")
            
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling mark price data: {e}")
    
    async def _handle_liquidation(self, data: Dict[str, Any]) -> None:
        """Handle liquidation data."""
        try:
            symbol = data['s']
            
            liquidation = Liquidation(
                symbol=symbol,
                side=data['S'],
                order_type=data['o'],
                time_in_force=data['f'],
                quantity=float(data['q']),
                price=float(data['p']),
                average_price=float(data['ap']),
                order_status=data['X'],
                last_filled_quantity=float(data['l']),
                filled_accumulated_quantity=float(data['z']),
                trade_time=datetime.fromtimestamp(data['T'] / 1000)
            )
            
            # Create market data
            market_data = MarketData(
                symbol=symbol,
                timestamp=liquidation.trade_time,
                liquidation=liquidation
            )
            
            # Trigger callbacks
            for callback in self.liquidation_callbacks:
                try:
                    callback(liquidation)
                except Exception as e:
                    self.logger.error(f"Error in liquidation callback: {e}")
            
            for callback in self.data_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling liquidation data: {e}")
    
    def get_latest_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get latest order book for symbol."""
        return self.order_books.get(symbol)
    
    def get_latest_klines(self, symbol: str, count: int = 100) -> List[Kline]:
        """Get latest klines for symbol."""
        klines = self.klines.get(symbol, [])
        return klines[-count:] if klines else []
    
    def get_latest_mark_price(self, symbol: str) -> Optional[MarkPrice]:
        """Get latest mark price for symbol."""
        return self.mark_prices.get(symbol)
    
    def is_stream_healthy(self) -> bool:
        """Check if WebSocket stream is healthy."""
        return self.is_connected and self.websocket is not None