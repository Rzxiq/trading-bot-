"""
Order execution module for Binance Futures API.
"""
import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from urllib.parse import urlencode
import json

from binance.um_futures import UMFutures
from binance.error import ClientError, ServerError

from .data_models import (
    TradingSignal, Order, OrderSide, OrderType, OrderStatus, Position, 
    PositionSide, PerformanceMetrics
)
from .logger import TradingLogger


class BinanceFuturesExecutor:
    """Binance Futures order execution engine."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # API configuration
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.testnet = config.get('testnet', True)
        self.base_url = config.get('base_url', 'https://testnet.binancefuture.com')
        
        # Initialize Binance client
        self.client = UMFutures(
            key=self.api_key,
            secret=self.secret_key,
            base_url=self.base_url
        )
        
        # Execution state
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.account_info: Dict[str, Any] = {}
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.execution_times: List[float] = []
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Callbacks
        self.order_update_callbacks: List[Callable[[Order], None]] = []
        self.position_update_callbacks: List[Callable[[Position], None]] = []
        self.trade_execution_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def add_order_update_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for order updates."""
        self.order_update_callbacks.append(callback)
    
    def add_position_update_callback(self, callback: Callable[[Position], None]) -> None:
        """Add callback for position updates."""
        self.position_update_callbacks.append(callback)
    
    def add_trade_execution_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for trade executions."""
        self.trade_execution_callbacks.append(callback)
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def execute_signal(self, signal: TradingSignal) -> Optional[Order]:
        """Execute trading signal."""
        start_time = datetime.utcnow()
        
        try:
            await self._rate_limit()
            
            # Prepare order parameters
            order_params = self._prepare_order_params(signal)
            
            # Place order
            response = await self._place_order(order_params)
            
            if response:
                # Create order object
                order = self._create_order_from_response(response, signal)
                
                # Store active order
                self.active_orders[order.client_order_id] = order
                
                # Log trade
                self.logger.trade_log(
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=order.quantity,
                    price=order.price,
                    order_id=order.order_id,
                    status=order.status.value
                )
                
                # Track execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.execution_times.append(execution_time)
                
                # Trigger callbacks
                for callback in self.order_update_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        self.logger.error(f"Error in order update callback: {e}")
                
                return order
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            self.failed_trades += 1
            return None
    
    def _prepare_order_params(self, signal: TradingSignal) -> Dict[str, Any]:
        """Prepare order parameters for Binance API."""
        params = {
            'symbol': signal.symbol,
            'side': signal.side.value,
            'type': signal.order_type.value,
            'quantity': signal.quantity,
            'newClientOrderId': f"bot_{int(time.time() * 1000)}"
        }
        
        # Add price for limit orders
        if signal.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.TAKE_PROFIT]:
            params['price'] = signal.price
            params['timeInForce'] = 'GTC'
        
        # Add stop price for stop orders
        if signal.order_type in [OrderType.STOP_MARKET, OrderType.STOP]:
            params['stopPrice'] = signal.stop_loss
        
        # Add OCO order for stop loss and take profit
        if signal.stop_loss and signal.take_profit:
            params['type'] = 'OCO'
            params['stopPrice'] = signal.stop_loss
            params['stopLimitPrice'] = signal.stop_loss
            params['price'] = signal.take_profit
            params['timeInForce'] = 'GTC'
        
        return params
    
    async def _place_order(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Place order via Binance API."""
        try:
            if params['type'] == 'OCO':
                response = self.client.new_oco_order(**params)
            else:
                response = self.client.new_order(**params)
            
            return response
            
        except ClientError as e:
            self.logger.error(f"Client error placing order: {e}")
            return None
        except ServerError as e:
            self.logger.error(f"Server error placing order: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error placing order: {e}")
            return None
    
    def _create_order_from_response(self, response: Dict[str, Any], signal: TradingSignal) -> Order:
        """Create Order object from API response."""
        return Order(
            symbol=response['symbol'],
            order_id=int(response['orderId']),
            client_order_id=response['clientOrderId'],
            side=OrderSide(response['side']),
            order_type=OrderType(response['type']),
            quantity=float(response['origQty']),
            price=float(response.get('price', 0)),
            stop_price=float(response.get('stopPrice', 0)),
            time_in_force=response.get('timeInForce', 'GTC'),
            status=OrderStatus(response['status']),
            filled_quantity=float(response.get('executedQty', 0)),
            average_price=float(response.get('avgPrice', 0)),
            commission=float(response.get('commission', 0)),
            commission_asset=response.get('commissionAsset', 'USDT'),
            created_time=datetime.fromtimestamp(response['transactTime'] / 1000),
            updated_time=datetime.fromtimestamp(response['transactTime'] / 1000)
        )
    
    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel active order."""
        try:
            await self._rate_limit()
            
            response = self.client.cancel_order(symbol=symbol, orderId=order_id)
            
            if response:
                # Update order status
                client_order_id = response.get('clientOrderId')
                if client_order_id in self.active_orders:
                    self.active_orders[client_order_id].status = OrderStatus.CANCELED
                    self.active_orders[client_order_id].updated_time = datetime.utcnow()
                
                self.logger.info(f"Order cancelled: {symbol} {order_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all active orders for symbol."""
        try:
            await self._rate_limit()
            
            response = self.client.cancel_all_open_orders(symbol=symbol)
            
            if response:
                # Update all active orders for this symbol
                for order in self.active_orders.values():
                    if order.symbol == symbol:
                        order.status = OrderStatus.CANCELED
                        order.updated_time = datetime.utcnow()
                
                self.logger.info(f"All orders cancelled for {symbol}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            await self._rate_limit()
            
            response = self.client.account()
            self.account_info = response
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        try:
            await self._rate_limit()
            
            response = self.client.get_position_risk()
            positions = []
            
            for pos_data in response:
                if float(pos_data['positionAmt']) != 0:  # Only non-zero positions
                    position = Position(
                        symbol=pos_data['symbol'],
                        position_side=PositionSide(pos_data['positionSide']),
                        position_amount=float(pos_data['positionAmt']),
                        entry_price=float(pos_data['entryPrice']),
                        mark_price=float(pos_data['markPrice']),
                        unrealized_pnl=float(pos_data['unRealizedProfit']),
                        percentage=float(pos_data['percentage']),
                        notional=float(pos_data['notional']),
                        isolated=pos_data['isolated'] == 'true',
                        isolated_margin=float(pos_data['isolatedMargin']),
                        leverage=int(pos_data['leverage']),
                        initial_margin=float(pos_data['initialMargin']),
                        maint_margin=float(pos_data['maintMargin']),
                        timestamp=datetime.utcnow()
                    )
                    positions.append(position)
                    self.positions[position.symbol] = position
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        try:
            await self._rate_limit()
            
            params = {'symbol': symbol} if symbol else {}
            response = self.client.get_open_orders(**params)
            
            orders = []
            for order_data in response:
                order = Order(
                    symbol=order_data['symbol'],
                    order_id=int(order_data['orderId']),
                    client_order_id=order_data['clientOrderId'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data['type']),
                    quantity=float(order_data['origQty']),
                    price=float(order_data.get('price', 0)),
                    stop_price=float(order_data.get('stopPrice', 0)),
                    time_in_force=order_data.get('timeInForce', 'GTC'),
                    status=OrderStatus(order_data['status']),
                    filled_quantity=float(order_data.get('executedQty', 0)),
                    average_price=float(order_data.get('avgPrice', 0)),
                    created_time=datetime.fromtimestamp(order_data['time'] / 1000),
                    updated_time=datetime.fromtimestamp(order_data['updateTime'] / 1000)
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []
    
    async def update_order_status(self, order: Order) -> Order:
        """Update order status from exchange."""
        try:
            await self._rate_limit()
            
            response = self.client.query_order(
                symbol=order.symbol,
                orderId=order.order_id
            )
            
            # Update order with latest data
            order.status = OrderStatus(response['status'])
            order.filled_quantity = float(response.get('executedQty', 0))
            order.average_price = float(response.get('avgPrice', 0))
            order.updated_time = datetime.fromtimestamp(response['updateTime'] / 1000)
            
            # Check if order is filled
            if order.is_filled():
                self.successful_trades += 1
                
                # Record trade
                trade_record = {
                    'timestamp': order.updated_time,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.filled_quantity,
                    'price': order.average_price,
                    'pnl': 0.0,  # Will be calculated later
                    'order_id': order.order_id
                }
                
                self.trade_history.append(trade_record)
                
                # Trigger trade execution callbacks
                for callback in self.trade_execution_callbacks:
                    try:
                        callback(trade_record)
                    except Exception as e:
                        self.logger.error(f"Error in trade execution callback: {e}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error updating order status: {e}")
            return order
    
    async def monitor_orders(self) -> None:
        """Monitor active orders and update their status."""
        for client_order_id, order in list(self.active_orders.items()):
            if order.is_active():
                try:
                    updated_order = await self.update_order_status(order)
                    self.active_orders[client_order_id] = updated_order
                    
                    # Trigger order update callbacks
                    for callback in self.order_update_callbacks:
                        try:
                            callback(updated_order)
                        except Exception as e:
                            self.logger.error(f"Error in order update callback: {e}")
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring order {order.order_id}: {e}")
    
    async def close_position(self, symbol: str, side: OrderSide, quantity: float) -> Optional[Order]:
        """Close position by placing opposite order."""
        try:
            # Determine opposite side
            opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            
            # Create close signal
            close_signal = TradingSignal(
                symbol=symbol,
                side=opposite_side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                reason="Position close"
            )
            
            # Execute close order
            return await self.execute_signal(close_signal)
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        if not self.execution_times:
            return {
                'average_execution_time': 0,
                'max_execution_time': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'success_rate': 0
            }
        
        total_trades = self.successful_trades + self.failed_trades
        success_rate = self.successful_trades / total_trades if total_trades > 0 else 0
        
        return {
            'average_execution_time': sum(self.execution_times) / len(self.execution_times),
            'max_execution_time': max(self.execution_times),
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': success_rate,
            'total_trades': total_trades,
            'active_orders': len(self.active_orders)
        }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        if not self.trade_history:
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                total_pnl=0.0,
                daily_pnl=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                average_win=0.0,
                average_loss=0.0,
                profit_factor=0.0,
                latency_ms=0.0
            )
        
        # Calculate metrics
        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        average_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        average_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Calculate profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['pnl'] for t in self.trade_history]
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            sharpe_ratio = mean_return / (variance ** 0.5) if variance > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_pnl = 0
        max_pnl = 0
        max_drawdown = 0
        
        for trade in self.trade_history:
            cumulative_pnl += trade['pnl']
            max_pnl = max(max_pnl, cumulative_pnl)
            drawdown = max_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate average latency
        avg_latency = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            total_pnl=total_pnl,
            daily_pnl=0.0,  # Would need daily tracking
            win_rate=win_rate,
            total_trades=len(self.trade_history),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            latency_ms=avg_latency
        )