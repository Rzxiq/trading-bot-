"""
Backtesting engine and paper trading mode.
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .data_models import Kline, TradingSignal, OrderSide, OrderType, Order, OrderStatus, Position, PositionSide
from .logger import TradingLogger


@dataclass
class BacktestResult:
    """Backtest result data structure."""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_pnl: float
    daily_returns: List[float]
    equity_curve: List[float]
    trade_history: List[Dict[str, Any]]


class PaperTradingExecutor:
    """Paper trading executor for simulation."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Simulation parameters
        self.initial_balance = config.get('initial_balance', 10000)
        self.commission = config.get('commission', 0.0004)  # 0.04%
        self.slippage = config.get('slippage', 0.0001)  # 0.01%
        
        # Account state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.daily_equity: List[float] = []
        self.max_equity = self.initial_balance
        
        # Callbacks
        self.order_update_callbacks: List[Callable[[Order], None]] = []
        self.position_update_callbacks: List[Callable[[Position], None]] = []
        self.trade_execution_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def add_order_update_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for order updates."""
        self.order_update_callbacks.append(callback)
    
    def add_position_update_callback(self, callback: Callable[[Position], None]) -> None:
        """Add callback for position updates."""
        self.position_update_callbacks.append(callback)
    
    def add_trade_execution_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for trade executions."""
        self.trade_execution_callbacks.append(callback)
    
    async def execute_signal(self, signal: TradingSignal, current_price: float) -> Optional[Order]:
        """Execute trading signal in paper trading mode."""
        try:
            # Calculate execution price with slippage
            if signal.side == OrderSide.BUY:
                execution_price = current_price * (1 + self.slippage)
            else:
                execution_price = current_price * (1 - self.slippage)
            
            # Calculate order value
            order_value = signal.quantity * execution_price
            
            # Check if we have enough balance
            if order_value > self.balance:
                self.logger.warning(f"Insufficient balance for order: {order_value} > {self.balance}")
                return None
            
            # Create order
            order_id = len(self.orders) + 1
            order = Order(
                symbol=signal.symbol,
                order_id=order_id,
                client_order_id=f"paper_{int(datetime.utcnow().timestamp() * 1000)}",
                side=signal.side,
                order_type=signal.order_type,
                quantity=signal.quantity,
                price=execution_price,
                stop_price=signal.stop_loss,
                status=OrderStatus.FILLED,  # Paper trading fills immediately
                filled_quantity=signal.quantity,
                average_price=execution_price,
                commission=order_value * self.commission,
                created_time=datetime.utcnow(),
                updated_time=datetime.utcnow()
            )
            
            # Store order
            self.orders[order.client_order_id] = order
            
            # Update balance
            self.balance -= order_value + order.commission
            
            # Update position
            await self._update_position(signal, execution_price)
            
            # Record trade
            trade_data = {
                'timestamp': order.created_time,
                'symbol': signal.symbol,
                'side': signal.side.value,
                'quantity': signal.quantity,
                'price': execution_price,
                'pnl': 0.0,  # Will be calculated when position is closed
                'order_id': order.order_id,
                'commission': order.commission
            }
            
            self.trade_history.append(trade_data)
            
            # Trigger callbacks
            for callback in self.order_update_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    self.logger.error(f"Error in order update callback: {e}")
            
            for callback in self.trade_execution_callbacks:
                try:
                    callback(trade_data)
                except Exception as e:
                    self.logger.error(f"Error in trade execution callback: {e}")
            
            self.logger.info(f"Paper trade executed: {signal.symbol} {signal.side.value} {signal.quantity} @ {execution_price}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}")
            return None
    
    async def _update_position(self, signal: TradingSignal, execution_price: float) -> None:
        """Update position after trade execution."""
        symbol = signal.symbol
        
        if symbol not in self.positions:
            # Create new position
            position_side = PositionSide.LONG if signal.side == OrderSide.BUY else PositionSide.SHORT
            self.positions[symbol] = Position(
                symbol=symbol,
                position_side=position_side,
                position_amount=signal.quantity,
                entry_price=execution_price,
                mark_price=execution_price,
                unrealized_pnl=0.0,
                percentage=0.0,
                notional=signal.quantity * execution_price,
                isolated=False,
                isolated_margin=0.0,
                leverage=1,
                initial_margin=0.0,
                maint_margin=0.0,
                timestamp=datetime.utcnow()
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if (position.position_side == PositionSide.LONG and signal.side == OrderSide.BUY) or \
               (position.position_side == PositionSide.SHORT and signal.side == OrderSide.SELL):
                # Adding to position
                total_quantity = position.position_amount + signal.quantity
                total_value = (position.position_amount * position.entry_price) + (signal.quantity * execution_price)
                new_entry_price = total_value / total_quantity
                
                position.position_amount = total_quantity
                position.entry_price = new_entry_price
                position.notional = total_quantity * execution_price
            else:
                # Closing or reducing position
                if signal.quantity >= position.position_amount:
                    # Closing position
                    pnl = self._calculate_pnl(position, execution_price)
                    self.balance += pnl
                    del self.positions[symbol]
                else:
                    # Reducing position
                    pnl = self._calculate_pnl(position, execution_price) * (signal.quantity / position.position_amount)
                    self.balance += pnl
                    position.position_amount -= signal.quantity
                    position.notional = position.position_amount * execution_price
        
        # Update equity
        self._update_equity()
        
        # Trigger position update callbacks
        if symbol in self.positions:
            for callback in self.position_update_callbacks:
                try:
                    callback(self.positions[symbol])
                except Exception as e:
                    self.logger.error(f"Error in position update callback: {e}")
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Calculate PnL for position."""
        if position.position_side == PositionSide.LONG:
            return position.position_amount * (current_price - position.entry_price)
        else:  # SHORT
            return position.position_amount * (position.entry_price - current_price)
    
    def _update_equity(self) -> None:
        """Update equity calculation."""
        unrealized_pnl = 0.0
        for position in self.positions.values():
            unrealized_pnl += self._calculate_pnl(position, position.mark_price)
        
        self.equity = self.balance + unrealized_pnl
        
        # Track max equity for drawdown calculation
        if self.equity > self.max_equity:
            self.max_equity = self.equity
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Update market prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].mark_price = price
                self.positions[symbol].unrealized_pnl = self._calculate_pnl(self.positions[symbol], price)
        
        self._update_equity()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'unrealized_pnl': self.equity - self.balance,
            'positions': len(self.positions),
            'total_trades': len(self.trade_history)
        }
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.trade_history.copy()


class BacktestEngine:
    """Backtesting engine for strategy validation."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.paper_executor = PaperTradingExecutor(config, logger)
        
        # Backtest parameters
        self.start_date = datetime.fromisoformat(config.get('start_date', '2024-01-01'))
        self.end_date = datetime.fromisoformat(config.get('end_date', '2024-01-31'))
        self.symbols = config.get('symbols', ['BTCUSDT'])
        
        # Data storage
        self.historical_data: Dict[str, List[Kline]] = {}
        self.processed_data: List[Dict[str, Any]] = []
        
    async def load_historical_data(self) -> None:
        """Load historical data for backtesting."""
        self.logger.info(f"Loading historical data from {self.start_date} to {self.end_date}")
        
        # This would typically load from Binance API or data files
        # For now, we'll create synthetic data
        for symbol in self.symbols:
            self.historical_data[symbol] = self._generate_synthetic_data(symbol)
        
        self.logger.info(f"Loaded historical data for {len(self.symbols)} symbols")
    
    def _generate_synthetic_data(self, symbol: str) -> List[Kline]:
        """Generate synthetic historical data for testing."""
        data = []
        current_time = self.start_date
        base_price = 50000 if 'BTC' in symbol else 3000  # Base price
        
        while current_time <= self.end_date:
            # Generate random price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            base_price *= (1 + price_change)
            
            # Generate OHLC data
            open_price = base_price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price * (1 + np.random.normal(0, 0.005))
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            kline = Kline(
                symbol=symbol,
                open_time=current_time,
                close_time=current_time + timedelta(minutes=1),
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=np.random.uniform(100, 1000),
                quote_volume=close_price * np.random.uniform(100, 1000),
                trades_count=np.random.randint(10, 100),
                taker_buy_base_volume=np.random.uniform(50, 500),
                taker_buy_quote_volume=close_price * np.random.uniform(50, 500),
                interval='1m'
            )
            
            data.append(kline)
            current_time += timedelta(minutes=1)
        
        return data
    
    async def run_backtest(self, strategy, data_processor) -> BacktestResult:
        """Run backtest with given strategy."""
        self.logger.info("Starting backtest...")
        
        # Load historical data
        await self.load_historical_data()
        
        # Process data through strategy
        signals = []
        for symbol in self.symbols:
            symbol_data = self.historical_data[symbol]
            
            # Process each kline
            for i, kline in enumerate(symbol_data):
                # Create market data
                market_data = {
                    'symbol': symbol,
                    'timestamp': kline.close_time,
                    'indicators': {},
                    'signals': {}
                }
                
                # Process through data processor (simplified)
                await data_processor.process_market_data(market_data)
                
                # Get strategy signal
                signal = await strategy.analyze(market_data)
                if signal:
                    signals.append((kline.close_time, signal, kline.close_price))
        
        # Execute signals
        for timestamp, signal, price in signals:
            await self.paper_executor.execute_signal(signal, price)
            self.paper_executor.update_market_prices({signal.symbol: price})
        
        # Calculate results
        result = self._calculate_backtest_results()
        
        self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        
        return result
    
    def _calculate_backtest_results(self) -> BacktestResult:
        """Calculate backtest results."""
        trade_history = self.paper_executor.get_trade_history()
        account_info = self.paper_executor.get_account_info()
        
        # Basic metrics
        initial_balance = self.paper_executor.initial_balance
        final_balance = account_info['equity']
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Trade metrics
        total_trades = len(trade_history)
        winning_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trade_history if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        average_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        average_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown calculation
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio (simplified)
        daily_returns = self._calculate_daily_returns()
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            total_pnl=final_balance - initial_balance,
            daily_returns=daily_returns,
            equity_curve=self.paper_executor.daily_equity,
            trade_history=trade_history
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.paper_executor.daily_equity:
            return 0.0
        
        peak = self.paper_executor.daily_equity[0]
        max_dd = 0.0
        
        for equity in self.paper_executor.daily_equity:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns."""
        if len(self.paper_executor.daily_equity) < 2:
            return [0.0]
        
        returns = []
        for i in range(1, len(self.paper_executor.daily_equity)):
            daily_return = (self.paper_executor.daily_equity[i] - self.paper_executor.daily_equity[i-1]) / self.paper_executor.daily_equity[i-1]
            returns.append(daily_return)
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def save_results(self, result: BacktestResult, filename: str) -> None:
        """Save backtest results to file."""
        try:
            # Convert to JSON-serializable format
            data = {
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_balance': result.initial_balance,
                'final_balance': result.final_balance,
                'total_return': result.total_return,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'average_win': result.average_win,
                'average_loss': result.average_loss,
                'total_pnl': result.total_pnl,
                'daily_returns': result.daily_returns,
                'equity_curve': result.equity_curve,
                'trade_history': result.trade_history
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")