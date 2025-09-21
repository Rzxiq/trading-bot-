"""
Comprehensive risk management system for trading operations.
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math

from .data_models import TradingSignal, OrderSide, OrderType, Position, Order
from .logger import TradingLogger


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RiskEvent:
    """Risk event data structure."""
    event_type: str
    symbol: str
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    action_required: bool = False


class PositionSizer:
    """Position sizing calculator."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_open_positions = config.get('max_open_positions', 3)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        self.fixed_size = config.get('fixed_size', 0.01)
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float, 
                              current_positions: Dict[str, Position], 
                              volatility: float = 0.02) -> float:
        """Calculate position size based on risk parameters."""
        symbol = signal.symbol
        
        # Check if we already have a position in this symbol
        if symbol in current_positions:
            return 0.0
        
        # Check max open positions
        if len(current_positions) >= self.max_open_positions:
            return 0.0
        
        # Calculate base position size
        base_size = self._calculate_base_size(signal, account_balance, volatility)
        
        # Apply position size limits
        max_size = account_balance * self.max_position_size
        position_size = min(base_size, max_size)
        
        # Ensure minimum viable position
        min_size = account_balance * 0.001  # 0.1% minimum
        if position_size < min_size:
            return 0.0
        
        return position_size
    
    def _calculate_base_size(self, signal: TradingSignal, account_balance: float, 
                           volatility: float) -> float:
        """Calculate base position size using different methods."""
        method = self.config.get('position_sizing_method', 'fixed')
        
        if method == 'fixed':
            return account_balance * self.fixed_size
        
        elif method == 'volatility_adjusted':
            # Adjust position size based on volatility
            base_size = account_balance * self.max_position_size
            volatility_adjustment = 0.02 / max(volatility, 0.001)  # Target 2% volatility
            return base_size * min(volatility_adjustment, 2.0)  # Cap at 2x
        
        elif method == 'kelly':
            # Kelly Criterion (simplified)
            win_rate = self.config.get('win_rate', 0.55)
            avg_win = self.config.get('avg_win', 0.02)
            avg_loss = self.config.get('avg_loss', 0.01)
            
            if avg_loss == 0:
                return account_balance * self.fixed_size
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction))
            
            return account_balance * kelly_fraction
        
        else:
            return account_balance * self.fixed_size


class StopLossManager:
    """Stop loss and take profit management."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.04)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.01)
        self.use_atr_stop = config.get('use_atr_stop', True)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
    
    def calculate_stop_loss(self, signal: TradingSignal, atr: Optional[float] = None) -> float:
        """Calculate stop loss level."""
        if signal.stop_loss:
            return signal.stop_loss
        
        entry_price = signal.price or 0
        if entry_price == 0:
            return 0
        
        if self.use_atr_stop and atr:
            # Use ATR-based stop loss
            stop_distance = atr * self.atr_multiplier
        else:
            # Use percentage-based stop loss
            stop_distance = entry_price * self.stop_loss_pct
        
        if signal.side == OrderSide.BUY:
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance
    
    def calculate_take_profit(self, signal: TradingSignal, atr: Optional[float] = None) -> float:
        """Calculate take profit level."""
        if signal.take_profit:
            return signal.take_profit
        
        entry_price = signal.price or 0
        if entry_price == 0:
            return 0
        
        if self.use_atr_stop and atr:
            # Use ATR-based take profit
            profit_distance = atr * self.atr_multiplier * 2  # 2:1 risk-reward
        else:
            # Use percentage-based take profit
            profit_distance = entry_price * self.take_profit_pct
        
        if signal.side == OrderSide.BUY:
            return entry_price + profit_distance
        else:  # SELL
            return entry_price - profit_distance
    
    def update_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """Update trailing stop loss."""
        if not hasattr(position, 'trailing_stop') or position.trailing_stop is None:
            return None
        
        if position.position_side.value == 'LONG':
            # For long positions, trail upward
            new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
            if new_trailing_stop > position.trailing_stop:
                return new_trailing_stop
        else:  # SHORT
            # For short positions, trail downward
            new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
            if new_trailing_stop < position.trailing_stop:
                return new_trailing_stop
        
        return position.trailing_stop


class CircuitBreaker:
    """Circuit breaker for risk management."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.volatility_threshold = config.get('volatility_threshold', 0.05)
        self.latency_threshold = config.get('latency_threshold', 1000)  # ms
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        
        # State tracking
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.last_reset_date = datetime.now().date()
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.latency_history: List[float] = []
    
    def check_circuit_breaker(self, current_equity: float, latency: float = 0) -> RiskEvent:
        """Check if circuit breaker should be triggered."""
        current_date = datetime.now().date()
        
        # Reset daily tracking if new day
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
        
        # Update max equity
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        # Calculate current drawdown
        current_drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return RiskEvent(
                event_type='daily_loss_limit',
                symbol='ALL',
                risk_level=RiskLevel.CRITICAL,
                message=f'Daily loss limit exceeded: {self.daily_pnl:.2%}',
                value=abs(self.daily_pnl),
                threshold=self.max_daily_loss,
                timestamp=datetime.utcnow(),
                action_required=True
            )
        
        # Check drawdown limit
        if current_drawdown > self.max_drawdown:
            return RiskEvent(
                event_type='drawdown_limit',
                symbol='ALL',
                risk_level=RiskLevel.CRITICAL,
                message=f'Maximum drawdown exceeded: {current_drawdown:.2%}',
                value=current_drawdown,
                threshold=self.max_drawdown,
                timestamp=datetime.utcnow(),
                action_required=True
            )
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return RiskEvent(
                event_type='consecutive_losses',
                symbol='ALL',
                risk_level=RiskLevel.HIGH,
                message=f'Too many consecutive losses: {self.consecutive_losses}',
                value=self.consecutive_losses,
                threshold=self.max_consecutive_losses,
                timestamp=datetime.utcnow(),
                action_required=True
            )
        
        # Check latency
        if latency > self.latency_threshold:
            return RiskEvent(
                event_type='high_latency',
                symbol='ALL',
                risk_level=RiskLevel.MEDIUM,
                message=f'High latency detected: {latency:.0f}ms',
                value=latency,
                threshold=self.latency_threshold,
                timestamp=datetime.utcnow(),
                action_required=False
            )
        
        return None
    
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Update circuit breaker with trade result."""
        pnl = trade_result.get('pnl', 0)
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Add to trade history
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'pnl': pnl,
            'symbol': trade_result.get('symbol', ''),
            'side': trade_result.get('side', '')
        })
        
        # Keep only recent trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def update_latency(self, latency: float) -> None:
        """Update latency tracking."""
        self.latency_history.append(latency)
        
        # Keep only recent latency measurements
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]
    
    def get_average_latency(self) -> float:
        """Get average latency."""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state."""
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        self.logger.info("Circuit breaker reset")


class RiskManager:
    """Main risk management system."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.position_sizer = PositionSizer(config, logger)
        self.stop_loss_manager = StopLossManager(config, logger)
        self.circuit_breaker = CircuitBreaker(config, logger)
        
        # Risk event callbacks
        self.risk_event_callbacks: List[Callable[[RiskEvent], None]] = []
        
        # Current state
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.account_balance = 0.0
        self.equity = 0.0
        
        # Risk limits
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_open_positions = config.get('max_open_positions', 3)
        self.position_timeout = config.get('position_timeout', 3600)  # seconds
    
    def add_risk_event_callback(self, callback: Callable[[RiskEvent], None]) -> None:
        """Add callback for risk events."""
        self.risk_event_callbacks.append(callback)
    
    async def validate_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> bool:
        """Validate trading signal against risk parameters."""
        try:
            # Check circuit breaker
            risk_event = self.circuit_breaker.check_circuit_breaker(self.equity)
            if risk_event and risk_event.action_required:
                await self._handle_risk_event(risk_event)
                return False
            
            # Check if we already have a position in this symbol
            if signal.symbol in self.positions:
                self.logger.warning(f"Position already exists for {signal.symbol}")
                return False
            
            # Check max open positions
            if len(self.positions) >= self.max_open_positions:
                self.logger.warning(f"Maximum open positions reached: {len(self.positions)}")
                return False
            
            # Calculate position size
            volatility = market_data.get('indicators', {}).get('volatility', 0.02)
            position_size = self.position_sizer.calculate_position_size(
                signal, self.account_balance, self.positions, volatility
            )
            
            if position_size <= 0:
                self.logger.warning(f"Position size too small for {signal.symbol}")
                return False
            
            # Update signal with calculated values
            signal.quantity = position_size
            
            # Calculate stop loss and take profit
            atr = market_data.get('indicators', {}).get('atr')
            signal.stop_loss = self.stop_loss_manager.calculate_stop_loss(signal, atr)
            signal.take_profit = self.stop_loss_manager.calculate_take_profit(signal, atr)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    async def monitor_positions(self, current_prices: Dict[str, float]) -> List[RiskEvent]:
        """Monitor existing positions for risk events."""
        risk_events = []
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.mark_price)
            
            # Check position timeout
            position_age = (datetime.utcnow() - position.timestamp).total_seconds()
            if position_age > self.position_timeout:
                risk_events.append(RiskEvent(
                    event_type='position_timeout',
                    symbol=symbol,
                    risk_level=RiskLevel.MEDIUM,
                    message=f'Position timeout: {position_age:.0f}s',
                    value=position_age,
                    threshold=self.position_timeout,
                    timestamp=datetime.utcnow(),
                    action_required=True
                ))
            
            # Check stop loss
            if position.position_side.value == 'LONG' and current_price <= position.entry_price * 0.98:
                risk_events.append(RiskEvent(
                    event_type='stop_loss_triggered',
                    symbol=symbol,
                    risk_level=RiskLevel.HIGH,
                    message=f'Stop loss triggered: {current_price:.2f}',
                    value=current_price,
                    threshold=position.entry_price * 0.98,
                    timestamp=datetime.utcnow(),
                    action_required=True
                ))
            elif position.position_side.value == 'SHORT' and current_price >= position.entry_price * 1.02:
                risk_events.append(RiskEvent(
                    event_type='stop_loss_triggered',
                    symbol=symbol,
                    risk_level=RiskLevel.HIGH,
                    message=f'Stop loss triggered: {current_price:.2f}',
                    value=current_price,
                    threshold=position.entry_price * 1.02,
                    timestamp=datetime.utcnow(),
                    action_required=True
                ))
            
            # Check take profit
            if position.position_side.value == 'LONG' and current_price >= position.entry_price * 1.04:
                risk_events.append(RiskEvent(
                    event_type='take_profit_triggered',
                    symbol=symbol,
                    risk_level=RiskLevel.LOW,
                    message=f'Take profit triggered: {current_price:.2f}',
                    value=current_price,
                    threshold=position.entry_price * 1.04,
                    timestamp=datetime.utcnow(),
                    action_required=True
                ))
            elif position.position_side.value == 'SHORT' and current_price <= position.entry_price * 0.96:
                risk_events.append(RiskEvent(
                    event_type='take_profit_triggered',
                    symbol=symbol,
                    risk_level=RiskLevel.LOW,
                    message=f'Take profit triggered: {current_price:.2f}',
                    value=current_price,
                    threshold=position.entry_price * 0.96,
                    timestamp=datetime.utcnow(),
                    action_required=True
                ))
        
        return risk_events
    
    async def _handle_risk_event(self, risk_event: RiskEvent) -> None:
        """Handle risk event."""
        self.logger.warning(f"Risk event: {risk_event.event_type} - {risk_event.message}")
        
        # Trigger callbacks
        for callback in self.risk_event_callbacks:
            try:
                callback(risk_event)
            except Exception as e:
                self.logger.error(f"Error in risk event callback: {e}")
        
        # Take action based on risk level
        if risk_event.action_required:
            if risk_event.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                await self._emergency_stop()
    
    async def _emergency_stop(self) -> None:
        """Emergency stop - close all positions."""
        self.logger.critical("EMERGENCY STOP - Closing all positions")
        
        # This would trigger order execution to close all positions
        # Implementation depends on the execution module
        
        # Reset circuit breaker
        self.circuit_breaker.reset_circuit_breaker()
    
    def update_position(self, position: Position) -> None:
        """Update position data."""
        self.positions[position.symbol] = position
    
    def remove_position(self, symbol: str) -> None:
        """Remove position."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def update_account_balance(self, balance: float) -> None:
        """Update account balance."""
        self.account_balance = balance
    
    def update_equity(self, equity: float) -> None:
        """Update equity."""
        self.equity = equity
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        total_exposure = sum(abs(pos.notional) for pos in self.positions.values())
        
        return {
            'account_balance': self.account_balance,
            'equity': self.equity,
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / self.account_balance if self.account_balance > 0 else 0,
            'open_positions': len(self.positions),
            'max_positions': self.max_open_positions,
            'daily_pnl': self.circuit_breaker.daily_pnl,
            'consecutive_losses': self.circuit_breaker.consecutive_losses,
            'average_latency': self.circuit_breaker.get_average_latency(),
            'circuit_breaker_active': self.circuit_breaker.circuit_breaker_active
        }