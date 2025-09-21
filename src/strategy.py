"""
Modular strategy engine for trading algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .data_models import TradingSignal, OrderSide, OrderType, MarketData
from .logger import TradingLogger


class SignalStrength(Enum):
    """Signal strength enumeration."""
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class StrategySignal:
    """Strategy signal with metadata."""
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0
    strength: SignalStrength = SignalStrength.WEAK
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_trading_signal(self) -> TradingSignal:
        """Convert to TradingSignal."""
        return TradingSignal(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
            order_type=self.order_type,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            reason=self.reason,
            confidence=self.confidence,
            timestamp=self.timestamp
        )


class StrategyBase(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.name = self.__class__.__name__
        self.is_active = True
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.signal_callbacks: List[Callable[[StrategySignal], None]] = []
        
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze market data and generate trading signal."""
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators for this strategy."""
        pass
    
    def add_signal_callback(self, callback: Callable[[StrategySignal], None]) -> None:
        """Add callback for strategy signals."""
        self.signal_callbacks.append(callback)
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> None:
        """Process market data and generate signals."""
        if not self.is_active:
            return
        
        try:
            signal = await self.analyze(market_data)
            if signal:
                await self._emit_signal(signal)
        except Exception as e:
            self.logger.error(f"Error in strategy {self.name}: {e}")
    
    async def _emit_signal(self, signal: StrategySignal) -> None:
        """Emit strategy signal to callbacks."""
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"Error in strategy signal callback: {e}")
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """Update position data."""
        self.positions[symbol] = position_data
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if strategy has position in symbol."""
        return symbol in self.positions and self.positions[symbol].get('quantity', 0) != 0
    
    def activate(self) -> None:
        """Activate strategy."""
        self.is_active = True
        self.logger.info(f"Strategy {self.name} activated")
    
    def deactivate(self) -> None:
        """Deactivate strategy."""
        self.is_active = False
        self.logger.info(f"Strategy {self.name} deactivated")


class RSIBollingerStrategy(StrategyBase):
    """RSI + Bollinger Bands mean reversion strategy."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        super().__init__(config, logger)
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.order_book_levels = config.get('order_book_levels', 20)
        self.min_volume = config.get('min_volume', 1000000)
        
        # Position sizing
        self.position_size = config.get('position_size', 0.1)  # 10% of balance
        self.max_positions = config.get('max_positions', 3)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4%
        
        # Signal filtering
        self.min_confidence = config.get('min_confidence', 0.6)
        self.require_order_book_confirmation = config.get('require_order_book_confirmation', True)
        
        self.logger.info(f"Initialized {self.name} with RSI({self.rsi_period}) and BB({self.bb_period}, {self.bb_std})")
    
    def get_required_indicators(self) -> List[str]:
        """Get required indicators."""
        return ['rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'order_book', 'volume_profile']
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze market data and generate trading signal."""
        symbol = market_data['symbol']
        indicators = market_data.get('indicators', {})
        signals = market_data.get('signals', {})
        
        # Check if we have required indicators
        if not all(indicator in indicators for indicator in self.get_required_indicators()):
            return None
        
        # Check volume requirement
        volume_profile = indicators.get('volume_profile', {})
        if volume_profile.get('total_volume', 0) < self.min_volume:
            return None
        
        # Get current price
        price_action = indicators.get('price_action', {})
        current_price = price_action.get('close_price', 0)
        if current_price == 0:
            return None
        
        # Check if we already have a position
        if self.has_position(symbol):
            return None
        
        # Check if we have reached max positions
        if len(self.positions) >= self.max_positions:
            return None
        
        # Analyze RSI
        rsi = indicators.get('rsi', 50)
        rsi_signal = self._analyze_rsi(rsi)
        
        # Analyze Bollinger Bands
        bb_signal = self._analyze_bollinger_bands(indicators, current_price)
        
        # Analyze order book
        order_book_signal = self._analyze_order_book(indicators)
        
        # Analyze market sentiment
        sentiment_signal = self._analyze_sentiment(indicators, signals)
        
        # Combine signals
        combined_signal = self._combine_signals(
            rsi_signal, bb_signal, order_book_signal, sentiment_signal
        )
        
        if combined_signal:
            # Calculate position size
            quantity = self._calculate_position_size(symbol, current_price)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(
                current_price, combined_signal['side']
            )
            
            # Create strategy signal
            signal = StrategySignal(
                symbol=symbol,
                side=combined_signal['side'],
                quantity=quantity,
                price=current_price,
                order_type=OrderType.LIMIT,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=combined_signal['reason'],
                confidence=combined_signal['confidence'],
                strength=combined_signal['strength'],
                timestamp=datetime.utcnow()
            )
            
            return signal
        
        return None
    
    def _analyze_rsi(self, rsi: float) -> Optional[Dict[str, Any]]:
        """Analyze RSI indicator."""
        if rsi < self.rsi_oversold:
            return {
                'side': OrderSide.BUY,
                'strength': self._calculate_rsi_strength(rsi, self.rsi_oversold),
                'reason': f'RSI oversold: {rsi:.2f}',
                'confidence': min(1.0, (self.rsi_oversold - rsi) / 10)
            }
        elif rsi > self.rsi_overbought:
            return {
                'side': OrderSide.SELL,
                'strength': self._calculate_rsi_strength(rsi, self.rsi_overbought),
                'reason': f'RSI overbought: {rsi:.2f}',
                'confidence': min(1.0, (rsi - self.rsi_overbought) / 10)
            }
        return None
    
    def _analyze_bollinger_bands(self, indicators: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Analyze Bollinger Bands indicator."""
        bb_upper = indicators.get('bb_upper', 0)
        bb_middle = indicators.get('bb_middle', 0)
        bb_lower = indicators.get('bb_lower', 0)
        
        if bb_upper == 0 or bb_middle == 0 or bb_lower == 0:
            return None
        
        # Check for mean reversion signals
        if current_price <= bb_lower:
            # Price at or below lower band - potential buy signal
            distance_from_lower = (bb_lower - current_price) / bb_lower
            return {
                'side': OrderSide.BUY,
                'strength': self._calculate_bb_strength(distance_from_lower),
                'reason': f'Price below BB lower band: {current_price:.2f} < {bb_lower:.2f}',
                'confidence': min(1.0, distance_from_lower * 10)
            }
        elif current_price >= bb_upper:
            # Price at or above upper band - potential sell signal
            distance_from_upper = (current_price - bb_upper) / bb_upper
            return {
                'side': OrderSide.SELL,
                'strength': self._calculate_bb_strength(distance_from_upper),
                'reason': f'Price above BB upper band: {current_price:.2f} > {bb_upper:.2f}',
                'confidence': min(1.0, distance_from_upper * 10)
            }
        
        return None
    
    def _analyze_order_book(self, indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze order book for confirmation."""
        order_book = indicators.get('order_book', {})
        imbalance = order_book.get('imbalance', 0)
        net_pressure = order_book.get('net_pressure', 0)
        
        if abs(imbalance) > 0.2:  # Significant imbalance
            side = OrderSide.BUY if imbalance > 0 else OrderSide.SELL
            return {
                'side': side,
                'strength': SignalStrength.MEDIUM,
                'reason': f'Order book imbalance: {imbalance:.3f}',
                'confidence': min(1.0, abs(imbalance) * 2)
            }
        
        return None
    
    def _analyze_sentiment(self, indicators: Dict[str, Any], signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market sentiment."""
        sentiment_score = indicators.get('sentiment_score', 0)
        market_strength = indicators.get('market_strength', 0)
        
        # Only consider sentiment if market strength is sufficient
        if market_strength < 0.3:
            return None
        
        if sentiment_score > 0.3:
            return {
                'side': OrderSide.BUY,
                'strength': SignalStrength.MEDIUM,
                'reason': f'Bullish sentiment: {sentiment_score:.3f}',
                'confidence': min(1.0, sentiment_score * 2)
            }
        elif sentiment_score < -0.3:
            return {
                'side': OrderSide.SELL,
                'strength': SignalStrength.MEDIUM,
                'reason': f'Bearish sentiment: {sentiment_score:.3f}',
                'confidence': min(1.0, abs(sentiment_score) * 2)
            }
        
        return None
    
    def _combine_signals(self, rsi_signal: Optional[Dict], bb_signal: Optional[Dict], 
                        order_book_signal: Optional[Dict], sentiment_signal: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Combine multiple signals into final trading decision."""
        signals = [s for s in [rsi_signal, bb_signal, order_book_signal, sentiment_signal] if s is not None]
        
        if not signals:
            return None
        
        # Count signals by side
        buy_signals = [s for s in signals if s['side'] == OrderSide.BUY]
        sell_signals = [s for s in signals if s['side'] == OrderSide.SELL]
        
        # Require at least 2 signals in same direction
        if len(buy_signals) >= 2 and len(buy_signals) > len(sell_signals):
            # Buy signal
            avg_confidence = sum(s['confidence'] for s in buy_signals) / len(buy_signals)
            if avg_confidence >= self.min_confidence:
                return {
                    'side': OrderSide.BUY,
                    'reason': ' + '.join(s['reason'] for s in buy_signals),
                    'confidence': avg_confidence,
                    'strength': max(s['strength'] for s in buy_signals)
                }
        
        elif len(sell_signals) >= 2 and len(sell_signals) > len(buy_signals):
            # Sell signal
            avg_confidence = sum(s['confidence'] for s in sell_signals) / len(sell_signals)
            if avg_confidence >= self.min_confidence:
                return {
                    'side': OrderSide.SELL,
                    'reason': ' + '.join(s['reason'] for s in sell_signals),
                    'confidence': avg_confidence,
                    'strength': max(s['strength'] for s in sell_signals)
                }
        
        return None
    
    def _calculate_rsi_strength(self, rsi: float, threshold: float) -> SignalStrength:
        """Calculate RSI signal strength."""
        deviation = abs(rsi - threshold)
        if deviation > 20:
            return SignalStrength.VERY_STRONG
        elif deviation > 15:
            return SignalStrength.STRONG
        elif deviation > 10:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    def _calculate_bb_strength(self, distance: float) -> SignalStrength:
        """Calculate Bollinger Bands signal strength."""
        if distance > 0.02:  # 2% deviation
            return SignalStrength.VERY_STRONG
        elif distance > 0.01:  # 1% deviation
            return SignalStrength.STRONG
        elif distance > 0.005:  # 0.5% deviation
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management."""
        # This would typically use account balance and risk parameters
        # For now, return a fixed percentage of notional value
        return self.position_size
    
    def _calculate_risk_levels(self, entry_price: float, side: OrderSide) -> tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        if side == OrderSide.BUY:
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        return stop_loss, take_profit


class StrategyManager:
    """Manager for multiple trading strategies."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.strategies: Dict[str, StrategyBase] = {}
        self.signal_callbacks: List[Callable[[StrategySignal], None]] = []
    
    def add_strategy(self, strategy: StrategyBase) -> None:
        """Add strategy to manager."""
        self.strategies[strategy.name] = strategy
        strategy.add_signal_callback(self._handle_strategy_signal)
        self.logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str) -> None:
        """Remove strategy from manager."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy: {strategy_name}")
    
    def add_signal_callback(self, callback: Callable[[StrategySignal], None]) -> None:
        """Add callback for strategy signals."""
        self.signal_callbacks.append(callback)
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> None:
        """Process market data through all strategies."""
        for strategy in self.strategies.values():
            try:
                await strategy.process_market_data(market_data)
            except Exception as e:
                self.logger.error(f"Error processing market data in strategy {strategy.name}: {e}")
    
    async def _handle_strategy_signal(self, signal: StrategySignal) -> None:
        """Handle signal from strategy."""
        self.logger.info(f"Strategy signal: {signal.symbol} {signal.side.value} {signal.quantity} - {signal.reason}")
        
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"Error in strategy signal callback: {e}")
    
    def get_strategy(self, name: str) -> Optional[StrategyBase]:
        """Get strategy by name."""
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, StrategyBase]:
        """Get all strategies."""
        return self.strategies.copy()
    
    def activate_strategy(self, name: str) -> None:
        """Activate strategy."""
        strategy = self.get_strategy(name)
        if strategy:
            strategy.activate()
    
    def deactivate_strategy(self, name: str) -> None:
        """Deactivate strategy."""
        strategy = self.get_strategy(name)
        if strategy:
            strategy.deactivate()
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics for all strategies."""
        stats = {}
        for name, strategy in self.strategies.items():
            stats[name] = {
                'is_active': strategy.is_active,
                'positions': len(strategy.positions),
                'position_symbols': list(strategy.positions.keys())
            }
        return stats