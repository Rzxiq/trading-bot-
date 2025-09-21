"""
Data processing pipeline for real-time market data analysis.
"""
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import numpy as np
from collections import defaultdict

from .data_models import MarketData, Kline, OrderBook, Trade, MarkPrice, Liquidation
from .indicators import TechnicalIndicators, IndicatorBuffer
from .logger import TradingLogger


class DataProcessor:
    """Real-time data processing pipeline."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.indicator_buffer = IndicatorBuffer(max_size=1000)
        
        # Data buffers
        self.klines: Dict[str, List[Kline]] = defaultdict(list)
        self.order_books: Dict[str, OrderBook] = {}
        self.mark_prices: Dict[str, MarkPrice] = {}
        self.trades: Dict[str, List[Trade]] = defaultdict(list)
        
        # Processing callbacks
        self.processed_data_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.signal_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Strategy parameters
        self.strategy_config = config.get('strategy', {})
        self.rsi_period = self.strategy_config.get('rsi_period', 14)
        self.bb_period = self.strategy_config.get('bb_period', 20)
        self.bb_std = self.strategy_config.get('bb_std', 2)
        self.order_book_levels = self.strategy_config.get('order_book_levels', 20)
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.max_processing_time = 0.0
        
    def add_processed_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for processed data."""
        self.processed_data_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for trading signals."""
        self.signal_callbacks.append(callback)
    
    async def process_market_data(self, market_data: MarketData) -> None:
        """Process incoming market data."""
        start_time = datetime.utcnow()
        
        try:
            symbol = market_data.symbol
            processed_data = {
                'symbol': symbol,
                'timestamp': market_data.timestamp,
                'indicators': {},
                'signals': {},
                'raw_data': market_data.to_dict()
            }
            
            # Process different types of data
            if market_data.kline:
                await self._process_kline(market_data.kline, processed_data)
            
            if market_data.order_book:
                await self._process_order_book(market_data.order_book, processed_data)
            
            if market_data.trade:
                await self._process_trade(market_data.trade, processed_data)
            
            if market_data.mark_price:
                await self._process_mark_price(market_data.mark_price, processed_data)
            
            if market_data.liquidation:
                await self._process_liquidation(market_data.liquidation, processed_data)
            
            # Calculate combined indicators
            await self._calculate_combined_indicators(symbol, processed_data)
            
            # Generate trading signals
            await self._generate_signals(symbol, processed_data)
            
            # Track processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)
            self.max_processing_time = max(self.max_processing_time, processing_time)
            
            # Keep only recent processing times
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
            
            # Trigger callbacks
            for callback in self.processed_data_callbacks:
                try:
                    callback(processed_data)
                except Exception as e:
                    self.logger.error(f"Error in processed data callback: {e}")
            
            # Log performance if processing time is high
            if processing_time > 100:  # 100ms threshold
                self.logger.warning(f"High processing time: {processing_time:.2f}ms for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
    
    async def _process_kline(self, kline: Kline, processed_data: Dict[str, Any]) -> None:
        """Process kline data and calculate indicators."""
        symbol = kline.symbol
        
        # Update kline buffer
        self.klines[symbol].append(kline)
        if len(self.klines[symbol]) > 1000:
            self.klines[symbol] = self.klines[symbol][-1000:]
        
        # Calculate technical indicators
        kline_list = self.klines[symbol]
        prices = [k.close_price for k in kline_list]
        
        # RSI
        rsi = TechnicalIndicators.rsi(prices, self.rsi_period)
        if rsi is not None:
            self.indicator_buffer.add_indicator(symbol, 'rsi', rsi, kline.close_time.timestamp())
            processed_data['indicators']['rsi'] = rsi
        
        # Bollinger Bands
        bb = TechnicalIndicators.bollinger_bands(prices, self.bb_period, self.bb_std)
        if bb is not None:
            upper, middle, lower = bb
            self.indicator_buffer.add_indicator(symbol, 'bb_upper', upper, kline.close_time.timestamp())
            self.indicator_buffer.add_indicator(symbol, 'bb_middle', middle, kline.close_time.timestamp())
            self.indicator_buffer.add_indicator(symbol, 'bb_lower', lower, kline.close_time.timestamp())
            processed_data['indicators']['bb_upper'] = upper
            processed_data['indicators']['bb_middle'] = middle
            processed_data['indicators']['bb_lower'] = lower
        
        # MACD
        macd = TechnicalIndicators.macd(prices)
        if macd is not None:
            macd_line, signal_line, histogram = macd
            self.indicator_buffer.add_indicator(symbol, 'macd', macd_line, kline.close_time.timestamp())
            self.indicator_buffer.add_indicator(symbol, 'macd_signal', signal_line, kline.close_time.timestamp())
            self.indicator_buffer.add_indicator(symbol, 'macd_histogram', histogram, kline.close_time.timestamp())
            processed_data['indicators']['macd'] = macd_line
            processed_data['indicators']['macd_signal'] = signal_line
            processed_data['indicators']['macd_histogram'] = histogram
        
        # Stochastic
        stoch = TechnicalIndicators.stochastic(kline_list)
        if stoch is not None:
            k_percent, d_percent = stoch
            self.indicator_buffer.add_indicator(symbol, 'stoch_k', k_percent, kline.close_time.timestamp())
            self.indicator_buffer.add_indicator(symbol, 'stoch_d', d_percent, kline.close_time.timestamp())
            processed_data['indicators']['stoch_k'] = k_percent
            processed_data['indicators']['stoch_d'] = d_percent
        
        # Williams %R
        williams_r = TechnicalIndicators.williams_r(kline_list)
        if williams_r is not None:
            self.indicator_buffer.add_indicator(symbol, 'williams_r', williams_r, kline.close_time.timestamp())
            processed_data['indicators']['williams_r'] = williams_r
        
        # ATR
        atr = TechnicalIndicators.atr(kline_list)
        if atr is not None:
            self.indicator_buffer.add_indicator(symbol, 'atr', atr, kline.close_time.timestamp())
            processed_data['indicators']['atr'] = atr
        
        # ADX
        adx = TechnicalIndicators.adx(kline_list)
        if adx is not None:
            self.indicator_buffer.add_indicator(symbol, 'adx', adx, kline.close_time.timestamp())
            processed_data['indicators']['adx'] = adx
        
        # Volume profile
        volume_profile = TechnicalIndicators.volume_profile(kline_list)
        if volume_profile is not None:
            processed_data['indicators']['volume_profile'] = volume_profile
        
        # Volatility
        volatility = TechnicalIndicators.volatility(kline_list)
        if volatility is not None:
            self.indicator_buffer.add_indicator(symbol, 'volatility', volatility, kline.close_time.timestamp())
            processed_data['indicators']['volatility'] = volatility
        
        # Price action analysis
        processed_data['indicators']['price_action'] = {
            'is_bullish': kline.is_bullish(),
            'is_bearish': kline.is_bearish(),
            'body_size': kline.get_body_size(),
            'upper_shadow': kline.get_upper_shadow(),
            'lower_shadow': kline.get_lower_shadow(),
            'close_price': kline.close_price,
            'volume': kline.volume
        }
    
    async def _process_order_book(self, order_book: OrderBook, processed_data: Dict[str, Any]) -> None:
        """Process order book data."""
        symbol = order_book.symbol
        
        # Update order book buffer
        self.order_books[symbol] = order_book
        
        # Calculate order book indicators
        imbalance = order_book.get_imbalance(self.order_book_levels)
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()
        
        pressure = TechnicalIndicators.order_book_pressure(order_book, self.order_book_levels)
        
        processed_data['indicators']['order_book'] = {
            'imbalance': imbalance,
            'spread': spread,
            'mid_price': mid_price,
            'bid_pressure': pressure['bid_pressure'],
            'ask_pressure': pressure['ask_pressure'],
            'net_pressure': pressure['net_pressure'],
            'best_bid': order_book.get_best_bid().price if order_book.get_best_bid() else None,
            'best_ask': order_book.get_best_ask().price if order_book.get_best_ask() else None
        }
    
    async def _process_trade(self, trade: Trade, processed_data: Dict[str, Any]) -> None:
        """Process trade data."""
        symbol = trade.symbol
        
        # Update trade buffer
        self.trades[symbol].append(trade)
        if len(self.trades[symbol]) > 1000:
            self.trades[symbol] = self.trades[symbol][-1000:]
        
        # Calculate trade-based indicators
        recent_trades = self.trades[symbol][-100:]  # Last 100 trades
        
        if recent_trades:
            buy_trades = [t for t in recent_trades if t.is_buy()]
            sell_trades = [t for t in recent_trades if not t.is_buy()]
            
            buy_volume = sum(t.quantity for t in buy_trades)
            sell_volume = sum(t.quantity for t in sell_trades)
            total_volume = buy_volume + sell_volume
            
            volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            processed_data['indicators']['trade_flow'] = {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'volume_imbalance': volume_imbalance,
                'buy_trades_count': len(buy_trades),
                'sell_trades_count': len(sell_trades),
                'average_trade_size': total_volume / len(recent_trades) if recent_trades else 0
            }
    
    async def _process_mark_price(self, mark_price: MarkPrice, processed_data: Dict[str, Any]) -> None:
        """Process mark price data."""
        symbol = mark_price.symbol
        
        # Update mark price buffer
        self.mark_prices[symbol] = mark_price
        
        processed_data['indicators']['funding'] = {
            'mark_price': mark_price.mark_price,
            'index_price': mark_price.index_price,
            'funding_rate': mark_price.funding_rate,
            'next_funding_time': mark_price.next_funding_time,
            'price_deviation': abs(mark_price.mark_price - mark_price.index_price) / mark_price.index_price * 100
        }
    
    async def _process_liquidation(self, liquidation: Liquidation, processed_data: Dict[str, Any]) -> None:
        """Process liquidation data."""
        symbol = liquidation.symbol
        
        processed_data['indicators']['liquidation'] = {
            'side': liquidation.side,
            'quantity': liquidation.quantity,
            'price': liquidation.price,
            'order_type': liquidation.order_type
        }
    
    async def _calculate_combined_indicators(self, symbol: str, processed_data: Dict[str, Any]) -> None:
        """Calculate combined indicators using multiple data sources."""
        indicators = processed_data['indicators']
        
        # Market sentiment score
        sentiment_score = 0.0
        sentiment_factors = []
        
        # RSI sentiment
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70:
                sentiment_factors.append(-1.0)  # Overbought
            elif rsi < 30:
                sentiment_factors.append(1.0)   # Oversold
            else:
                sentiment_factors.append(0.0)   # Neutral
        
        # Bollinger Bands sentiment
        if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_upper = indicators['bb_upper']
            bb_middle = indicators['bb_middle']
            bb_lower = indicators['bb_lower']
            close_price = indicators.get('price_action', {}).get('close_price', 0)
            
            if close_price > 0:
                if close_price > bb_upper:
                    sentiment_factors.append(-0.5)  # Above upper band
                elif close_price < bb_lower:
                    sentiment_factors.append(0.5)   # Below lower band
                else:
                    sentiment_factors.append(0.0)   # Within bands
        
        # Order book sentiment
        if 'order_book' in indicators:
            imbalance = indicators['order_book'].get('imbalance', 0)
            sentiment_factors.append(imbalance)
        
        # Trade flow sentiment
        if 'trade_flow' in indicators:
            volume_imbalance = indicators['trade_flow'].get('volume_imbalance', 0)
            sentiment_factors.append(volume_imbalance * 0.5)
        
        # Calculate overall sentiment
        if sentiment_factors:
            sentiment_score = sum(sentiment_factors) / len(sentiment_factors)
        
        processed_data['indicators']['sentiment_score'] = sentiment_score
        
        # Market strength
        strength_factors = []
        
        if 'adx' in indicators:
            adx = indicators['adx']
            strength_factors.append(min(adx / 50, 1.0))  # Normalize ADX
        
        if 'volatility' in indicators:
            volatility = indicators['volatility']
            strength_factors.append(min(volatility / 5.0, 1.0))  # Normalize volatility
        
        if strength_factors:
            market_strength = sum(strength_factors) / len(strength_factors)
            processed_data['indicators']['market_strength'] = market_strength
    
    async def _generate_signals(self, symbol: str, processed_data: Dict[str, Any]) -> None:
        """Generate trading signals based on processed data."""
        indicators = processed_data['indicators']
        signals = {}
        
        # RSI signals
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                signals['rsi_oversold'] = True
            elif rsi > 70:
                signals['rsi_overbought'] = True
            else:
                signals['rsi_oversold'] = False
                signals['rsi_overbought'] = False
        
        # Bollinger Bands signals
        if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_upper = indicators['bb_upper']
            bb_middle = indicators['bb_middle']
            bb_lower = indicators['bb_lower']
            close_price = indicators.get('price_action', {}).get('close_price', 0)
            
            if close_price > bb_upper:
                signals['bb_breakout_upper'] = True
            elif close_price < bb_lower:
                signals['bb_breakout_lower'] = True
            else:
                signals['bb_breakout_upper'] = False
                signals['bb_breakout_lower'] = False
        
        # Order book signals
        if 'order_book' in indicators:
            imbalance = indicators['order_book'].get('imbalance', 0)
            if imbalance > 0.3:
                signals['order_book_bullish'] = True
            elif imbalance < -0.3:
                signals['order_book_bearish'] = True
            else:
                signals['order_book_bullish'] = False
                signals['order_book_bearish'] = False
        
        # Trade flow signals
        if 'trade_flow' in indicators:
            volume_imbalance = indicators['trade_flow'].get('volume_imbalance', 0)
            if volume_imbalance > 0.2:
                signals['trade_flow_bullish'] = True
            elif volume_imbalance < -0.2:
                signals['trade_flow_bearish'] = True
            else:
                signals['trade_flow_bullish'] = False
                signals['trade_flow_bearish'] = False
        
        # Combined signal strength
        bullish_signals = sum(1 for signal in signals.values() if signal and 'bullish' in signal or 'oversold' in signal or 'breakout_lower' in signal)
        bearish_signals = sum(1 for signal in signals.values() if signal and 'bearish' in signal or 'overbought' in signal or 'breakout_upper' in signal)
        
        signals['bullish_strength'] = bullish_signals
        signals['bearish_strength'] = bearish_signals
        signals['signal_strength'] = abs(bullish_signals - bearish_signals)
        
        processed_data['signals'] = signals
        
        # Trigger signal callbacks
        for callback in self.signal_callbacks:
            try:
                callback(processed_data)
            except Exception as e:
                self.logger.error(f"Error in signal callback: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        if not self.processing_times:
            return {'average_processing_time': 0, 'max_processing_time': 0, 'total_processed': 0}
        
        return {
            'average_processing_time': sum(self.processing_times) / len(self.processing_times),
            'max_processing_time': self.max_processing_time,
            'total_processed': len(self.processing_times),
            'recent_avg': sum(self.processing_times[-100:]) / min(100, len(self.processing_times))
        }
    
    def get_latest_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get latest indicators for a symbol."""
        return self.indicator_buffer.get_all_indicators(symbol)