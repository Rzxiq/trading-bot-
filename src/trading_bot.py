"""
Main trading bot orchestrator.
"""
import asyncio
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .config import Config
from .logger import TradingLogger
from .data_feed import BinanceWebSocketFeed
from .data_processor import DataProcessor
from .strategy import StrategyManager, RSIBollingerStrategy
from .risk_manager import RiskManager
from .execution import BinanceFuturesExecutor
from .monitoring import MonitoringDashboard
from .data_models import TradingSignal, StrategySignal


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = Config(config_path)
        
        # Initialize logger
        self.logger = TradingLogger(self.config.get_logging_config())
        
        # Initialize components
        self.data_feed = None
        self.data_processor = None
        self.strategy_manager = None
        self.risk_manager = None
        self.executor = None
        self.monitoring_dashboard = None
        
        # Bot state
        self.is_running = False
        self.is_initialized = False
        self.start_time = None
        
        # Task management
        self.tasks: List[asyncio.Task] = []
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("Trading bot initialized")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> None:
        """Initialize all bot components."""
        try:
            self.logger.info("Initializing trading bot components...")
            
            # Initialize data feed
            websocket_config = self.config.get_websocket_config()
            self.data_feed = BinanceWebSocketFeed(websocket_config, self.logger)
            
            # Initialize data processor
            self.data_processor = DataProcessor(self.config.get_strategy_config(), self.logger)
            
            # Initialize strategy manager
            self.strategy_manager = StrategyManager(self.config.get_strategy_config(), self.logger)
            
            # Add RSI Bollinger strategy
            rsi_bb_strategy = RSIBollingerStrategy(self.config.get_strategy_config(), self.logger)
            self.strategy_manager.add_strategy(rsi_bb_strategy)
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config.get_risk_config(), self.logger)
            
            # Initialize executor
            api_config = self.config.get_api_config()
            self.executor = BinanceFuturesExecutor(api_config, self.logger)
            
            # Initialize monitoring dashboard
            monitoring_config = self.config.get_monitoring_config()
            self.monitoring_dashboard = MonitoringDashboard(monitoring_config, self.logger)
            
            # Setup component connections
            await self._setup_connections()
            
            self.is_initialized = True
            self.logger.info("Trading bot initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    async def _setup_connections(self) -> None:
        """Setup connections between components."""
        # Data feed -> Data processor
        self.data_feed.add_data_callback(self.data_processor.process_market_data)
        
        # Data processor -> Strategy manager
        self.data_processor.add_processed_data_callback(self.strategy_manager.process_market_data)
        
        # Strategy manager -> Risk manager
        self.strategy_manager.add_signal_callback(self._handle_strategy_signal)
        
        # Risk manager -> Executor
        self.risk_manager.add_risk_event_callback(self._handle_risk_event)
        
        # Executor callbacks
        self.executor.add_order_update_callback(self._handle_order_update)
        self.executor.add_position_update_callback(self._handle_position_update)
        self.executor.add_trade_execution_callback(self._handle_trade_execution)
        
        # Monitoring callbacks
        self.monitoring_dashboard.performance_tracker.add_metrics_callback(self._handle_performance_metrics)
        self.monitoring_dashboard.health_monitor.add_health_check_callback(self._handle_health_check)
    
    async def start(self) -> None:
        """Start the trading bot."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            self.logger.warning("Trading bot is already running")
            return
        
        try:
            self.logger.info("Starting trading bot...")
            self.is_running = True
            self.start_time = datetime.utcnow()
            
            # Get trading symbols
            symbols = self.config.get_symbols()
            
            # Connect to WebSocket
            await self.data_feed.connect(symbols)
            
            # Start monitoring dashboard
            components = {
                'data_feed': self.data_feed,
                'data_processor': self.data_processor,
                'executor': self.executor,
                'risk_manager': self.risk_manager
            }
            
            # Start main tasks
            self.tasks = [
                asyncio.create_task(self._data_feed_loop()),
                asyncio.create_task(self._execution_monitoring_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self.monitoring_dashboard.start(components))
            ]
            
            self.logger.info(f"Trading bot started successfully for symbols: {symbols}")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect from WebSocket
        if self.data_feed:
            await self.data_feed.disconnect()
        
        # Stop monitoring
        if self.monitoring_dashboard:
            await self.monitoring_dashboard.stop()
        
        # Log final statistics
        if self.start_time:
            runtime = datetime.utcnow() - self.start_time
            self.logger.info(f"Trading bot stopped. Runtime: {runtime}")
        
        self.logger.info("Trading bot stopped successfully")
    
    async def _data_feed_loop(self) -> None:
        """Main data feed loop."""
        while self.is_running:
            try:
                await self.data_feed.listen()
            except Exception as e:
                self.logger.error(f"Error in data feed loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _execution_monitoring_loop(self) -> None:
        """Monitor order execution."""
        while self.is_running:
            try:
                if self.executor:
                    await self.executor.monitor_orders()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in execution monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _risk_monitoring_loop(self) -> None:
        """Monitor risk management."""
        while self.is_running:
            try:
                if self.risk_manager and self.executor:
                    # Get current prices (simplified - would need real price data)
                    current_prices = {}
                    positions = await self.executor.get_positions()
                    for position in positions:
                        current_prices[position.symbol] = position.mark_price
                    
                    # Check for risk events
                    risk_events = await self.risk_manager.monitor_positions(current_prices)
                    
                    for risk_event in risk_events:
                        await self._handle_risk_event(risk_event)
                
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _handle_strategy_signal(self, signal: StrategySignal) -> None:
        """Handle strategy signal."""
        try:
            self.logger.info(f"Received strategy signal: {signal.symbol} {signal.side.value} {signal.quantity}")
            
            # Convert to trading signal
            trading_signal = signal.to_trading_signal()
            
            # Get current market data for validation
            market_data = {
                'symbol': signal.symbol,
                'indicators': self.data_processor.get_latest_indicators(signal.symbol),
                'timestamp': datetime.utcnow()
            }
            
            # Validate signal with risk manager
            if await self.risk_manager.validate_signal(trading_signal, market_data):
                # Execute signal
                order = await self.executor.execute_signal(trading_signal)
                if order:
                    self.logger.info(f"Order placed: {order.order_id}")
                else:
                    self.logger.error(f"Failed to place order for {signal.symbol}")
            else:
                self.logger.warning(f"Signal rejected by risk manager: {signal.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error handling strategy signal: {e}")
    
    async def _handle_risk_event(self, risk_event) -> None:
        """Handle risk event."""
        try:
            self.logger.warning(f"Risk event: {risk_event.event_type} - {risk_event.message}")
            
            if risk_event.action_required:
                if risk_event.event_type in ['daily_loss_limit', 'drawdown_limit', 'consecutive_losses']:
                    # Emergency stop - close all positions
                    await self._emergency_stop()
                elif risk_event.event_type in ['stop_loss_triggered', 'take_profit_triggered']:
                    # Close specific position
                    await self._close_position(risk_event.symbol)
                    
        except Exception as e:
            self.logger.error(f"Error handling risk event: {e}")
    
    async def _handle_order_update(self, order) -> None:
        """Handle order update."""
        try:
            self.logger.debug(f"Order update: {order.symbol} {order.status.value}")
            
            # Update risk manager with order status
            if order.is_filled():
                # Order filled - update position tracking
                pass
            elif order.status.value in ['CANCELED', 'REJECTED', 'EXPIRED']:
                # Order failed - remove from tracking
                pass
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    async def _handle_position_update(self, position) -> None:
        """Handle position update."""
        try:
            self.logger.debug(f"Position update: {position.symbol} {position.position_amount}")
            
            # Update risk manager
            self.risk_manager.update_position(position)
            
        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")
    
    async def _handle_trade_execution(self, trade_data) -> None:
        """Handle trade execution."""
        try:
            self.logger.info(f"Trade executed: {trade_data['symbol']} {trade_data['side']} {trade_data['quantity']}")
            
            # Update performance tracker
            self.monitoring_dashboard.performance_tracker.record_trade(trade_data)
            
            # Update risk manager
            self.risk_manager.circuit_breaker.update_trade_result(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error handling trade execution: {e}")
    
    async def _handle_performance_metrics(self, metrics) -> None:
        """Handle performance metrics update."""
        try:
            # Log performance metrics periodically
            if metrics.total_trades > 0 and metrics.total_trades % 10 == 0:
                self.logger.performance_log(metrics.to_dict())
                
        except Exception as e:
            self.logger.error(f"Error handling performance metrics: {e}")
    
    async def _handle_health_check(self, health_check) -> None:
        """Handle health check update."""
        try:
            if health_check.status == 'critical':
                self.logger.critical(f"Critical health check: {health_check.component} - {health_check.message}")
            elif health_check.status == 'warning':
                self.logger.warning(f"Health check warning: {health_check.component} - {health_check.message}")
            else:
                self.logger.debug(f"Health check: {health_check.component} - {health_check.message}")
                
        except Exception as e:
            self.logger.error(f"Error handling health check: {e}")
    
    async def _emergency_stop(self) -> None:
        """Emergency stop - close all positions."""
        try:
            self.logger.critical("EMERGENCY STOP - Closing all positions")
            
            # Get all positions
            positions = await self.executor.get_positions()
            
            # Close each position
            for position in positions:
                if position.position_amount != 0:
                    side = 'SELL' if position.position_amount > 0 else 'BUY'
                    await self.executor.close_position(
                        position.symbol, 
                        side, 
                        abs(position.position_amount)
                    )
            
            # Cancel all open orders
            open_orders = await self.executor.get_open_orders()
            for order in open_orders:
                await self.executor.cancel_order(order.symbol, order.order_id)
            
            self.logger.critical("Emergency stop completed")
            
        except Exception as e:
            self.logger.error(f"Error in emergency stop: {e}")
    
    async def _close_position(self, symbol: str) -> None:
        """Close specific position."""
        try:
            positions = await self.executor.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)
            
            if position and position.position_amount != 0:
                side = 'SELL' if position.position_amount > 0 else 'BUY'
                await self.executor.close_position(symbol, side, abs(position.position_amount))
                self.logger.info(f"Closed position: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status."""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.utcnow() - self.start_time) if self.start_time else None,
            'active_tasks': len([t for t in self.tasks if not t.done()]) if self.tasks else 0,
            'config': {
                'symbols': self.config.get_symbols(),
                'leverage': self.config.get_leverage(),
                'testnet': self.config.is_testnet()
            }
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        if self.monitoring_dashboard:
            return self.monitoring_dashboard.get_dashboard_data()
        return {}


async def main():
    """Main entry point."""
    try:
        # Create and start bot
        bot = TradingBot()
        await bot.start()
        
    except KeyboardInterrupt:
        print("\\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())