"""
Monitoring and performance tracking system.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

from .data_models import PerformanceMetrics, Order, Position
from .logger import TradingLogger


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: datetime
    latency_ms: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_latency: float
    websocket_connections: int
    active_orders: int
    open_positions: int
    data_processing_time: float
    strategy_processing_time: float
    execution_time: float


class PerformanceTracker:
    """Performance tracking and metrics collection."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Performance data
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_pnl: Dict[str, float] = {}
        self.hourly_pnl: Dict[str, float] = {}
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Latency tracking
        self.latency_history: List[float] = []
        self.data_processing_times: List[float] = []
        self.strategy_processing_times: List[float] = []
        self.execution_times: List[float] = []
        
        # System metrics
        self.system_metrics: List[SystemMetrics] = []
        
        # Callbacks
        self.metrics_callbacks: List[Callable[[PerformanceMetrics], None]] = []
        self.health_check_callbacks: List[Callable[[HealthCheck], None]] = []
    
    def add_metrics_callback(self, callback: Callable[[PerformanceMetrics], None]) -> None:
        """Add callback for performance metrics updates."""
        self.metrics_callbacks.append(callback)
    
    def add_health_check_callback(self, callback: Callable[[HealthCheck], None]) -> None:
        """Add callback for health check updates."""
        self.health_check_callbacks.append(callback)
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record trade execution."""
        self.trade_history.append(trade_data)
        
        # Update daily PnL
        date_key = trade_data['timestamp'].strftime('%Y-%m-%d')
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = 0.0
        self.daily_pnl[date_key] += trade_data.get('pnl', 0.0)
        
        # Update hourly PnL
        hour_key = trade_data['timestamp'].strftime('%Y-%m-%d %H:00')
        if hour_key not in self.hourly_pnl:
            self.hourly_pnl[hour_key] = 0.0
        self.hourly_pnl[hour_key] += trade_data.get('pnl', 0.0)
        
        # Update drawdown tracking
        self._update_drawdown_tracking()
    
    def record_latency(self, latency_ms: float) -> None:
        """Record latency measurement."""
        self.latency_history.append(latency_ms)
        
        # Keep only recent measurements
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
    
    def record_data_processing_time(self, processing_time_ms: float) -> None:
        """Record data processing time."""
        self.data_processing_times.append(processing_time_ms)
        
        if len(self.data_processing_times) > 1000:
            self.data_processing_times = self.data_processing_times[-1000:]
    
    def record_strategy_processing_time(self, processing_time_ms: float) -> None:
        """Record strategy processing time."""
        self.strategy_processing_times.append(processing_time_ms)
        
        if len(self.strategy_processing_times) > 1000:
            self.strategy_processing_times = self.strategy_processing_times[-1000:]
    
    def record_execution_time(self, execution_time_ms: float) -> None:
        """Record order execution time."""
        self.execution_times.append(execution_time_ms)
        
        if len(self.execution_times) > 1000:
            self.execution_times = self.execution_times[-1000:]
    
    def _update_drawdown_tracking(self) -> None:
        """Update maximum drawdown tracking."""
        current_equity = sum(self.daily_pnl.values())
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        current_drawdown = self.peak_equity - current_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
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
        
        # Calculate basic metrics
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trade_history)
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        average_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        average_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Calculate profit factor
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate Sharpe ratio
        returns = [t.get('pnl', 0) for t in self.trade_history]
        if len(returns) > 1:
            mean_return = statistics.mean(returns)
            variance = statistics.variance(returns)
            sharpe_ratio = mean_return / (variance ** 0.5) if variance > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate daily PnL
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl.get(today, 0.0)
        
        # Calculate average latency
        avg_latency = statistics.mean(self.latency_history) if self.latency_history else 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            win_rate=win_rate,
            total_trades=len(self.trade_history),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            latency_ms=avg_latency
        )
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # This would typically integrate with system monitoring tools
        # For now, return basic metrics
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=0.0,  # Would be populated by system monitoring
            memory_usage=0.0,  # Would be populated by system monitoring
            network_latency=statistics.mean(self.latency_history) if self.latency_history else 0.0,
            websocket_connections=1,  # Would be tracked by WebSocket manager
            active_orders=0,  # Would be tracked by execution engine
            open_positions=0,  # Would be tracked by execution engine
            data_processing_time=statistics.mean(self.data_processing_times) if self.data_processing_times else 0.0,
            strategy_processing_time=statistics.mean(self.strategy_processing_times) if self.strategy_processing_times else 0.0,
            execution_time=statistics.mean(self.execution_times) if self.execution_times else 0.0
        )
    
    async def emit_metrics(self) -> None:
        """Emit performance metrics to callbacks."""
        metrics = self.get_performance_metrics()
        
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                self.logger.error(f"Error in metrics callback: {e}")


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Health check intervals
        self.health_check_interval = config.get('health_check_interval', 30)
        self.latency_threshold = config.get('latency_threshold', 1000)
        
        # Component health status
        self.component_status: Dict[str, HealthCheck] = {}
        
        # Callbacks
        self.health_check_callbacks: List[Callable[[HealthCheck], None]] = []
    
    def add_health_check_callback(self, callback: Callable[[HealthCheck], None]) -> None:
        """Add callback for health check updates."""
        self.health_check_callbacks.append(callback)
    
    async def check_websocket_health(self, data_feed) -> HealthCheck:
        """Check WebSocket connection health."""
        start_time = datetime.utcnow()
        
        try:
            is_healthy = data_feed.is_stream_healthy()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if is_healthy:
                status = 'healthy'
                message = 'WebSocket connection active'
            else:
                status = 'critical'
                message = 'WebSocket connection lost'
            
            health_check = HealthCheck(
                component='websocket',
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details={'is_connected': is_healthy}
            )
            
            self.component_status['websocket'] = health_check
            return health_check
            
        except Exception as e:
            health_check = HealthCheck(
                component='websocket',
                status='critical',
                message=f'WebSocket health check failed: {e}',
                timestamp=datetime.utcnow(),
                details={'error': str(e)}
            )
            
            self.component_status['websocket'] = health_check
            return health_check
    
    async def check_execution_health(self, executor) -> HealthCheck:
        """Check order execution health."""
        start_time = datetime.utcnow()
        
        try:
            # Get execution stats
            stats = executor.get_execution_stats()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            success_rate = stats.get('success_rate', 0)
            avg_execution_time = stats.get('average_execution_time', 0)
            
            if success_rate > 0.9 and avg_execution_time < 1000:
                status = 'healthy'
                message = f'Execution healthy: {success_rate:.1%} success rate'
            elif success_rate > 0.7:
                status = 'warning'
                message = f'Execution degraded: {success_rate:.1%} success rate'
            else:
                status = 'critical'
                message = f'Execution critical: {success_rate:.1%} success rate'
            
            health_check = HealthCheck(
                component='execution',
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details=stats
            )
            
            self.component_status['execution'] = health_check
            return health_check
            
        except Exception as e:
            health_check = HealthCheck(
                component='execution',
                status='critical',
                message=f'Execution health check failed: {e}',
                timestamp=datetime.utcnow(),
                details={'error': str(e)}
            )
            
            self.component_status['execution'] = health_check
            return health_check
    
    async def check_data_processing_health(self, data_processor) -> HealthCheck:
        """Check data processing health."""
        start_time = datetime.utcnow()
        
        try:
            stats = data_processor.get_processing_stats()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            avg_processing_time = stats.get('average_processing_time', 0)
            max_processing_time = stats.get('max_processing_time', 0)
            
            if avg_processing_time < 50 and max_processing_time < 200:
                status = 'healthy'
                message = f'Data processing healthy: {avg_processing_time:.1f}ms avg'
            elif avg_processing_time < 100:
                status = 'warning'
                message = f'Data processing slow: {avg_processing_time:.1f}ms avg'
            else:
                status = 'critical'
                message = f'Data processing critical: {avg_processing_time:.1f}ms avg'
            
            health_check = HealthCheck(
                component='data_processing',
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details=stats
            )
            
            self.component_status['data_processing'] = health_check
            return health_check
            
        except Exception as e:
            health_check = HealthCheck(
                component='data_processing',
                status='critical',
                message=f'Data processing health check failed: {e}',
                timestamp=datetime.utcnow(),
                details={'error': str(e)}
            )
            
            self.component_status['data_processing'] = health_check
            return health_check
    
    async def check_risk_health(self, risk_manager) -> HealthCheck:
        """Check risk management health."""
        start_time = datetime.utcnow()
        
        try:
            metrics = risk_manager.get_risk_metrics()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            exposure_ratio = metrics.get('exposure_ratio', 0)
            consecutive_losses = metrics.get('consecutive_losses', 0)
            circuit_breaker_active = metrics.get('circuit_breaker_active', False)
            
            if circuit_breaker_active:
                status = 'critical'
                message = 'Circuit breaker active'
            elif exposure_ratio > 0.8 or consecutive_losses > 3:
                status = 'warning'
                message = f'High risk: {exposure_ratio:.1%} exposure, {consecutive_losses} losses'
            else:
                status = 'healthy'
                message = f'Risk management healthy: {exposure_ratio:.1%} exposure'
            
            health_check = HealthCheck(
                component='risk_management',
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details=metrics
            )
            
            self.component_status['risk_management'] = health_check
            return health_check
            
        except Exception as e:
            health_check = HealthCheck(
                component='risk_management',
                status='critical',
                message=f'Risk management health check failed: {e}',
                timestamp=datetime.utcnow(),
                details={'error': str(e)}
            )
            
            self.component_status['risk_management'] = health_check
            return health_check
    
    async def run_health_checks(self, components: Dict[str, Any]) -> List[HealthCheck]:
        """Run all health checks."""
        health_checks = []
        
        # Check WebSocket
        if 'data_feed' in components:
            health_check = await self.check_websocket_health(components['data_feed'])
            health_checks.append(health_check)
        
        # Check execution
        if 'executor' in components:
            health_check = await self.check_execution_health(components['executor'])
            health_checks.append(health_check)
        
        # Check data processing
        if 'data_processor' in components:
            health_check = await self.check_data_processing_health(components['data_processor'])
            health_checks.append(health_check)
        
        # Check risk management
        if 'risk_manager' in components:
            health_check = await self.check_risk_health(components['risk_manager'])
            health_checks.append(health_check)
        
        # Emit health check results
        for health_check in health_checks:
            for callback in self.health_check_callbacks:
                try:
                    callback(health_check)
                except Exception as e:
                    self.logger.error(f"Error in health check callback: {e}")
        
        return health_checks
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        if not self.component_status:
            return 'unknown'
        
        statuses = [check.status for check in self.component_status.values()]
        
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'healthy'
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all components."""
        return {
            'overall_health': self.get_overall_health(),
            'components': {name: asdict(check) for name, check in self.component_status.items()},
            'timestamp': datetime.utcnow().isoformat()
        }


class MonitoringDashboard:
    """Monitoring dashboard for real-time metrics."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.performance_tracker = PerformanceTracker(config, logger)
        self.health_monitor = HealthMonitor(config, logger)
        
        # Dashboard state
        self.is_running = False
        self.metrics_interval = config.get('metrics_interval', 30)
        
    async def start(self, components: Dict[str, Any]) -> None:
        """Start monitoring dashboard."""
        self.is_running = True
        self.logger.info("Starting monitoring dashboard")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._metrics_loop()),
            asyncio.create_task(self._health_check_loop(components))
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in monitoring dashboard: {e}")
        finally:
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop monitoring dashboard."""
        self.is_running = False
        self.logger.info("Stopping monitoring dashboard")
    
    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self.is_running:
            try:
                await self.performance_tracker.emit_metrics()
                await asyncio.sleep(self.metrics_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self, components: Dict[str, Any]) -> None:
        """Health check loop."""
        while self.is_running:
            try:
                await self.health_monitor.run_health_checks(components)
                await asyncio.sleep(self.health_monitor.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return {
            'performance': self.performance_tracker.get_performance_metrics().__dict__,
            'system_metrics': self.performance_tracker.get_system_metrics().__dict__,
            'health': self.health_monitor.get_health_summary(),
            'timestamp': datetime.utcnow().isoformat()
        }