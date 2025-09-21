"""
Resilience features for robust trading bot operation.
"""
import asyncio
import json
import os
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from .logger import TradingLogger


class StateManager:
    """State persistence and recovery manager."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.state_file = config.get('state_file', 'data/bot_state.pkl')
        self.backup_dir = config.get('backup_dir', 'data/backups')
        
        # Ensure directories exist
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # State data
        self.state = {
            'positions': {},
            'orders': {},
            'account_balance': 0.0,
            'equity': 0.0,
            'last_update': datetime.utcnow(),
            'version': '1.0.0'
        }
    
    def save_state(self, bot_state: Dict[str, Any]) -> None:
        """Save bot state to disk."""
        try:
            self.state.update(bot_state)
            self.state['last_update'] = datetime.utcnow()
            
            # Create backup
            backup_file = f"{self.backup_dir}/state_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            if os.path.exists(self.state_file):
                os.rename(self.state_file, backup_file)
            
            # Save current state
            with open(self.state_file, 'wb') as f:
                pickle.dump(self.state, f)
            
            self.logger.debug("State saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def load_state(self) -> Dict[str, Any]:
        """Load bot state from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.logger.info("State loaded successfully")
                return state
            else:
                self.logger.info("No state file found, starting fresh")
                return self.state
                
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return self.state
    
    def cleanup_old_backups(self, max_backups: int = 10) -> None:
        """Clean up old backup files."""
        try:
            backup_files = sorted(Path(self.backup_dir).glob('state_backup_*.pkl'))
            
            if len(backup_files) > max_backups:
                files_to_delete = backup_files[:-max_backups]
                for file in files_to_delete:
                    file.unlink()
                    self.logger.debug(f"Deleted old backup: {file}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")


class HealthChecker:
    """System health monitoring and recovery."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Health check parameters
        self.check_interval = config.get('health_check_interval', 30)
        self.max_failures = config.get('max_health_failures', 3)
        self.recovery_timeout = config.get('recovery_timeout', 300)  # 5 minutes
        
        # Health tracking
        self.health_failures = 0
        self.last_health_check = datetime.utcnow()
        self.is_healthy = True
        
        # Component health status
        self.component_health: Dict[str, bool] = {}
        
        # Recovery callbacks
        self.recovery_callbacks: List[Callable[[str], None]] = []
    
    def add_recovery_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for component recovery."""
        self.recovery_callbacks.append(callback)
    
    async def check_system_health(self, components: Dict[str, Any]) -> bool:
        """Check overall system health."""
        try:
            self.last_health_check = datetime.utcnow()
            all_healthy = True
            
            # Check WebSocket connection
            if 'data_feed' in components:
                ws_healthy = components['data_feed'].is_stream_healthy()
                self.component_health['websocket'] = ws_healthy
                if not ws_healthy:
                    all_healthy = False
                    await self._handle_component_failure('websocket')
            
            # Check execution engine
            if 'executor' in components:
                try:
                    stats = components['executor'].get_execution_stats()
                    exec_healthy = stats.get('success_rate', 0) > 0.5
                    self.component_health['execution'] = exec_healthy
                    if not exec_healthy:
                        all_healthy = False
                        await self._handle_component_failure('execution')
                except Exception:
                    self.component_health['execution'] = False
                    all_healthy = False
                    await self._handle_component_failure('execution')
            
            # Check data processor
            if 'data_processor' in components:
                try:
                    stats = components['data_processor'].get_processing_stats()
                    proc_healthy = stats.get('average_processing_time', 0) < 1000
                    self.component_health['data_processing'] = proc_healthy
                    if not proc_healthy:
                        all_healthy = False
                        await self._handle_component_failure('data_processing')
                except Exception:
                    self.component_health['data_processing'] = False
                    all_healthy = False
                    await self._handle_component_failure('data_processing')
            
            # Update overall health status
            if all_healthy:
                self.health_failures = 0
                self.is_healthy = True
            else:
                self.health_failures += 1
                self.is_healthy = False
                
                if self.health_failures >= self.max_failures:
                    await self._handle_system_failure()
            
            return all_healthy
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return False
    
    async def _handle_component_failure(self, component: str) -> None:
        """Handle individual component failure."""
        self.logger.warning(f"Component failure detected: {component}")
        
        # Trigger recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                callback(component)
            except Exception as e:
                self.logger.error(f"Error in recovery callback: {e}")
    
    async def _handle_system_failure(self) -> None:
        """Handle system-wide failure."""
        self.logger.critical("System failure detected - initiating emergency procedures")
        
        # This would trigger emergency shutdown procedures
        # Implementation depends on the specific requirements


class AutoReconnector:
    """Automatic reconnection manager."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Reconnection parameters
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.reconnect_delay = config.get('reconnect_delay', 5)
        self.exponential_backoff = config.get('exponential_backoff', True)
        self.max_delay = config.get('max_delay', 300)  # 5 minutes
        
        # Reconnection tracking
        self.reconnect_attempts = 0
        self.last_reconnect = datetime.utcnow()
        self.is_reconnecting = False
    
    async def attempt_reconnection(self, component_name: str, reconnect_func: Callable) -> bool:
        """Attempt to reconnect a component."""
        if self.is_reconnecting:
            self.logger.warning(f"Reconnection already in progress for {component_name}")
            return False
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts reached for {component_name}")
            return False
        
        try:
            self.is_reconnecting = True
            self.reconnect_attempts += 1
            
            self.logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} for {component_name}")
            
            # Calculate delay
            delay = self._calculate_delay()
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Attempt reconnection
            success = await reconnect_func()
            
            if success:
                self.logger.info(f"Successfully reconnected {component_name}")
                self.reconnect_attempts = 0
                self.last_reconnect = datetime.utcnow()
                return True
            else:
                self.logger.warning(f"Reconnection failed for {component_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during reconnection of {component_name}: {e}")
            return False
        finally:
            self.is_reconnecting = False
    
    def _calculate_delay(self) -> float:
        """Calculate reconnection delay."""
        if not self.exponential_backoff:
            return self.reconnect_delay
        
        # Exponential backoff with jitter
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), self.max_delay)
        jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)  # ±10% jitter
        return max(0, delay + jitter)
    
    def reset_attempts(self) -> None:
        """Reset reconnection attempts."""
        self.reconnect_attempts = 0
        self.logger.debug("Reconnection attempts reset")


class GracefulShutdown:
    """Graceful shutdown manager."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Shutdown parameters
        self.shutdown_timeout = config.get('shutdown_timeout', 30)
        self.force_shutdown = config.get('force_shutdown', True)
        
        # Shutdown state
        self.is_shutting_down = False
        self.shutdown_started = None
        
        # Shutdown callbacks
        self.shutdown_callbacks: List[Callable[[], None]] = []
    
    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for shutdown procedures."""
        self.shutdown_callbacks.append(callback)
    
    async def initiate_shutdown(self, reason: str = "Manual shutdown") -> None:
        """Initiate graceful shutdown."""
        if self.is_shutting_down:
            self.logger.warning("Shutdown already in progress")
            return
        
        self.logger.info(f"Initiating graceful shutdown: {reason}")
        self.is_shutting_down = True
        self.shutdown_started = datetime.utcnow()
        
        try:
            # Execute shutdown callbacks
            for callback in self.shutdown_callbacks:
                try:
                    await callback()
                except Exception as e:
                    self.logger.error(f"Error in shutdown callback: {e}")
            
            # Wait for shutdown to complete
            await self._wait_for_shutdown_completion()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.logger.info("Graceful shutdown completed")
    
    async def _wait_for_shutdown_completion(self) -> None:
        """Wait for shutdown to complete within timeout."""
        if not self.shutdown_started:
            return
        
        elapsed = (datetime.utcnow() - self.shutdown_started).total_seconds()
        remaining = self.shutdown_timeout - elapsed
        
        if remaining > 0:
            self.logger.info(f"Waiting {remaining:.1f}s for shutdown completion...")
            await asyncio.sleep(remaining)
        
        if self.force_shutdown and self.is_shutting_down:
            self.logger.warning("Force shutdown triggered")
            # Force exit
            os._exit(1)


class ResilienceManager:
    """Main resilience management system."""
    
    def __init__(self, config: Dict[str, Any], logger: TradingLogger):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.state_manager = StateManager(config, logger)
        self.health_checker = HealthChecker(config, logger)
        self.auto_reconnector = AutoReconnector(config, logger)
        self.graceful_shutdown = GracefulShutdown(config, logger)
        
        # Resilience state
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Component references
        self.components: Dict[str, Any] = {}
    
    def register_component(self, name: str, component: Any) -> None:
        """Register component for monitoring."""
        self.components[name] = component
        self.logger.debug(f"Registered component: {name}")
    
    async def start_monitoring(self) -> None:
        """Start resilience monitoring."""
        if self.is_monitoring:
            return
        
        self.logger.info("Starting resilience monitoring")
        self.is_monitoring = True
        
        # Start health monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop resilience monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping resilience monitoring")
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check system health
                await self.health_checker.check_system_health(self.components)
                
                # Save state periodically
                if hasattr(self, 'bot_state'):
                    self.state_manager.save_state(self.bot_state)
                
                # Clean up old backups
                self.state_manager.cleanup_old_backups()
                
                await asyncio.sleep(self.health_checker.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def handle_component_failure(self, component_name: str) -> None:
        """Handle component failure with reconnection."""
        self.logger.warning(f"Handling component failure: {component_name}")
        
        if component_name == 'websocket' and 'data_feed' in self.components:
            # Attempt WebSocket reconnection
            data_feed = self.components['data_feed']
            success = await self.auto_reconnector.attempt_reconnection(
                'websocket',
                lambda: data_feed.connect(data_feed.symbols)
            )
            
            if success:
                self.logger.info("WebSocket reconnection successful")
            else:
                self.logger.error("WebSocket reconnection failed")
        
        # Add other component recovery logic here
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            asyncio.create_task(self.graceful_shutdown.initiate_shutdown(f"Signal {signum}"))
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get current resilience status."""
        return {
            'is_monitoring': self.is_monitoring,
            'is_healthy': self.health_checker.is_healthy,
            'health_failures': self.health_checker.health_failures,
            'reconnect_attempts': self.auto_reconnector.reconnect_attempts,
            'is_reconnecting': self.auto_reconnector.is_reconnecting,
            'is_shutting_down': self.graceful_shutdown.is_shutting_down,
            'component_health': self.health_checker.component_health,
            'last_health_check': self.health_checker.last_health_check.isoformat()
        }