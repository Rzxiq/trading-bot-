#!/usr/bin/env python3
"""
Main entry point for the Binance Futures Trading Bot.
"""
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.trading_bot import TradingBot
from src.backtesting import BacktestEngine
from src.logger import TradingLogger
from src.config import Config


async def run_live_trading(config_path: str):
    """Run live trading bot."""
    print("🚀 Starting Live Trading Bot...")
    
    try:
        bot = TradingBot(config_path)
        await bot.start()
    except KeyboardInterrupt:
        print("\\n⏹️  Trading bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def run_backtest(config_path: str, start_date: str, end_date: str, output_file: str = None):
    """Run backtest."""
    print("📊 Starting Backtest...")
    
    try:
        # Load configuration
        config = Config(config_path)
        logger = TradingLogger(config.get_logging_config())
        
        # Update backtest config
        backtest_config = config.get_backtesting_config()
        backtest_config['start_date'] = start_date
        backtest_config['end_date'] = end_date
        backtest_config['enabled'] = True
        
        # Create backtest engine
        backtest_engine = BacktestEngine(backtest_config, logger)
        
        # Create strategy
        from src.strategy import RSIBollingerStrategy
        strategy = RSIBollingerStrategy(config.get_strategy_config(), logger)
        
        # Create data processor
        from src.data_processor import DataProcessor
        data_processor = DataProcessor(config.get_strategy_config(), logger)
        
        # Run backtest
        result = await backtest_engine.run_backtest(strategy, data_processor)
        
        # Print results
        print("\\n📈 Backtest Results:")
        print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Balance: ${result.initial_balance:,.2f}")
        print(f"Final Balance: ${result.final_balance:,.2f}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        
        # Save results if output file specified
        if output_file:
            backtest_engine.save_results(result, output_file)
            print(f"\\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        sys.exit(1)


async def run_paper_trading(config_path: str):
    """Run paper trading mode."""
    print("📝 Starting Paper Trading...")
    
    try:
        # Load configuration
        config = Config(config_path)
        logger = TradingLogger(config.get_logging_config())
        
        # Update config for paper trading
        api_config = config.get_api_config()
        api_config['testnet'] = True  # Force testnet for paper trading
        
        # Create and start bot
        bot = TradingBot(config_path)
        await bot.start()
        
    except KeyboardInterrupt:
        print("\\n⏹️  Paper trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def create_sample_config():
    """Create sample configuration file."""
    config_content = """# Binance Futures Trading Bot Configuration

# API Configuration
api:
  binance:
    api_key: "YOUR_API_KEY_HERE"
    secret_key: "YOUR_SECRET_KEY_HERE"
    testnet: true  # Set to false for live trading
    base_url: "https://testnet.binancefuture.com"  # Use https://fapi.binance.com for live

# Trading Configuration
trading:
  symbols: ["BTCUSDT", "ETHUSDT"]
  leverage: 10
  max_position_size: 0.1  # 10% of balance per position
  max_daily_loss: 0.05    # 5% max daily loss
  max_drawdown: 0.15      # 15% max drawdown

# WebSocket Configuration
websocket:
  base_url: "wss://fstream.binance.com/ws"  # Use wss://fstream.binance.com/ws for live
  reconnect_interval: 5
  max_reconnect_attempts: 10
  ping_interval: 30
  ping_timeout: 10

# Strategy Configuration
strategy:
  name: "rsi_bollinger"
  params:
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    bb_period: 20
    bb_std: 2
    order_book_levels: 20
    min_volume: 1000000  # Minimum 24h volume in USDT

# Risk Management
risk:
  stop_loss_pct: 0.02     # 2% stop loss
  take_profit_pct: 0.04   # 4% take profit
  max_open_positions: 3
  position_timeout: 3600  # 1 hour max position hold time
  volatility_threshold: 0.05  # 5% volatility threshold for circuit breaker

# Logging Configuration
logging:
  level: "INFO"
  file: "logs/trading_bot.log"
  max_file_size: "10MB"
  backup_count: 5
  telegram:
    enabled: false
    bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
  discord:
    enabled: false
    webhook_url: "YOUR_DISCORD_WEBHOOK_URL"

# Performance Monitoring
monitoring:
  metrics_interval: 30  # seconds
  health_check_interval: 30  # seconds
  latency_threshold: 1000  # milliseconds
  enable_prometheus: false
  prometheus_port: 8000

# Backtesting Configuration
backtesting:
  enabled: false
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  initial_balance: 10000
  commission: 0.0004  # 0.04% commission
  slippage: 0.0001    # 0.01% slippage

# Resilience Configuration
resilience:
  state_file: "data/bot_state.pkl"
  backup_dir: "data/backups"
  health_check_interval: 30
  max_health_failures: 3
  max_reconnect_attempts: 10
  reconnect_delay: 5
  exponential_backoff: true
  max_delay: 300
  shutdown_timeout: 30
  force_shutdown: true
"""
    
    with open('config.yaml', 'w') as f:
        f.write(config_content)
    
    print("✅ Sample configuration created: config.yaml")
    print("📝 Please edit config.yaml with your API keys and preferences")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', '-m', choices=['live', 'paper', 'backtest'], default='paper',
                       help='Trading mode')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', help='Backtest output file')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("💡 Use --create-config to create a sample configuration file")
        sys.exit(1)
    
    # Run appropriate mode
    if args.mode == 'live':
        asyncio.run(run_live_trading(args.config))
    elif args.mode == 'paper':
        asyncio.run(run_paper_trading(args.config))
    elif args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            print("❌ Backtest mode requires --start-date and --end-date")
            sys.exit(1)
        asyncio.run(run_backtest(args.config, args.start_date, args.end_date, args.output))


if __name__ == '__main__':
    main()