# 🚀 Binance Futures Trading Bot

A high-frequency, event-driven futures trading bot built with Python that connects to Binance's WebSocket streams for real-time market data, implements customizable trading strategies, and executes orders via Binance Futures REST API.

## ✨ Features

### 🔌 WebSocket Integration
- **Real-time Data Streams**: Order book, klines, trades, mark price, liquidations
- **Multiple Stream Support**: Simultaneous subscription to multiple symbols and intervals
- **Auto-reconnection**: Robust reconnection logic with exponential backoff
- **Data Synchronization**: Thread-safe data buffering and processing

### 📊 Advanced Data Processing
- **Real-time Indicators**: RSI, Bollinger Bands, MACD, Stochastic, Williams %R, ATR, ADX
- **Order Book Analysis**: Imbalance calculation, pressure metrics, spread analysis
- **Volume Profile**: VWAP, volume trends, trade flow analysis
- **Market Sentiment**: Combined sentiment scoring from multiple data sources

### 🎯 Strategy Engine
- **Modular Design**: Easy to implement custom strategies
- **Sample Strategy**: RSI + Bollinger Bands mean reversion with order book confirmation
- **Signal Generation**: Multi-factor signal validation and confidence scoring
- **Strategy Management**: Multiple strategies with independent risk management

### 🛡️ Risk Management
- **Position Sizing**: Fixed, volatility-adjusted, and Kelly Criterion methods
- **Stop Loss & Take Profit**: ATR-based and percentage-based levels
- **Circuit Breakers**: Daily loss limits, drawdown protection, consecutive loss limits
- **Real-time Monitoring**: Continuous risk assessment and position tracking

### ⚡ Order Execution
- **Binance Futures API**: Full integration with Binance Futures REST API
- **Order Types**: Market, limit, stop-market, OCO orders
- **Rate Limiting**: Built-in rate limiting and retry logic
- **Execution Tracking**: Real-time order status monitoring and performance metrics

### 📈 Performance Monitoring
- **Real-time Metrics**: PnL, win rate, drawdown, Sharpe ratio, latency tracking
- **Health Checks**: Component health monitoring with automatic recovery
- **Alerting**: Telegram and Discord notifications for critical events
- **Dashboard**: Real-time performance dashboard with key metrics

### 🔄 Resilience Features
- **State Persistence**: Automatic state saving and crash recovery
- **Graceful Shutdown**: Clean shutdown with position protection
- **Health Monitoring**: Continuous system health checks
- **Auto-recovery**: Automatic component recovery and reconnection

### 🧪 Backtesting & Paper Trading
- **Historical Backtesting**: Strategy validation on historical data
- **Paper Trading**: Risk-free simulation with realistic execution
- **Performance Analysis**: Comprehensive backtest results and metrics
- **Data Export**: Export results for further analysis

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd binance-futures-bot

# Install dependencies
pip install -r requirements.txt

# Create configuration file
python main.py --create-config
```

### 2. Configuration

Edit `config.yaml` with your settings:

```yaml
# API Configuration
api:
  binance:
    api_key: "YOUR_API_KEY_HERE"
    secret_key: "YOUR_SECRET_KEY_HERE"
    testnet: true  # Start with testnet!

# Trading Configuration
trading:
  symbols: ["BTCUSDT", "ETHUSDT"]
  leverage: 10
  max_position_size: 0.1  # 10% of balance per position

# Strategy Configuration
strategy:
  name: "rsi_bollinger"
  params:
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    bb_period: 20
    bb_std: 2
```

### 3. Run the Bot

```bash
# Paper trading (recommended for testing)
python main.py --mode paper

# Live trading (use with caution!)
python main.py --mode live

# Backtesting
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31
```

## 📁 Project Structure

```
binance-futures-bot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Advanced logging system
│   ├── data_models.py         # Data structures and models
│   ├── data_feed.py           # WebSocket data feed
│   ├── indicators.py          # Technical indicators
│   ├── data_processor.py      # Data processing pipeline
│   ├── strategy.py            # Strategy engine
│   ├── risk_manager.py        # Risk management system
│   ├── execution.py           # Order execution engine
│   ├── monitoring.py          # Performance monitoring
│   ├── backtesting.py         # Backtesting engine
│   ├── resilience.py          # Resilience features
│   └── trading_bot.py         # Main bot orchestrator
├── logs/                      # Log files
├── data/                      # Data and state files
├── tests/                     # Test files
├── config.yaml               # Configuration file
├── requirements.txt          # Dependencies
├── main.py                   # Entry point
└── README.md                 # This file
```

## 🎯 Strategy Development

### Creating a Custom Strategy

```python
from src.strategy import StrategyBase, StrategySignal, OrderSide, SignalStrength

class MyCustomStrategy(StrategyBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        # Initialize your strategy parameters
        
    async def analyze(self, market_data):
        # Your strategy logic here
        indicators = market_data.get('indicators', {})
        
        # Example: Simple moving average crossover
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            if sma_20 > sma_50:
                return StrategySignal(
                    symbol=market_data['symbol'],
                    side=OrderSide.BUY,
                    quantity=0.1,
                    reason="SMA crossover bullish",
                    confidence=0.8,
                    strength=SignalStrength.MEDIUM
                )
        
        return None
    
    def get_required_indicators(self):
        return ['sma_20', 'sma_50']
```

### Adding Strategy to Bot

```python
# In trading_bot.py
my_strategy = MyCustomStrategy(config, logger)
strategy_manager.add_strategy(my_strategy)
```

## 🛡️ Risk Management

### Position Sizing Methods

1. **Fixed Size**: Fixed percentage of balance
2. **Volatility Adjusted**: Adjust based on market volatility
3. **Kelly Criterion**: Optimal position sizing based on win rate and average win/loss

### Risk Limits

- **Daily Loss Limit**: Maximum daily loss percentage
- **Drawdown Limit**: Maximum drawdown from peak equity
- **Position Timeout**: Maximum time to hold a position
- **Consecutive Losses**: Maximum consecutive losing trades

### Circuit Breakers

- **High Latency**: Stop trading if latency exceeds threshold
- **Low Success Rate**: Stop if execution success rate drops
- **System Health**: Stop if critical components fail

## 📊 Monitoring & Alerts

### Real-time Metrics

- **Performance**: PnL, win rate, Sharpe ratio, max drawdown
- **Execution**: Order success rate, average execution time
- **System**: CPU usage, memory usage, network latency
- **Risk**: Current exposure, position count, risk metrics

### Alerting

Configure alerts in `config.yaml`:

```yaml
logging:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
  discord:
    enabled: true
    webhook_url: "YOUR_WEBHOOK_URL"
```

## 🧪 Backtesting

### Running Backtests

```bash
# Basic backtest
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31

# With output file
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31 --output results.json
```

### Backtest Results

- **Performance Metrics**: Total return, Sharpe ratio, max drawdown
- **Trade Analysis**: Win rate, average win/loss, profit factor
- **Risk Metrics**: Volatility, VaR, maximum drawdown duration
- **Equity Curve**: Daily equity progression

## 🔧 Configuration Reference

### API Configuration

```yaml
api:
  binance:
    api_key: "your_api_key"
    secret_key: "your_secret_key"
    testnet: true  # Use testnet for testing
    base_url: "https://testnet.binancefuture.com"
```

### Trading Configuration

```yaml
trading:
  symbols: ["BTCUSDT", "ETHUSDT"]  # Trading symbols
  leverage: 10                     # Leverage multiplier
  max_position_size: 0.1          # Max position size (10% of balance)
  max_daily_loss: 0.05            # Max daily loss (5%)
  max_drawdown: 0.15              # Max drawdown (15%)
```

### Strategy Configuration

```yaml
strategy:
  name: "rsi_bollinger"
  params:
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    bb_period: 20
    bb_std: 2
    order_book_levels: 20
    min_volume: 1000000
```

### Risk Management

```yaml
risk:
  stop_loss_pct: 0.02           # 2% stop loss
  take_profit_pct: 0.04         # 4% take profit
  max_open_positions: 3         # Max concurrent positions
  position_timeout: 3600        # Max position hold time (seconds)
  volatility_threshold: 0.05    # Volatility circuit breaker
```

## 🚨 Important Warnings

### ⚠️ Risk Disclaimer

- **High Risk**: Futures trading involves substantial risk of loss
- **No Guarantees**: Past performance does not guarantee future results
- **Test First**: Always test strategies on paper trading or testnet
- **Monitor Closely**: Never leave the bot unattended for extended periods
- **Start Small**: Begin with small position sizes and gradually increase

### 🔒 Security Best Practices

- **API Keys**: Use testnet keys for testing, never share live keys
- **Permissions**: Limit API key permissions to only what's needed
- **Monitoring**: Set up alerts for unusual activity
- **Backups**: Regularly backup configuration and state files

## 🐛 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check internet connection
   - Verify API keys are correct
   - Ensure testnet/live configuration matches

2. **Order Execution Failed**
   - Check account balance
   - Verify symbol is supported
   - Check leverage settings

3. **High Latency**
   - Check internet connection
   - Consider using VPS closer to exchange
   - Reduce data processing complexity

### Debug Mode

Enable debug logging in `config.yaml`:

```yaml
logging:
  level: "DEBUG"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Binance for providing excellent API documentation
- The Python community for amazing libraries
- Contributors and testers

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the configuration examples

---

**Happy Trading! 🚀📈**

*Remember: Trade responsibly and never risk more than you can afford to lose.*