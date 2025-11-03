# Trading Bot V2 - Copilot Instructions

## Project Overview
Professional Python trading bot with clean, modular architecture supporting multiple exchanges (Binance, AsterDEX), ML integration, and comprehensive risk management.

## Architecture Principles
- **Clean Code**: Follow SOLID principles, use type hints, proper error handling
- **Modularity**: Each module has single responsibility
- **Scalability**: Easy to add new strategies, exchanges, or indicators
- **Testability**: All components should be unit testable

## Code Style Guidelines
- Use Python 3.9+ features
- Follow PEP 8 style guide
- Use type hints for all functions
- Document all classes and complex functions
- Keep functions under 50 lines
- Use dependency injection for better testability

## Project Structure
```
bot_trading_v2/
├── core/               # Core utilities and base classes
├── data/              # Data management and streaming
├── indicators/        # Technical indicators
├── strategies/        # Trading strategies
├── execution/         # Order execution and management
├── ml/                # Machine learning models
├── risk/              # Risk management
├── backtesting/       # Backtesting engine
├── monitoring/        # Logging and alerts
└── config/            # Configuration files
```

## Development Guidelines
- Always use async/await for I/O operations
- Implement proper error handling with custom exceptions
- Use dataclasses for data models
- Keep configuration separate from code
- Write unit tests for critical components
- Log all important events
- Never hardcode API keys (use environment variables)

## Testing Requirements
- Unit tests for all business logic
- Integration tests for exchange connections
- Backtesting before live deployment
- Paper trading verification

## Security
- Never commit API keys or secrets
- Use .env files for sensitive data
- Validate all external inputs
- Implement rate limiting for API calls
