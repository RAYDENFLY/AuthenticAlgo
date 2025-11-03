"""
Bot Trading V2 - Main Entry Point
Professional Trading Bot with ML Integration
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_config, setup_logger, get_logger
from core.exceptions import BotTradingException


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bot Trading V2 - Professional Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in paper trading mode
  python main.py --mode paper

  # Run backtest
  python main.py --mode backtest --start 2023-01-01 --end 2024-12-31

  # Run live trading
  python main.py --mode live

  # Run specific strategy
  python main.py --mode paper --strategy RSI_MACD_Strategy
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['paper', 'live', 'backtest'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        help='Specific strategy name to run'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def run_paper_trading(config, strategy_name=None):
    """Run bot in paper trading mode"""
    logger = get_logger()
    logger.info("🚀 Starting Bot in PAPER TRADING mode...")
    logger.info("=" * 60)
    
    # Paper trading implementation will go here
    logger.warning("⚠️ Paper trading mode is not yet implemented")
    logger.info("This is where the trading bot will run in simulation mode")
    logger.info("No real money will be used")
    
    logger.info("=" * 60)


def run_live_trading(config, strategy_name=None):
    """Run bot in live trading mode"""
    logger = get_logger()
    logger.warning("⚠️ STARTING LIVE TRADING MODE - REAL MONEY AT RISK! ⚠️")
    logger.info("=" * 60)
    
    # Confirmation check
    confirmation = input("Are you sure you want to trade with real money? (yes/no): ")
    if confirmation.lower() != 'yes':
        logger.info("Live trading cancelled by user")
        return
    
    # Live trading implementation will go here
    logger.warning("⚠️ Live trading mode is not yet implemented")
    logger.info("This is where the trading bot will execute real trades")
    
    logger.info("=" * 60)


def run_backtest(config, strategy_name=None, start_date=None, end_date=None):
    """Run backtesting"""
    logger = get_logger()
    logger.info("📊 Starting BACKTESTING mode...")
    logger.info("=" * 60)
    logger.info(f"Strategy: {strategy_name or 'All strategies'}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Backtesting implementation will go here
    logger.warning("⚠️ Backtesting mode is not yet implemented")
    logger.info("This is where historical strategy testing will occur")
    
    logger.info("=" * 60)


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           🤖 BOT TRADING V2 - Professional Bot 🤖          ║
    ║                                                           ║
    ║                    Version 2.0.0                          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner
        print_banner()
        
        # Setup logger
        setup_logger(log_level=args.log_level)
        logger = get_logger()
        
        # Load configuration
        config = get_config() if args.config is None else get_config(args.config)
        
        logger.info(f"Trading Mode: {args.mode.upper()}")
        logger.info(f"Configuration loaded from: {config.config_path}")
        logger.info(f"Environment: {config.get_env('ENV', 'development')}")
        
        # Run based on mode
        if args.mode == 'paper':
            run_paper_trading(config, args.strategy)
        elif args.mode == 'live':
            run_live_trading(config, args.strategy)
        elif args.mode == 'backtest':
            if not args.start or not args.end:
                logger.error("Backtest mode requires --start and --end dates")
                sys.exit(1)
            run_backtest(config, args.strategy, args.start, args.end)
        
    except KeyboardInterrupt:
        logger = get_logger()
        logger.info("\n⚠️ Bot stopped by user (Ctrl+C)")
        sys.exit(0)
    except BotTradingException as e:
        logger = get_logger()
        logger.error(f"Bot Trading Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger = get_logger()
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
