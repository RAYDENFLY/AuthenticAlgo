"""
Demo: AsterDEX Futures Integration
Test connectivity and basic operations
"""

import asyncio
import os
from dotenv import load_dotenv
from execution.asterdex import AsterDEXFutures
from core.logger import setup_logger
from loguru import logger

# Setup logger once
setup_logger()

# Load environment variables
load_dotenv()


async def demo_connection():
    """Demo 1: Test connection and ping"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Connection Test")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'your_api_key_here'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'your_secret_here'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com'),
        'ws_url': os.getenv('ASTERDEX_WS_URL', 'wss://fstream.asterdex.com'),
        'leverage': 10,
        'margin_type': 'ISOLATED'
    }
    
    exchange = AsterDEXFutures(config)
    
    try:
        # Test connection
        logger.info("Testing connection...")
        is_connected = await exchange.ping()
        
        if is_connected:
            logger.info("✅ Connection successful!")
        else:
            logger.error("❌ Connection failed!")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        await exchange.close()


async def demo_market_data():
    """Demo 2: Fetch market data"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Market Data")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com')
    }
    
    exchange = AsterDEXFutures(config)
    
    try:
        # Get exchange info
        logger.info("\n1. Exchange Info:")
        info = await exchange.get_exchange_info()
        symbols = [s['symbol'] for s in info.get('symbols', [])[:10]]
        logger.info(f"   Available symbols (first 10): {', '.join(symbols)}")
        
        # Get BTC/USDT ticker
        logger.info("\n2. BTC/USDT 24h Ticker:")
        ticker = await exchange.get_ticker_24h('BTCUSDT')
        logger.info(f"   Price: ${float(ticker.get('lastPrice', 0)):,.2f}")
        logger.info(f"   24h Change: {float(ticker.get('priceChangePercent', 0)):.2f}%")
        logger.info(f"   24h Volume: {float(ticker.get('volume', 0)):,.2f} BTC")
        logger.info(f"   24h High: ${float(ticker.get('highPrice', 0)):,.2f}")
        logger.info(f"   24h Low: ${float(ticker.get('lowPrice', 0)):,.2f}")
        
        # Get order book
        logger.info("\n3. BTC/USDT Order Book:")
        order_book = await exchange.get_order_book('BTCUSDT', limit=5)
        logger.info("   Top 5 Bids:")
        for bid in order_book.get('bids', [])[:5]:
            logger.info(f"      ${float(bid[0]):,.2f} - {float(bid[1]):.4f} BTC")
        logger.info("   Top 5 Asks:")
        for ask in order_book.get('asks', [])[:5]:
            logger.info(f"      ${float(ask[0]):,.2f} - {float(ask[1]):.4f} BTC")
        
        # Get recent trades
        logger.info("\n4. Recent Trades:")
        trades = await exchange.get_recent_trades('BTCUSDT', limit=5)
        for trade in trades[:5]:
            side = "🟢 BUY" if trade.get('isBuyerMaker') else "🔴 SELL"
            logger.info(f"   {side} - ${float(trade.get('price', 0)):,.2f} x {float(trade.get('qty', 0)):.4f}")
        
        # Get klines
        logger.info("\n5. Recent 5 Candles (15m):")
        klines = await exchange.get_klines('BTCUSDT', '15m', limit=5)
        for i, kline in enumerate(klines, 1):
            open_price = float(kline[1])
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            volume = float(kline[5])
            change = ((close - open_price) / open_price) * 100
            candle = "🟢" if close > open_price else "🔴"
            logger.info(f"   {candle} Candle {i}: O=${open_price:,.2f} H=${high:,.2f} L=${low:,.2f} C=${close:,.2f} ({change:+.2f}%) Vol={volume:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        await exchange.close()


async def demo_account_info():
    """Demo 3: Account information (requires valid API keys)"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Account Information (Requires Valid API Keys)")
    logger.info("="*60)
    
    api_key = os.getenv('ASTERDEX_API_KEY')
    api_secret = os.getenv('ASTERDEX_API_SECRET')
    
    if not api_key or not api_secret or api_key == 'your_api_key_here':
        logger.warning("⚠️ No valid API keys found!")
        logger.warning("   Please set ASTERDEX_API_KEY and ASTERDEX_API_SECRET in .env file")
        logger.warning("   Skipping account info demo...")
        return
    
    config = {
        'api_key': api_key,
        'api_secret': api_secret,
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com'),
        'leverage': 10,
        'margin_type': 'ISOLATED'
    }
    
    exchange = AsterDEXFutures(config)
    
    try:
        # Get account balance
        logger.info("\n1. Account Balance:")
        balances = await exchange.get_balance()
        for balance in balances:
            asset = balance.get('asset')
            free = float(balance.get('availableBalance', 0))
            if free > 0:
                logger.info(f"   {asset}: {free:.8f}")
        
        # Get account info
        logger.info("\n2. Account Info:")
        account = await exchange.get_account_info()
        logger.info(f"   Total Wallet Balance: {float(account.get('totalWalletBalance', 0)):.2f} USDT")
        logger.info(f"   Total Unrealized PnL: {float(account.get('totalUnrealizedProfit', 0)):.2f} USDT")
        logger.info(f"   Available Balance: {float(account.get('availableBalance', 0)):.2f} USDT")
        
        # Get positions
        logger.info("\n3. Open Positions:")
        positions = await exchange.get_position_risk()
        open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        if open_positions:
            for pos in open_positions:
                symbol = pos.get('symbol')
                amt = float(pos.get('positionAmt', 0))
                entry = float(pos.get('entryPrice', 0))
                mark = float(pos.get('markPrice', 0))
                pnl = float(pos.get('unRealizedProfit', 0))
                side = "LONG" if amt > 0 else "SHORT"
                pnl_emoji = "🟢" if pnl > 0 else "🔴"
                logger.info(f"   {symbol} {side}: {abs(amt):.4f} @ ${entry:,.2f}")
                logger.info(f"      Mark Price: ${mark:,.2f}")
                logger.info(f"      {pnl_emoji} PnL: ${pnl:,.2f}")
        else:
            logger.info("   No open positions")
        
        # Get open orders
        logger.info("\n4. Open Orders:")
        orders = await exchange.get_open_orders()
        if orders:
            for order in orders:
                symbol = order.get('symbol')
                side = order.get('side')
                order_type = order.get('type')
                price = float(order.get('price', 0))
                qty = float(order.get('origQty', 0))
                logger.info(f"   {symbol} {side} {order_type}: {qty:.4f} @ ${price:,.2f}")
        else:
            logger.info("   No open orders")
        
        # Get rate limit status
        logger.info("\n5. Rate Limit Status:")
        rate_status = exchange.get_rate_limit_status()
        logger.info(f"   Requests: {rate_status['request_count']}/{rate_status['requests_limit']}")
        logger.info(f"   Orders: {rate_status['order_count']}/{rate_status['orders_limit']}")
        logger.info(f"   Weight: {rate_status['weight_used']}/{rate_status['weight_limit']}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        await exchange.close()


async def demo_paper_trading():
    """Demo 4: Paper trading simulation (no real orders)"""
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Paper Trading Simulation")
    logger.info("="*60)
    
    config = {
        'api_key': os.getenv('ASTERDEX_API_KEY', 'test'),
        'api_secret': os.getenv('ASTERDEX_API_SECRET', 'test'),
        'base_url': os.getenv('ASTERDEX_BASE_URL', 'https://fapi.asterdex.com'),
        'leverage': 10,
        'margin_type': 'ISOLATED'
    }
    
    exchange = AsterDEXFutures(config)
    
    try:
        # Get current price
        ticker = await exchange.get_ticker_price('BTCUSDT')
        current_price = float(ticker.get('price', 0))
        logger.info(f"\n📊 Current BTC Price: ${current_price:,.2f}")
        
        # Simulate trading with $5 capital
        capital = 5.0
        leverage = 10
        position_size = capital * leverage  # $50 with 10x leverage
        
        logger.info(f"\n💰 Paper Trading Setup:")
        logger.info(f"   Capital: ${capital:.2f}")
        logger.info(f"   Leverage: {leverage}x")
        logger.info(f"   Position Size: ${position_size:.2f}")
        logger.info(f"   BTC Amount: {(position_size / current_price):.6f} BTC")
        
        # Simulate BUY order
        logger.info(f"\n🟢 Simulated BUY Order:")
        logger.info(f"   Symbol: BTCUSDT")
        logger.info(f"   Side: LONG")
        logger.info(f"   Entry: ${current_price:,.2f}")
        logger.info(f"   Quantity: {(position_size / current_price):.6f} BTC")
        logger.info(f"   Notional: ${position_size:.2f}")
        
        # Calculate profit scenarios
        logger.info(f"\n📈 Profit Scenarios:")
        
        # 1% profit
        profit_1pct = current_price * 1.01
        pnl_1pct = (profit_1pct - current_price) / current_price * position_size
        roi_1pct = (pnl_1pct / capital) * 100
        logger.info(f"   +1% move (${profit_1pct:,.2f}): PnL=${pnl_1pct:,.2f} ROI={roi_1pct:,.1f}%")
        
        # 2% profit
        profit_2pct = current_price * 1.02
        pnl_2pct = (profit_2pct - current_price) / current_price * position_size
        roi_2pct = (pnl_2pct / capital) * 100
        logger.info(f"   +2% move (${profit_2pct:,.2f}): PnL=${pnl_2pct:,.2f} ROI={roi_2pct:,.1f}%")
        
        # 5% profit
        profit_5pct = current_price * 1.05
        pnl_5pct = (profit_5pct - current_price) / current_price * position_size
        roi_5pct = (pnl_5pct / capital) * 100
        logger.info(f"   +5% move (${profit_5pct:,.2f}): PnL=${pnl_5pct:,.2f} ROI={roi_5pct:,.1f}%")
        
        logger.info(f"\n⚠️  Risk Warning:")
        logger.info(f"   -2% move would lose ${abs((current_price * 0.98 - current_price) / current_price * position_size):.2f}")
        logger.info(f"   -5% move would lose ${abs((current_price * 0.95 - current_price) / current_price * position_size):.2f}")
        logger.info(f"   -10% move would lose ${abs((current_price * 0.90 - current_price) / current_price * position_size):.2f} (100% of capital!)")
        
        logger.info(f"\n💡 Recommendation:")
        logger.info(f"   ✅ Always use stop-loss (recommend: 2-3%)")
        logger.info(f"   ✅ Start with lower leverage (5-10x)")
        logger.info(f"   ✅ Never risk more than 1-2% per trade")
        logger.info(f"   ✅ Use proper risk management!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        await exchange.close()


async def main():
    """Run all demos"""
    logger.info("="*60)
    logger.info("AsterDEX Futures Integration Demo")
    logger.info("="*60)
    
    # Demo 1: Connection test
    await demo_connection()
    await asyncio.sleep(1)
    
    # Demo 2: Market data (public, no API keys needed)
    await demo_market_data()
    await asyncio.sleep(1)
    
    # Demo 3: Account info (requires valid API keys)
    await demo_account_info()
    await asyncio.sleep(1)
    
    # Demo 4: Paper trading simulation
    await demo_paper_trading()
    
    logger.info("\n" + "="*60)
    logger.info("✅ All demos completed!")
    logger.info("="*60)
    logger.info("\nNext Steps:")
    logger.info("1. Set up valid API keys in .env file")
    logger.info("2. Test account info and balance")
    logger.info("3. Try paper trading with existing strategies")
    logger.info("4. Run backtests with historical data")
    logger.info("5. Start live trading with proper risk management!")


if __name__ == "__main__":
    asyncio.run(main())
