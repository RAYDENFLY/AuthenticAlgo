"""
Execution Module Demo
Demonstrates exchange connection, position sizing, and order management
"""

import asyncio
from core.logger import logger
from execution import (
    create_exchange,
    OrderManager,
    PositionSizer,
    OrderSide,
    SizingMethod
)


async def demo_exchange_connection():
    """Demo: Connect to AsterDEX and fetch account info"""
    logger.info("=" * 70)
    logger.info("🔌 DEMO 1: Exchange Connection")
    logger.info("=" * 70)
    
    try:
        # Create exchange instance (testnet for safety)
        exchange = create_exchange("asterdex", testnet=True)
        
        # Connect to exchange
        logger.info("\n📡 Connecting to AsterDEX (testnet)...")
        await exchange.connect()
        
        # Fetch balance
        logger.info("\n💰 Fetching account balance...")
        balances = await exchange.fetch_balance()
        
        logger.info(f"\n✅ Connected! Found {len(balances)} assets:")
        for asset, balance in list(balances.items())[:5]:  # Show first 5
            logger.info(
                f"  {asset}: {balance.total:.4f} "
                f"(Free: {balance.free:.4f}, Locked: {balance.locked:.4f})"
            )
        
        # Fetch open orders
        logger.info("\n📋 Fetching open orders...")
        open_orders = await exchange.get_open_orders()
        logger.info(f"Open orders: {len(open_orders)}")
        
        # Fetch positions (futures)
        logger.info("\n📊 Fetching open positions...")
        positions = await exchange.get_positions()
        logger.info(f"Open positions: {len(positions)}")
        
        for position in positions[:3]:  # Show first 3
            logger.info(
                f"  {position.symbol}: {position.side.value} "
                f"{position.size} @ ${position.entry_price:.2f} "
                f"(PnL: ${position.unrealized_pnl:.2f})"
            )
        
        await exchange.close()
        logger.info("\n✅ Exchange connection demo complete!")
        
    except Exception as e:
        logger.error(f"Exchange connection failed: {e}")
        logger.info("💡 Tip: Make sure API keys are configured in .env")


def demo_position_sizing():
    """Demo: Calculate position sizes using different methods"""
    logger.info("\n" + "=" * 70)
    logger.info("📏 DEMO 2: Position Sizing Strategies")
    logger.info("=" * 70)
    
    # Initialize position sizer
    account_balance = 10000  # $10,000 account
    sizer = PositionSizer(
        account_balance=account_balance,
        max_risk_percent=2.0,      # Risk max 2% per trade
        max_position_percent=10.0,  # Max 10% position size
        leverage=3                  # 3x leverage
    )
    
    current_price = 50000  # BTC price
    
    logger.info(f"\n📊 Account: ${account_balance:,.2f}")
    logger.info(f"📊 Asset Price: ${current_price:,.2f} (BTC)")
    logger.info(f"📊 Settings: Max Risk={sizer.max_risk_percent}%, Max Position={sizer.max_position_percent}%, Leverage={sizer.leverage}x")
    
    # Method 1: Fixed Percentage
    logger.info("\n" + "-" * 70)
    logger.info("1️⃣ FIXED PERCENTAGE SIZING")
    logger.info("-" * 70)
    
    size1 = sizer.fixed_percentage(current_price, position_percent=5.0)
    logger.info(f"Result: {size1}")
    logger.info(f"  → Buy {size1.quantity:.4f} BTC")
    logger.info(f"  → Position Value: ${size1.position_value:,.2f}")
    logger.info(f"  → Risk: ${size1.risk_amount:,.2f} ({size1.risk_percent:.2f}%)")
    
    # Method 2: Kelly Criterion
    logger.info("\n" + "-" * 70)
    logger.info("2️⃣ KELLY CRITERION SIZING")
    logger.info("-" * 70)
    
    size2 = sizer.kelly_criterion(
        current_price=current_price,
        win_rate=0.55,    # 55% win rate
        avg_win=500,      # $500 average win
        avg_loss=300      # $300 average loss
    )
    logger.info(f"Stats: WinRate=55%, AvgWin=$500, AvgLoss=$300")
    logger.info(f"Result: {size2}")
    logger.info(f"  → Buy {size2.quantity:.4f} BTC")
    logger.info(f"  → Position Value: ${size2.position_value:,.2f}")
    
    # Method 3: Volatility-Based
    logger.info("\n" + "-" * 70)
    logger.info("3️⃣ VOLATILITY-BASED SIZING")
    logger.info("-" * 70)
    
    size3 = sizer.volatility_based(
        current_price=current_price,
        volatility=0.04,         # 4% volatility
        target_volatility=0.02   # Target 2% volatility
    )
    logger.info(f"Volatility: Current=4%, Target=2%")
    logger.info(f"Result: {size3}")
    logger.info(f"  → Buy {size3.quantity:.4f} BTC")
    logger.info(f"  → Adjusted for volatility")
    
    # Method 4: Risk-Based (with stop-loss)
    logger.info("\n" + "-" * 70)
    logger.info("4️⃣ RISK-BASED SIZING (with Stop-Loss)")
    logger.info("-" * 70)
    
    stop_loss_price = 48000  # Stop at $48k
    size4 = sizer.risk_based(
        current_price=current_price,
        stop_loss_price=stop_loss_price,
        risk_percent=2.0
    )
    logger.info(f"Entry: ${current_price:,.2f}, Stop-Loss: ${stop_loss_price:,.2f}")
    logger.info(f"Price Risk: ${current_price - stop_loss_price:,.2f} per unit")
    logger.info(f"Result: {size4}")
    logger.info(f"  → Buy {size4.quantity:.4f} BTC")
    logger.info(f"  → If stop hit, lose ${size4.risk_amount:,.2f} ({size4.risk_percent:.2f}%)")
    
    # Method 5: ATR-Based
    logger.info("\n" + "-" * 70)
    logger.info("5️⃣ ATR-BASED SIZING")
    logger.info("-" * 70)
    
    size5 = sizer.atr_based(
        current_price=current_price,
        atr=800,            # ATR = $800
        atr_multiplier=2.0  # 2x ATR stop
    )
    logger.info(f"ATR: $800, Stop Distance: {2.0 * 800} (2x ATR)")
    logger.info(f"Result: {size5}")
    logger.info(f"  → Buy {size5.quantity:.4f} BTC")
    logger.info(f"  → Dynamic stop based on ATR")
    
    # Validation
    logger.info("\n" + "-" * 70)
    logger.info("✅ VALIDATION")
    logger.info("-" * 70)
    
    for i, size in enumerate([size1, size2, size3, size4, size5], 1):
        valid = sizer.validate_position_size(size)
        status = "✅ Valid" if valid else "❌ Invalid"
        logger.info(f"  Method {i}: {status}")
    
    # Statistics
    logger.info("\n" + "-" * 70)
    logger.info("📊 POSITION SIZER STATISTICS")
    logger.info("-" * 70)
    
    stats = sizer.get_statistics()
    logger.info(f"  Balance: ${stats['account_balance']:,.2f}")
    logger.info(f"  Max Risk per Trade: ${stats['max_risk_amount']:,.2f} ({stats['max_risk_percent']}%)")
    logger.info(f"  Max Position Value: ${stats['max_position_value']:,.2f} ({stats['max_position_percent']}%)")
    logger.info(f"  Leverage: {stats['leverage']}x")
    
    logger.info("\n✅ Position sizing demo complete!")


async def demo_order_manager():
    """Demo: Order management (without actual exchange calls)"""
    logger.info("\n" + "=" * 70)
    logger.info("📋 DEMO 3: Order Manager (Simulation)")
    logger.info("=" * 70)
    
    try:
        # Create exchange (testnet)
        exchange = create_exchange("asterdex", testnet=True)
        await exchange.connect()
        
        # Create order manager
        order_manager = OrderManager(
            exchange=exchange,
            enable_validation=True,
            max_slippage_percent=1.0
        )
        
        logger.info("\n✅ OrderManager initialized")
        
        # Register callbacks
        async def on_order_filled(order):
            logger.info(f"🎉 CALLBACK: Order {order.order.order_id} filled!")
        
        order_manager.register_callback('order_filled', on_order_filled)
        logger.info("✅ Order callbacks registered")
        
        # Get statistics
        stats = order_manager.get_statistics()
        logger.info(f"\n📊 Order Statistics:")
        logger.info(f"  Total Orders: {stats['total_orders']}")
        logger.info(f"  Active Orders: {stats['active_orders']}")
        logger.info(f"  Fill Rate: {stats['fill_rate']:.2%}")
        
        # Note: Actual order placement requires testnet balance
        logger.info("\n💡 To place actual orders:")
        logger.info("  1. Ensure testnet API keys are configured")
        logger.info("  2. Fund testnet account")
        logger.info("  3. Use order_manager.place_market_order() / place_limit_order()")
        
        await exchange.close()
        logger.info("\n✅ Order manager demo complete!")
        
    except Exception as e:
        logger.error(f"Order manager demo failed: {e}")


async def main():
    """Run all demos"""
    logger.info("🚀 Starting Execution Module Demo")
    logger.info("=" * 70)
    
    # Demo 1: Exchange connection
    await demo_exchange_connection()
    
    # Demo 2: Position sizing (no API calls needed)
    demo_position_sizing()
    
    # Demo 3: Order manager
    await demo_order_manager()
    
    logger.info("\n" + "=" * 70)
    logger.info("✨ Phase 3 Execution Module: COMPLETE! ✨")
    logger.info("=" * 70)
    
    logger.info("\n📝 What we built:")
    logger.info("  ✅ BaseExchange - Abstract exchange interface")
    logger.info("  ✅ AsterDEXExchange - AsterDEX implementation with fallbacks")
    logger.info("  ✅ OrderManager - Order lifecycle management")
    logger.info("  ✅ PositionSizer - 5 position sizing strategies")
    logger.info("  ✅ Rate limiting and error handling")
    logger.info("  ✅ Testnet support for safe testing")
    
    logger.info("\n📊 Position Sizing Methods:")
    logger.info("  1. Fixed Percentage - Simple percentage of balance")
    logger.info("  2. Kelly Criterion - Mathematical edge-based sizing")
    logger.info("  3. Volatility-Based - Adjust for market volatility")
    logger.info("  4. Risk-Based - Size to risk% with stop-loss")
    logger.info("  5. ATR-Based - Dynamic stops using ATR")
    
    logger.info("\n🚀 Ready for Phase 4: Trading Strategies!")


if __name__ == "__main__":
    asyncio.run(main())
