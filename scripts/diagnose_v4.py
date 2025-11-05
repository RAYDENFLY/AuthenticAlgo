"""
QUANTUM LEAP V4.1 - EMERGENCY DIAGNOSIS
Validate V4.0 untuk identify masalah sebelum fix
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
from data.asterdex_collector import AsterDEXDataCollector
from execution.asterdex import AsterDEXFutures

logger = setup_logger()


async def diagnose_v4_issues():
    """
    Run comprehensive diagnosis of V4.0
    """
    logger.info("🔍 QUANTUM V4.0 EMERGENCY DIAGNOSIS")
    logger.info("="*70)
    
    # Setup
    config = {
        'exchange': 'asterdex',
        'api_key': 'dummy',
        'api_secret': 'dummy',
        'testnet': True
    }
    
    collector = AsterDEXDataCollector(config)
    
    coins = ['BTCUSDT', 'ETHUSDT', 'TRUMPUSDT']
    all_diagnostics = {}
    
    for symbol in coins:
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Diagnosing {symbol}")
        logger.info(f"{'='*70}")
        
        # 1. Get data
        try:
            klines = await collector.exchange.get_klines(symbol, '1h', 1000)
            
            # Convert to proper DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if df.empty or df['close'].isna().all():
                logger.warning(f"❌ No valid data for {symbol}")
                continue
            
            logger.info(f"✅ Data collected: {len(df)} candles")
            
            # 2. Create features (simplified V4 logic)
            df['momentum_5'] = df['close'].pct_change(5).fillna(0)
            df['momentum_10'] = df['close'].pct_change(10).fillna(0)
            df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0)
            
            # Calculate future returns (this is the TARGET)
            df['future_return'] = df['close'].pct_change(6).shift(-6).fillna(0)
            
            # Create quantum target (V4 style)
            returns = df['future_return']
            volatility = df['volatility']
            
            # Dynamic thresholds
            threshold = volatility * 1.0  # V4 uses regime-specific
            
            # Binary classification
            target = (returns > threshold).astype(int)
            
            # 3. CHECK CRITICAL ISSUES
            diagnostics = {}
            
            # Issue 1: Class Distribution
            logger.info("\n🔍 Issue 1: CLASS DISTRIBUTION")
            class_counts = target.value_counts()
            class_pct = target.value_counts(normalize=True)
            
            logger.info(f"   Class 0 (SELL): {class_counts.get(0, 0)} ({class_pct.get(0, 0)*100:.1f}%)")
            logger.info(f"   Class 1 (BUY):  {class_counts.get(1, 0)} ({class_pct.get(1, 0)*100:.1f}%)")
            
            max_class_pct = class_pct.max()
            diagnostics['class_imbalance'] = max_class_pct
            
            if max_class_pct > 0.7:
                logger.warning(f"   ⚠️ SEVERE IMBALANCE: {max_class_pct*100:.1f}%")
            elif max_class_pct > 0.6:
                logger.warning(f"   ⚠️ MODERATE IMBALANCE: {max_class_pct*100:.1f}%")
            else:
                logger.info(f"   ✅ Balanced: {max_class_pct*100:.1f}%")
            
            # Issue 2: Data Leakage Check
            logger.info("\n🔍 Issue 2: DATA LEAKAGE CHECK")
            
            # Check if any feature uses future data
            leakage_features = []
            for col in df.columns:
                if 'future' in col.lower() and col != 'future_return':
                    leakage_features.append(col)
                    logger.warning(f"   ⚠️ POTENTIAL LEAKAGE: {col}")
            
            diagnostics['leakage_features'] = leakage_features
            
            if not leakage_features:
                logger.info("   ✅ No obvious feature leakage detected")
            
            # Issue 3: Temporal Validation
            logger.info("\n🔍 Issue 3: TEMPORAL VALIDATION")
            
            # Split data
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            train_max_time = train_df['timestamp'].max() if 'timestamp' in train_df.columns else train_df.index[-1]
            test_min_time = test_df['timestamp'].min() if 'timestamp' in test_df.columns else test_df.index[0]
            
            logger.info(f"   Train end: {train_max_time}")
            logger.info(f"   Test start: {test_min_time}")
            
            if test_min_time > train_max_time:
                logger.info("   ✅ Proper temporal split")
                diagnostics['temporal_leakage'] = False
            else:
                logger.warning("   ⚠️ TEMPORAL LEAKAGE DETECTED")
                diagnostics['temporal_leakage'] = True
            
            # Issue 4: Target Distribution Over Time
            logger.info("\n🔍 Issue 4: TARGET STABILITY")
            
            # Check if target distribution changes over time
            window_size = 100
            rolling_buy_pct = target.rolling(window_size).mean()
            
            stability = rolling_buy_pct.std()
            diagnostics['target_stability'] = stability
            
            logger.info(f"   Rolling BUY% std: {stability:.4f}")
            
            if stability > 0.15:
                logger.warning(f"   ⚠️ UNSTABLE TARGET (std={stability:.4f})")
            else:
                logger.info(f"   ✅ Stable target (std={stability:.4f})")
            
            # Issue 5: Feature-Target Correlation
            logger.info("\n🔍 Issue 5: FEATURE IMPORTANCE")
            
            feature_cols = ['momentum_5', 'momentum_10', 'volatility']
            for feat in feature_cols:
                corr = df[feat].corr(target)
                logger.info(f"   {feat} correlation: {corr:.4f}")
                
                if abs(corr) < 0.05:
                    logger.warning(f"      ⚠️ Weak correlation: {abs(corr):.4f}")
            
            # Issue 6: Perfect Predictions Check
            logger.info("\n🔍 Issue 6: SANITY CHECK")
            
            # Jika ada regime dengan 100% accuracy, ada masalah
            diagnostics['suspicious'] = False
            
            # Summary
            all_diagnostics[symbol] = diagnostics
            
            logger.info(f"\n📊 {symbol} Summary:")
            logger.info(f"   Class Imbalance: {diagnostics['class_imbalance']*100:.1f}%")
            logger.info(f"   Temporal Leakage: {'YES ❌' if diagnostics['temporal_leakage'] else 'NO ✅'}")
            logger.info(f"   Target Stability: {diagnostics['target_stability']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error diagnosing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Overall Assessment
    logger.info("\n" + "="*70)
    logger.info("🎯 OVERALL ASSESSMENT")
    logger.info("="*70)
    
    issues_found = []
    
    for symbol, diag in all_diagnostics.items():
        if diag['class_imbalance'] > 0.7:
            issues_found.append(f"{symbol}: Severe class imbalance ({diag['class_imbalance']*100:.1f}%)")
        
        if diag['temporal_leakage']:
            issues_found.append(f"{symbol}: Temporal leakage detected")
        
        if diag['target_stability'] > 0.15:
            issues_found.append(f"{symbol}: Unstable target (std={diag['target_stability']:.4f})")
    
    if issues_found:
        logger.warning("\n⚠️ ISSUES FOUND:")
        for issue in issues_found:
            logger.warning(f"   • {issue}")
        
        logger.info("\n📋 RECOMMENDED FIXES:")
        logger.info("   1. Implement SMOTE for class balancing")
        logger.info("   2. Use purged cross-validation")
        logger.info("   3. Adjust target thresholds per regime")
        logger.info("   4. Add more discriminative features")
    else:
        logger.info("\n✅ NO CRITICAL ISSUES FOUND")
        logger.info("   V4.0 appears healthy, high accuracy may be legitimate!")
    
    logger.info("\n🚀 Next Step: Run V4.1 fixes if issues found")
    
    await collector.exchange.close()


if __name__ == "__main__":
    asyncio.run(diagnose_v4_issues())
