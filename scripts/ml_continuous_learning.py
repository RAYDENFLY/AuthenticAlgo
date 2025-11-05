"""
ML Continuous Learning dengan Auto-Screening
Belajar dari SEMUA coins di AsterDEX, bukan cuma top 5

Features:
- Auto-detect semua available symbols
- Smart filtering berdasarkan volume & volatility
- Priority training untuk coins dengan patterns menarik
- Adaptive learning berdasarkan market conditions
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import sqlite3
import json
from typing import Dict, List, Tuple
import aiohttp
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logger
from data.asterdex_collector import AsterDEXCollector
from ml.feature_engine import FeatureEngine
from ml.model_trainer import ModelTrainer

logger = setup_logger()


class MLContinuousLearnerEnhanced:
    """Enhanced ML learner dengan auto-screening semua coins"""
    
    def __init__(self, db_path: str = None):
        self.collector = AsterDEXCollector()
        self.feature_engine = FeatureEngine()
        self.trainer = ModelTrainer()
        
        # Database setup
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "database"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "ml_training_enhanced.db")
        
        self.db_path = db_path
        self.setup_enhanced_database()
        
        # CSV reports directory
        self.reports_dir = Path(__file__).parent.parent / "reports" / "ml_screening"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Screening results
        self.screening_results = []
        
        logger.info(f"🎯 ML Enhanced Learner dengan Auto-Screening")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Reports: {self.reports_dir}")
    
    def setup_enhanced_database(self):
        """Setup enhanced database untuk multi-coin screening"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Coin screening table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coin_screening (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL UNIQUE,
                current_price REAL,
                volume_24h REAL,
                price_change_24h REAL,
                volatility_score REAL,
                volume_score REAL,
                trend_score REAL,
                overall_score REAL,
                screening_rank INTEGER,
                recommendation TEXT,
                last_updated TEXT
            )
        """)
        
        # Pattern detection table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL,
                timeframe TEXT,
                description TEXT,
                potential_move REAL
            )
        """)
        
        # Market regime table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime_type TEXT NOT NULL,
                btc_dominance REAL,
                total_volume REAL,
                volatility_index REAL,
                trending_coins_count INTEGER,
                ranging_coins_count INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Enhanced database tables created")
    
    async def get_all_symbols(self) -> List[str]:
        """Get semua available symbols dari AsterDEX"""
        try:
            logger.info("📡 Fetching all available symbols from AsterDEX...")
            
            # Ganti dengan API call yang sesuai
            symbols = await self.collector.get_all_symbols()
            
            if not symbols:
                # Fallback: predefined symbols + auto-generate
                symbols = self._get_fallback_symbols()
            
            logger.info(f"✅ Found {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {e}")
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback symbols jika API tidak available"""
        # Major coins
        majors = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        
        # Mid-cap coins
        mid_caps = ["ADAUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT"]
        
        # Altcoins dengan volume decent
        alts = ["AVAXUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT", "ALGOUSDT"]
        
        # DeFi coins
        defi = ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT", "SUSHIUSDT"]
        
        # Meme coins & trending
        meme = ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT"]
        
        all_symbols = majors + mid_caps + alts + defi + meme
        return all_symbols
    
    async def screen_coin(self, symbol: str) -> Dict:
        """Screen individual coin untuk potential"""
        try:
            logger.debug(f"   Screening {symbol}...")
            
            # Fetch recent data
            data = await self.collector.fetch_klines(
                symbol=symbol,
                limit=100  # Cukup untuk screening
            )
            
            if data is None or len(data) < 50:
                return None
            
            # Calculate screening metrics
            metrics = self._calculate_screening_metrics(data, symbol)
            
            return metrics
            
        except Exception as e:
            logger.debug(f"   ❌ Failed to screen {symbol}: {e}")
            return None
    
    def _calculate_screening_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate screening metrics untuk coin"""
        try:
            df = data.copy()
            
            # Basic metrics
            current_price = df['close'].iloc[-1]
            volume_24h = df['volume'].tail(24).sum() if len(df) >= 24 else df['volume'].sum()
            price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100 if len(df) >= 24 else 0
            
            # Volatility score (ATR based)
            high_low_range = (df['high'] - df['low']) / df['close']
            volatility_score = high_low_range.rolling(window=20).mean().iloc[-1] * 100
            
            # Volume score (relative to average)
            volume_avg = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_score = (df['volume'].iloc[-1] / volume_avg) if volume_avg > 0 else 1
            
            # Trend score (multiple timeframe)
            trend_1h = self._calculate_trend_score(df, periods=[5, 10, 20])
            
            # Pattern detection
            patterns = self._detect_patterns(df)
            
            # Overall score (0-100)
            overall_score = self._calculate_overall_score(
                volatility_score, volume_score, trend_1h, patterns
            )
            
            # Recommendation
            recommendation = self._generate_recommendation(overall_score, patterns)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume_24h': volume_24h,
                'price_change_24h': price_change_24h,
                'volatility_score': volatility_score,
                'volume_score': volume_score,
                'trend_score': trend_1h,
                'overall_score': overall_score,
                'patterns': patterns,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return None
    
    def _calculate_trend_score(self, df: pd.DataFrame, periods: List[int]) -> float:
        """Calculate trend strength score"""
        try:
            scores = []
            for period in periods:
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean()
                    current_sma = sma.iloc[-1]
                    prev_sma = sma.iloc[-2] if len(sma) > 1 else current_sma
                    
                    # Score: 0.5 = sideways, >0.5 = uptrend, <0.5 = downtrend
                    trend_strength = (current_sma / prev_sma - 1) * 100
                    normalized_score = 0.5 + (trend_strength / 10)  # Normalize
                    scores.append(max(0, min(1, normalized_score)))
            
            return np.mean(scores) if scores else 0.5
            
        except:
            return 0.5
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect technical patterns"""
        patterns = []
        
        try:
            # Simple pattern detection
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Support/Resistance breakout
            if len(close) >= 20:
                resistance = np.max(high[-20:-5])  # Resistance dari 20 candles sebelumnya
                current_high = high[-1]
                
                if current_high > resistance:
                    patterns.append({
                        'type': 'RESISTANCE_BREAKOUT',
                        'confidence': 0.7,
                        'potential_move': (current_high / resistance - 1) * 100
                    })
            
            # Trend detection
            if len(close) >= 10:
                short_ma = np.mean(close[-5:])
                long_ma = np.mean(close[-10:])
                
                if short_ma > long_ma:
                    patterns.append({
                        'type': 'UPTREND',
                        'confidence': 0.6,
                        'potential_move': 5.0  # Conservative estimate
                    })
                else:
                    patterns.append({
                        'type': 'DOWNTREND', 
                        'confidence': 0.6,
                        'potential_move': -3.0
                    })
            
        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
        
        return patterns
    
    def _calculate_overall_score(self, volatility: float, volume: float, 
                               trend: float, patterns: List[Dict]) -> float:
        """Calculate overall screening score (0-100)"""
        try:
            # Weighted scoring
            volatility_weight = 0.3
            volume_weight = 0.3
            trend_weight = 0.2
            pattern_weight = 0.2
            
            # Normalize volatility (optimal range 2-10%)
            vol_score = max(0, 1 - abs(volatility - 6) / 10)
            
            # Volume score (higher is better, but not extreme)
            vol_mult_score = min(volume, 3) / 3  # Cap at 3x average
            
            # Trend score (already normalized)
            trend_score = trend
            
            # Pattern score
            pattern_score = 0
            if patterns:
                pattern_score = sum(p['confidence'] for p in patterns) / len(patterns)
            
            overall = (vol_score * volatility_weight + 
                      vol_mult_score * volume_weight + 
                      trend_score * trend_weight + 
                      pattern_score * pattern_weight)
            
            return overall * 100
            
        except:
            return 50.0
    
    def _generate_recommendation(self, score: float, patterns: List[Dict]) -> str:
        """Generate trading recommendation"""
        if score >= 80:
            return "STRONG_BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 60:
            return "WATCH"
        elif score >= 40:
            return "NEUTRAL"
        else:
            return "AVOID"
    
    async def run_mass_screening(self, max_concurrent: int = 10):
        """Run screening untuk semua coins"""
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MASS COIN SCREENING STARTED")
        logger.info("=" * 80)
        
        # Get all symbols
        symbols = await self.get_all_symbols()
        logger.info(f"📊 Screening {len(symbols)} coins...")
        
        # Run screening concurrently
        screening_results = []
        
        # Batch processing untuk avoid rate limits
        batch_size = max_concurrent
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"   Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            tasks = [self.screen_coin(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
            screening_results.extend(successful_results)
            
            # Small delay antara batches
            await asyncio.sleep(1)
        
        # Sort by overall score
        screening_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Save results
        self.screening_results = screening_results
        await self.save_screening_results(screening_results)
        
        logger.info(f"✅ Screening completed: {len(screening_results)}/{len(symbols)} coins analyzed")
        
        return screening_results
    
    async def save_screening_results(self, results: List[Dict]):
        """Save screening results ke database dan CSV"""
        try:
            # Save to database
            conn = sqlite3.connect(self.db_path)
            
            # Clear old results
            cursor = conn.cursor()
            cursor.execute("DELETE FROM coin_screening")
            
            # Insert new results
            for i, result in enumerate(results):
                cursor.execute("""
                    INSERT INTO coin_screening 
                    (timestamp, symbol, current_price, volume_24h, price_change_24h,
                     volatility_score, volume_score, trend_score, overall_score,
                     screening_rank, recommendation, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['timestamp'],
                    result['symbol'],
                    result['current_price'],
                    result['volume_24h'],
                    result['price_change_24h'],
                    result['volatility_score'],
                    result['volume_score'],
                    result['trend_score'],
                    result['overall_score'],
                    i + 1,  # rank
                    result['recommendation'],
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            # Save to CSV
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.reports_dir / f"coin_screening_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"💾 Screening results saved:")
            logger.info(f"   • Database: {len(results)} coins")
            logger.info(f"   • CSV: {csv_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Error saving screening results: {e}")
    
    def get_top_coins(self, top_n: int = 20, min_score: float = 60.0) -> List[Dict]:
        """Get top coins berdasarkan screening results"""
        if not self.screening_results:
            return []
        
        filtered = [coin for coin in self.screening_results 
                   if coin['overall_score'] >= min_score]
        
        return filtered[:top_n]
    
    async def smart_training_selection(self, top_n: int = 10):
        """Pilih coins terbaik untuk training berdasarkan screening"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 SMART TRAINING SELECTION")
        logger.info("=" * 80)
        
        top_coins = self.get_top_coins(top_n=top_n, min_score=65.0)
        
        if not top_coins:
            logger.warning("❌ No suitable coins found for training")
            return
        
        logger.info(f"🏆 Selected {len(top_coins)} coins for ML training:")
        for i, coin in enumerate(top_coins, 1):
            logger.info(f"   {i:2d}. {coin['symbol']:12} | Score: {coin['overall_score']:5.1f} | "
                       f"Rec: {coin['recommendation']:10} | "
                       f"Vol: {coin['volatility_score']:4.1f}%")
        
        # Train models untuk top coins
        from your_original_script import MLContinuousLearner
        
        base_learner = MLContinuousLearner()
        
        for coin in top_coins:
            symbol = coin['symbol']
            logger.info(f"\n🎓 Training ML models for {symbol}...")
            
            try:
                await base_learner.run_training_cycle(
                    symbol=symbol,
                    data_limit=1000
                )
            except Exception as e:
                logger.error(f"❌ Failed to train {symbol}: {e}")
            
            # Small delay antara training
            await asyncio.sleep(2)
    
    def generate_screening_report(self):
        """Generate comprehensive screening report"""
        if not self.screening_results:
            logger.warning("No screening results available")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 SCREENING SUMMARY REPORT")
        logger.info("=" * 80)
        
        df = pd.DataFrame(self.screening_results)
        
        # Statistics
        logger.info(f"\n📈 Screening Statistics:")
        logger.info(f"   Total coins analyzed: {len(df)}")
        logger.info(f"   Average score: {df['overall_score'].mean():.1f}")
        logger.info(f"   Best score: {df['overall_score'].max():.1f}")
        logger.info(f"   Worst score: {df['overall_score'].min():.1f}")
        
        # Recommendations breakdown
        rec_counts = df['recommendation'].value_counts()
        logger.info(f"\n🎯 Recommendations:")
        for rec, count in rec_counts.items():
            logger.info(f"   {rec:12}: {count:3d} coins")
        
        # Top 10 coins
        top_10 = df.head(10)
        logger.info(f"\n🏆 TOP 10 COINS:")
        logger.info("-" * 80)
        logger.info("Rank | Symbol      | Score | Recommendation | Volatility | Trend")
        logger.info("-" * 80)
        
        for i, (_, coin) in enumerate(top_10.iterrows(), 1):
            logger.info(f"{i:4} | {coin['symbol']:10} | {coin['overall_score']:5.1f} | "
                       f"{coin['recommendation']:13} | {coin['volatility_score']:10.1f} | "
                       f"{coin['trend_score']:5.1f}")


async def main_enhanced():
    """Enhanced main function dengan auto-screening"""
    learner = MLContinuousLearnerEnhanced()
    
    # Step 1: Run mass screening
    await learner.run_mass_screening(max_concurrent=15)
    
    # Step 2: Generate report
    learner.generate_screening_report()
    
    # Step 3: Smart training selection
    await learner.smart_training_selection(top_n=15)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ENHANCED ML SCREENING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main_enhanced())