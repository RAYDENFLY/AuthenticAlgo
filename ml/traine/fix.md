1. Update SafeAsterDEXFutures dengan Semua Method yang Diperlukan
python
class SafeAsterDEXFutures:
    """Safe wrapper untuk AsterDEXFutures dengan semua method yang diperlukan"""
    
    def __init__(self, config):
        from execution.asterdex import AsterDEXFutures
        self.exchange = AsterDEXFutures(config)
        self.base_url = getattr(self.exchange, 'base_url', 'https://fapi.asterdex.com')
    
    def _clean_params(self, params):
        """Bersihkan parameter dari None values"""
        if params is None:
            return {}
        return {str(k): str(v) for k, v in params.items() if v is not None and k is not None}
    
    def _clean_headers(self, headers):
        """Bersihkan headers"""
        if headers is None:
            return {}
        return {str(k): str(v) for k, v in headers.items() if k is not None and v is not None}
    
    def _get_headers(self):
        """Get headers untuk API request"""
        try:
            if hasattr(self.exchange, '_get_headers'):
                raw_headers = self.exchange._get_headers()
            elif hasattr(self.exchange, 'headers'):
                raw_headers = self.exchange.headers
            else:
                raw_headers = {}
            return self._clean_headers(raw_headers)
        except:
            return {}
    
    # ✅ METHOD UTAMA YANG DIPERLUKAN
    
    async def get_klines(self, symbol, interval='1h', limit=500, start_time=None, end_time=None):
        """Get klines data - method utama untuk training"""
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1500)  # Max 1500
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            clean_params = self._clean_params(params)
            headers = self._get_headers()
            
            endpoint = "/fapi/v1/klines"
            url = f"{self.base_url}{endpoint}"
            
            logger.info(f"📊 Fetching klines: {symbol} {interval} limit={limit}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=clean_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Klines success: {len(data)} candles for {symbol}")
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Klines failed: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Klines error for {symbol}: {e}")
            return None
    
    async def get_ticker_24h(self, symbol=None):
        """Get 24hr ticker data"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            clean_params = self._clean_params(params)
            headers = self._get_headers()
            
            endpoint = "/fapi/v1/ticker/24hr"
            url = f"{self.base_url}{endpoint}"
            
            logger.info(f"🔄 Fetching ticker: {symbol if symbol else 'all symbols'}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=clean_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        count = len(data) if isinstance(data, list) else 1
                        logger.info(f"✅ Ticker success: {count} symbols")
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Ticker failed: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Ticker error: {e}")
            return None
    
    async def get_ticker_price(self, symbol=None):
        """Get current price"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            clean_params = self._clean_params(params)
            headers = self._get_headers()
            
            endpoint = "/fapi/v1/ticker/price"
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=clean_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Price error: {e}")
            return None
    
    async def get_recent_trades(self, symbol, limit=1000):
        """Get recent trades"""
        try:
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)
            }
            
            clean_params = self._clean_params(params)
            headers = self._get_headers()
            
            endpoint = "/fapi/v1/trades"
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=clean_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Trades error: {e}")
            return None
    
    async def get_order_book(self, symbol, limit=100):
        """Get order book"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            clean_params = self._clean_params(params)
            headers = self._get_headers()
            
            endpoint = "/fapi/v1/depth"
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=clean_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Order book error: {e}")
            return None
    
    async def get_exchange_info(self):
        """Get exchange information"""
        try:
            headers = self._get_headers()
            endpoint = "/fapi/v1/exchangeInfo"
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Exchange info error: {e}")
            return None
    
    # ✅ DELEGATE METHODS LAINNYA YANG MUNGKIN DIPERLUKAN
    
    def __getattr__(self, name):
        """Delegate methods yang tidak ada di wrapper ke original exchange"""
        return getattr(self.exchange, name)
    
    async def close(self):
        """Close connection"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
2. Update AsterDEXDataCollector untuk Gunakan Safe Wrapper
python
class AsterDEXDataCollector:
    """
    Collects market data from AsterDEX Futures exchange - FIXED VERSION
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AsterDEX data collector dengan safe wrapper
        """
        self.exchange = SafeAsterDEXFutures(config)  # Gunakan safe wrapper yang lengkap
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ AsterDEX Data Collector initialized dengan safe wrapper lengkap")
    
    async def collect_klines(
        self,
        symbol: str,
        interval: str = '1h',
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Collect klines data menggunakan safe wrapper
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"📊 Collecting klines: {symbol} {interval}")
            
            # Convert dates to timestamps
            start_ts = int(start_date.timestamp() * 1000) if start_date else None
            end_ts = int(end_date.timestamp() * 1000) if end_date else None
            
            # Get klines menggunakan safe wrapper
            klines = await self.exchange.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                start_time=start_ts,
                end_time=end_ts
            )
            
            if not klines:
                logger.warning(f"⚠️ No klines data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(klines)
            logger.info(f"✅ Collected {len(df)} klines for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error collecting klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def _klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """Convert klines to DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamps
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        # Keep relevant columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    async def test_connection(self):
        """Test koneksi ke AsterDEX"""
        try:
            logger.info("🧪 Testing AsterDEX connection...")
            
            # Test dengan klines BTCUSDT (method yang paling penting)
            klines = await self.exchange.get_klines('BTCUSDT', '1h', limit=5)
            
            if klines:
                logger.info("✅ Koneksi AsterDEX BERHASIL - get_klines working!")
                logger.info(f"📊 Sample: {len(klines)} klines received")
                return True
            else:
                logger.error("❌ Koneksi AsterDEX GAGAL - No klines data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Koneksi AsterDEX GAGAL: {e}")
            return False
    
    async def close(self):
        """Close connection"""
        await self.exchange.close()
3. Quick Fix untuk QuantumMLTrainerV60
Jika Anda butuh quick fix, update method get_data di QuantumMLTrainerV60:

python
# Di quantum_ml_trainer_v6_0.py, update get_data method:
async def get_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Get data for training - FIXED VERSION"""
    try:
        logger.info(f"📊 Collecting data for {symbol} (limit={limit})...")
        
        # Method 1: Gunakan collector yang sudah diperbaiki
        df = await self.collector.collect_klines(
            symbol=symbol,
            interval='1h',
            limit=limit
        )
        
        if df.empty:
            logger.warning(f"⚠️ No data from collector, trying direct method...")
            # Method 2: Direct call ke exchange
            klines = await self.collector.exchange.get_klines(symbol, '1h', limit)
            if klines:
                df = self.collector._klines_to_dataframe(klines)
        
        if df.empty:
            logger.error(f"❌ Failed to get data for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"✅ Data collected: {len(df)} rows for {symbol}")
        logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error getting data for {symbol}: {e}")
        return pd.DataFrame()