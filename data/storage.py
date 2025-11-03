"""
Data Storage for saving and loading market data to/from database
"""

import sqlite3
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pandas as pd

from core import get_logger, get_config
from core.exceptions import DataError


class DataStorage:
    """Handles data storage and retrieval using SQLite"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize data storage
        
        Args:
            db_path: Path to SQLite database file (if None, uses default)
        """
        self.logger = get_logger()
        self.config = get_config()
        
        # Determine database path
        if db_path is None:
            base_path = Path(__file__).parent.parent
            db_path = base_path / "database" / "trading_bot.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_database()
        
        self.logger.info(f"DataStorage initialized with database: {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Create database tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            cursor = self.conn.cursor()
            
            # Create OHLCV table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp
                ON ohlcv(symbol, timeframe, timestamp)
            """)
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    status TEXT NOT NULL,
                    strategy TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    unrealized_pnl_pct REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    opened_at DATETIME NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, side)
                )
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    total_profit_loss REAL NOT NULL,
                    balance REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            self.conn.commit()
            self.logger.info("Database tables initialized successfully")
            
        except sqlite3.Error as e:
            error_msg = f"Database initialization error: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        replace: bool = False
    ) -> int:
        """
        Save OHLCV data to database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            df: DataFrame with OHLCV data
            replace: If True, replace existing data; if False, skip duplicates
            
        Returns:
            Number of rows inserted
        """
        try:
            if df.empty:
                self.logger.warning("Empty DataFrame provided, nothing to save")
                return 0
            
            # Prepare data
            df_copy = df.copy()
            if 'timestamp' not in df_copy.columns:
                df_copy.reset_index(inplace=True)
            
            # Convert timestamp to string format for SQL
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']).astype(str)
            
            df_copy['symbol'] = symbol
            df_copy['timeframe'] = timeframe
            
            # Select relevant columns
            columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_copy = df_copy[columns]
            
            # Insert data
            cursor = self.conn.cursor()
            rows_inserted = 0
            
            for _, row in df_copy.iterrows():
                try:
                    if replace:
                        cursor.execute("""
                            INSERT OR REPLACE INTO ohlcv 
                            (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, tuple(row))
                    else:
                        cursor.execute("""
                            INSERT OR IGNORE INTO ohlcv 
                            (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, tuple(row))
                    
                    if cursor.rowcount > 0:
                        rows_inserted += 1
                        
                except sqlite3.Error as e:
                    self.logger.warning(f"Error inserting row: {e}")
                    continue
            
            self.conn.commit()
            self.logger.info(f"Saved {rows_inserted} rows for {symbol} ({timeframe})")
            return rows_inserted
            
        except Exception as e:
            error_msg = f"Error saving OHLCV data: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            # Add date filters
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            # Add ordering and limit
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            df = pd.read_sql_query(query, self.conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Loaded {len(df)} rows for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            error_msg = f"Error loading OHLCV data: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def save_trade(self, trade_data: dict) -> int:
        """
        Save trade record to database
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            Trade ID
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades 
                (symbol, side, type, entry_price, exit_price, quantity, 
                 entry_time, exit_time, profit_loss, profit_loss_pct, 
                 status, strategy, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('type'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('quantity'),
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                trade_data.get('profit_loss'),
                trade_data.get('profit_loss_pct'),
                trade_data.get('status', 'open'),
                trade_data.get('strategy'),
                trade_data.get('notes'),
            ))
            
            self.conn.commit()
            trade_id = cursor.lastrowid
            
            self.logger.info(f"Saved trade {trade_id} for {trade_data.get('symbol')}")
            return trade_id
            
        except Exception as e:
            error_msg = f"Error saving trade: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get trade history
        
        Args:
            symbol: Filter by symbol
            status: Filter by status (open, closed)
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame with trade history
        """
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY entry_time DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, self.conn, params=params)
            
            self.logger.info(f"Retrieved {len(df)} trades")
            return df
            
        except Exception as e:
            error_msg = f"Error getting trades: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def clear_ohlcv(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear OHLCV data
        
        Args:
            symbol: Clear specific symbol (if None, clear all)
            timeframe: Clear specific timeframe (if None, clear all)
        """
        try:
            cursor = self.conn.cursor()
            
            if symbol and timeframe:
                cursor.execute("DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
            elif symbol:
                cursor.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
            elif timeframe:
                cursor.execute("DELETE FROM ohlcv WHERE timeframe = ?", (timeframe,))
            else:
                cursor.execute("DELETE FROM ohlcv")
            
            self.conn.commit()
            self.logger.info(f"Cleared OHLCV data (symbol={symbol}, timeframe={timeframe})")
            
        except Exception as e:
            error_msg = f"Error clearing OHLCV data: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def get_statistics(self) -> dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Count OHLCV records
            cursor.execute("SELECT COUNT(*) FROM ohlcv")
            ohlcv_count = cursor.fetchone()[0]
            
            # Count trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            trades_count = cursor.fetchone()[0]
            
            # Get symbols
            cursor.execute("SELECT DISTINCT symbol FROM ohlcv")
            symbols = [row[0] for row in cursor.fetchall()]
            
            stats = {
                'ohlcv_records': ohlcv_count,
                'total_trades': trades_count,
                'symbols': symbols,
                'database_path': str(self.db_path),
            }
            
            return stats
            
        except Exception as e:
            error_msg = f"Error getting statistics: {e}"
            self.logger.error(error_msg)
            raise DataError(error_msg)
    
    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __repr__(self) -> str:
        return f"DataStorage(db_path='{self.db_path}')"
