"""
Stock Price Data Collection Module

This module handles downloading stock price data using yfinance API
and calculating returns and volatility metrics for the prediction system.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collects and processes stock price data"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the stock data collector
        
        Args:
            data_dir: Directory to store data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.data_dir / "stock_data.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing stock data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stock prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        # Returns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                return_1d REAL,
                return_5d REAL,
                return_9d REAL,
                volatility_5d REAL,
                spy_return_1d REAL,
                spy_return_5d REAL,
                spy_return_9d REAL,
                spy_volatility_5d REAL,
                relative_return_5d REAL,
                relative_return_9d REAL,
                relative_volatility_5d REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def get_sp500_tickers(self) -> List[str]:
        """
        Get current S&P 500 tickers from Wikipedia
        
        Returns:
            List of ticker symbols
        """
        try:
            # Get S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers (some may have dots)
            cleaned_tickers = []
            for ticker in tickers:
                # Keep original format for yfinance
                cleaned_tickers.append(ticker)
                
            logger.info(f"Retrieved {len(cleaned_tickers)} S&P 500 tickers")
            return cleaned_tickers
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 tickers: {e}")
            # Fallback to a sample for testing
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "PG"]
    
    def download_stock_data(self, tickers: List[str], start_date: str = "2014-01-01", 
                           end_date: str = "2024-12-31", batch_size: int = 10) -> None:
        """
        Download stock price data for given tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            batch_size: Number of tickers to process in each batch
        """
        logger.info(f"Starting download of stock data for {len(tickers)} tickers")
        
        # First download SPY data for benchmark
        logger.info("Downloading SPY benchmark data...")
        self._download_ticker_data("SPY", start_date, end_date)
        
        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            # Use ThreadPoolExecutor for parallel downloads
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._download_ticker_data, ticker, start_date, end_date): ticker 
                    for ticker in batch
                }
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        future.result()
                        logger.info(f"Successfully downloaded data for {ticker}")
                    except Exception as e:
                        logger.error(f"Error downloading data for {ticker}: {e}")
            
            # Pause between batches to avoid rate limiting
            if i + batch_size < len(tickers):
                logger.info("Pausing between batches...")
                time.sleep(1)
    
    def _download_ticker_data(self, ticker: str, start_date: str, end_date: str) -> None:
        """Download stock data for a specific ticker"""
        try:
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return
            
            # Prepare data for database
            data.reset_index(inplace=True)
            data['ticker'] = ticker
            data['date'] = data['Date'].dt.date
            
            # Rename columns to match database schema
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add adj_close if not present
            if 'Adj Close' in data.columns:
                data['adj_close'] = data['Adj Close']
            else:
                data['adj_close'] = data['close']
            
            # Select relevant columns
            columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            data = data[columns]
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            data.to_sql('stock_prices', conn, if_exists='append', index=False)
            conn.close()
            
            logger.info(f"Stored {len(data)} records for {ticker}")
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
    
    def calculate_returns(self, tickers: List[str] = None) -> None:
        """
        Calculate returns and volatility metrics
        
        Args:
            tickers: List of tickers to process (None for all)
        """
        logger.info("Calculating returns and volatility metrics...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get SPY data for benchmark calculations
        spy_data = pd.read_sql_query("""
            SELECT date, adj_close 
            FROM stock_prices 
            WHERE ticker = 'SPY' 
            ORDER BY date
        """, conn)
        
        if spy_data.empty:
            logger.error("SPY data not found. Cannot calculate relative returns.")
            conn.close()
            return
        
        # Calculate SPY returns
        spy_data['spy_return_1d'] = spy_data['adj_close'].pct_change()
        spy_data['spy_return_5d'] = spy_data['adj_close'].pct_change(periods=5)
        spy_data['spy_return_9d'] = spy_data['adj_close'].pct_change(periods=9)
        spy_data['spy_volatility_5d'] = spy_data['spy_return_1d'].rolling(window=5).std()
        
        # Get list of tickers to process
        if tickers is None:
            ticker_query = "SELECT DISTINCT ticker FROM stock_prices WHERE ticker != 'SPY'"
            tickers_df = pd.read_sql_query(ticker_query, conn)
            tickers = tickers_df['ticker'].tolist()
        
        for ticker in tickers:
            try:
                self._calculate_ticker_returns(ticker, spy_data, conn)
            except Exception as e:
                logger.error(f"Error calculating returns for {ticker}: {e}")
        
        conn.close()
        logger.info("Returns calculation completed")
    
    def _calculate_ticker_returns(self, ticker: str, spy_data: pd.DataFrame, conn) -> None:
        """Calculate returns for a specific ticker"""
        
        # Get stock data
        stock_data = pd.read_sql_query("""
            SELECT date, adj_close 
            FROM stock_prices 
            WHERE ticker = ? 
            ORDER BY date
        """, conn, params=(ticker,))
        
        if stock_data.empty:
            return
        
        # Calculate returns
        stock_data['return_1d'] = stock_data['adj_close'].pct_change()
        stock_data['return_5d'] = stock_data['adj_close'].pct_change(periods=5)
        stock_data['return_9d'] = stock_data['adj_close'].pct_change(periods=9)
        stock_data['volatility_5d'] = stock_data['return_1d'].rolling(window=5).std()
        
        # Merge with SPY data
        merged_data = stock_data.merge(spy_data, on='date', how='left')
        
        # Calculate relative metrics
        merged_data['relative_return_5d'] = merged_data['return_5d'] - merged_data['spy_return_5d']
        merged_data['relative_return_9d'] = merged_data['return_9d'] - merged_data['spy_return_9d']
        merged_data['relative_volatility_5d'] = merged_data['volatility_5d'] - merged_data['spy_volatility_5d']
        
        # Add ticker column
        merged_data['ticker'] = ticker
        
        # Select columns for returns table
        returns_columns = [
            'ticker', 'date', 'return_1d', 'return_5d', 'return_9d', 'volatility_5d',
            'spy_return_1d', 'spy_return_5d', 'spy_return_9d', 'spy_volatility_5d',
            'relative_return_5d', 'relative_return_9d', 'relative_volatility_5d'
        ]
        
        returns_data = merged_data[returns_columns].dropna()
        
        # Store in database
        returns_data.to_sql('stock_returns', conn, if_exists='append', index=False)
        
        logger.info(f"Calculated returns for {ticker}: {len(returns_data)} records")
    
    def get_stock_data_summary(self) -> pd.DataFrame:
        """Get summary of downloaded stock data"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                ticker,
                COUNT(*) as records,
                MIN(date) as start_date,
                MAX(date) as end_date,
                AVG(volume) as avg_volume
            FROM stock_prices 
            GROUP BY ticker
            ORDER BY ticker
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_returns_summary(self) -> pd.DataFrame:
        """Get summary of calculated returns"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                ticker,
                COUNT(*) as records,
                AVG(return_5d) as avg_5d_return,
                AVG(relative_return_5d) as avg_relative_5d_return,
                AVG(volatility_5d) as avg_volatility,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM stock_returns 
            GROUP BY ticker
            ORDER BY ticker
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_filing_aligned_returns(self, filing_date: str, ticker: str, 
                                  days_before: int = 4, days_after: int = 5) -> Dict:
        """
        Get returns aligned with filing dates
        
        Args:
            filing_date: Date of the filing (YYYY-MM-DD)
            ticker: Stock ticker
            days_before: Days before filing to include
            days_after: Days after filing to include
            
        Returns:
            Dictionary with return metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        # Convert filing date to datetime
        filing_dt = pd.to_datetime(filing_date)
        start_date = (filing_dt - timedelta(days=days_before + 5)).strftime('%Y-%m-%d')
        end_date = (filing_dt + timedelta(days=days_after + 5)).strftime('%Y-%m-%d')
        
        query = """
            SELECT date, return_1d, return_5d, return_9d, relative_return_5d, relative_return_9d
            FROM stock_returns 
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """
        
        data = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        conn.close()
        
        if data.empty:
            return {}
        
        # Find the closest trading day to filing date
        data['date'] = pd.to_datetime(data['date'])
        data['days_from_filing'] = (data['date'] - filing_dt).dt.days
        
        # Get returns for the specific periods
        filing_row = data.loc[data['days_from_filing'].abs().idxmin()]
        
        return {
            'filing_date': filing_date,
            'ticker': ticker,
            'closest_trading_date': filing_row['date'].strftime('%Y-%m-%d'),
            'days_from_filing': filing_row['days_from_filing'],
            'return_5d': filing_row['return_5d'],
            'return_9d': filing_row['return_9d'],
            'relative_return_5d': filing_row['relative_return_5d'],
            'relative_return_9d': filing_row['relative_return_9d']
        }


if __name__ == "__main__":
    # Test the stock data collector
    collector = StockDataCollector(data_dir="../data")
    
    # Test with a small sample
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("Testing stock data collector...")
    collector.download_stock_data(test_tickers, "2023-01-01", "2024-12-31", batch_size=3)
    
    # Calculate returns
    collector.calculate_returns(test_tickers)
    
    # Show summaries
    print("\nStock Data Summary:")
    print(collector.get_stock_data_summary())
    
    print("\nReturns Summary:")
    print(collector.get_returns_summary())

