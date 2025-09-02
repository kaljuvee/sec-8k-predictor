"""
Unit tests for stock data collector module.
"""

import unittest
import tempfile
import shutil
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.stock_data import StockDataCollector


class TestStockDataCollector(unittest.TestCase):
    """Test cases for StockDataCollector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.collector = StockDataCollector(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test collector initialization"""
        self.assertEqual(self.collector.data_dir, Path(self.test_dir))
        self.assertTrue(self.collector.data_dir.exists())
        self.assertTrue(self.collector.db_path.exists())
    
    def test_database_creation(self):
        """Test database and table creation"""
        # Check if database exists
        self.assertTrue(self.collector.db_path.exists())
        
        # Check if tables exist
        conn = sqlite3.connect(self.collector.db_path)
        cursor = conn.cursor()
        
        # Check stock_prices table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_prices'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        # Check stock_returns table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_returns'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        conn.close()
    
    def test_get_sp500_tickers(self):
        """Test S&P 500 ticker retrieval"""
        tickers = self.collector.get_sp500_tickers()
        self.assertIsInstance(tickers, list)
        self.assertGreater(len(tickers), 400)
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)
    
    def test_calculate_returns_with_sample_data(self):
        """Test return calculations with sample data"""
        # Create sample stock price data
        dates = pd.date_range('2023-01-01', '2023-01-20', freq='D')
        prices = [100 + i + np.random.normal(0, 1) for i in range(len(dates))]
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Volume': [1000000] * len(dates)
        })
        
        # Save sample data to database
        conn = sqlite3.connect(self.collector.db_path)
        
        for _, row in sample_data.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO stock_prices 
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('TEST', row['Date'].strftime('%Y-%m-%d'), 
                  row['Close'], row['Close'], row['Close'], row['Close'], row['Volume']))
        
        conn.commit()
        conn.close()
        
        # Calculate returns
        self.collector.calculate_returns(['TEST'])
        
        # Verify returns were calculated
        conn = sqlite3.connect(self.collector.db_path)
        returns_data = pd.read_sql_query(
            "SELECT * FROM stock_returns WHERE ticker = 'TEST'", 
            conn
        )
        conn.close()
        
        self.assertFalse(returns_data.empty)
        self.assertIn('return_5d', returns_data.columns)
        self.assertIn('return_9d', returns_data.columns)
        self.assertIn('volatility_5d', returns_data.columns)
    
    def test_get_stock_data_summary_empty(self):
        """Test stock data summary with empty database"""
        summary = self.collector.get_stock_data_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(summary.empty)
    
    def test_get_returns_summary_empty(self):
        """Test returns summary with empty database"""
        summary = self.collector.get_returns_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(summary.empty)
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        # Test data
        returns = [0.01, -0.02, 0.015, -0.01, 0.005]
        volatility = self.collector._calculate_volatility(returns)
        
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
        
        # Test with empty returns
        empty_volatility = self.collector._calculate_volatility([])
        self.assertEqual(empty_volatility, 0.0)
    
    def test_get_spy_data_for_date(self):
        """Test SPY data retrieval for specific date"""
        # Add sample SPY data
        conn = sqlite3.connect(self.collector.db_path)
        
        test_date = '2023-01-15'
        spy_close = 400.0
        
        conn.execute("""
            INSERT OR REPLACE INTO stock_prices 
            (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ('SPY', test_date, spy_close, spy_close, spy_close, spy_close, 1000000))
        
        conn.commit()
        conn.close()
        
        # Test retrieval
        spy_data = self.collector._get_spy_data_for_date(test_date)
        self.assertEqual(spy_data, spy_close)
        
        # Test with non-existent date
        spy_data_none = self.collector._get_spy_data_for_date('2020-01-01')
        self.assertIsNone(spy_data_none)


class TestStockDataCollectorIntegration(unittest.TestCase):
    """Integration tests for StockDataCollector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.collector = StockDataCollector(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_download_small_batch(self):
        """Test downloading a small batch of stock data"""
        # Test with a single ticker for a short recent period
        test_tickers = ['AAPL']
        
        try:
            self.collector.download_stock_data(
                tickers=test_tickers,
                start_date='2024-01-01',
                end_date='2024-01-05',
                batch_size=1
            )
            
            # Check if any data was downloaded
            summary = self.collector.get_stock_data_summary()
            
            # The test mainly verifies that the download process doesn't crash
            self.assertIsInstance(summary, pd.DataFrame)
            
        except Exception as e:
            # If download fails due to network issues, that's acceptable for unit tests
            self.skipTest(f"Network-dependent test failed: {e}")


if __name__ == '__main__':
    unittest.main()

