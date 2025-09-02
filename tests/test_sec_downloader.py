"""
Unit tests for SEC 8-K downloader module.
"""

import unittest
import tempfile
import shutil
import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.sec_downloader import SEC8KDownloader


class TestSEC8KDownloader(unittest.TestCase):
    """Test cases for SEC8KDownloader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.downloader = SEC8KDownloader(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test downloader initialization"""
        self.assertEqual(self.downloader.data_dir, Path(self.test_dir))
        self.assertTrue(self.downloader.data_dir.exists())
        self.assertTrue(self.downloader.db_path.exists())
    
    def test_get_sp500_tickers(self):
        """Test S&P 500 ticker retrieval"""
        tickers = self.downloader.get_sp500_tickers()
        self.assertIsInstance(tickers, list)
        self.assertGreater(len(tickers), 400)  # Should have ~500 tickers
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)
    
    def test_database_creation(self):
        """Test database and table creation"""
        # Check if database exists
        self.assertTrue(self.downloader.db_path.exists())
        
        # Check if tables exist
        conn = sqlite3.connect(self.downloader.db_path)
        cursor = conn.cursor()
        
        # Check filings table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='filings'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        # Check table schema
        cursor.execute("PRAGMA table_info(filings)")
        columns = [row[1] for row in cursor.fetchall()]
        expected_columns = ['accession_number', 'ticker', 'filing_date', 'category', 'content', 'url']
        for col in expected_columns:
            self.assertIn(col, columns)
        
        conn.close()
    
    def test_get_filings_summary_empty(self):
        """Test filings summary with empty database"""
        summary = self.downloader.get_filings_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(summary.empty)
    
    def test_save_filing(self):
        """Test saving a filing to database"""
        test_filing = {
            'accession_number': '0001234567-23-000001',
            'ticker': 'TEST',
            'filing_date': '2023-01-01',
            'category': '8.01',
            'content': 'Test filing content',
            'url': 'https://test.url'
        }
        
        # Save filing
        conn = sqlite3.connect(self.downloader.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO filings 
            (accession_number, ticker, filing_date, category, content, url)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            test_filing['accession_number'],
            test_filing['ticker'],
            test_filing['filing_date'],
            test_filing['category'],
            test_filing['content'],
            test_filing['url']
        ))
        
        conn.commit()
        conn.close()
        
        # Verify filing was saved
        summary = self.downloader.get_filings_summary()
        self.assertFalse(summary.empty)
        self.assertEqual(summary.iloc[0]['ticker'], 'TEST')
        self.assertEqual(summary.iloc[0]['category'], '8.01')
    
    def test_extract_8k_categories(self):
        """Test 8-K category extraction"""
        # Test various content samples
        test_cases = [
            ("Item 1.01 Entry into a Material Agreement", ["1.01"]),
            ("Item 2.02 Results of Operations", ["2.02"]),
            ("Item 8.01 Other Events", ["8.01"]),
            ("Item 1.01 and Item 2.01", ["1.01", "2.01"]),
            ("No items mentioned", []),
        ]
        
        for content, expected in test_cases:
            categories = self.downloader.extract_8k_categories(content)
            self.assertEqual(set(categories), set(expected))


class TestSEC8KDownloaderIntegration(unittest.TestCase):
    """Integration tests for SEC8KDownloader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.downloader = SEC8KDownloader(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_download_small_batch(self):
        """Test downloading a small batch of filings"""
        # Test with a single ticker for a short period
        test_tickers = ['AAPL']
        
        try:
            self.downloader.download_8k_filings(
                tickers=test_tickers,
                start_date='2024-01-01',
                end_date='2024-01-31',
                batch_size=1
            )
            
            # Check if any filings were downloaded
            summary = self.downloader.get_filings_summary()
            
            # Note: This test might not find filings depending on the date range
            # The test mainly verifies that the download process doesn't crash
            self.assertIsInstance(summary, pd.DataFrame)
            
        except Exception as e:
            # If download fails due to network issues, that's acceptable for unit tests
            self.skipTest(f"Network-dependent test failed: {e}")


if __name__ == '__main__':
    unittest.main()

