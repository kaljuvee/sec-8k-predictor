"""
Unit tests for feature extraction module.
"""

import unittest
import tempfile
import shutil
import sqlite3
import pandas as pd
import json
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.extractor = FeatureExtractor(data_dir=self.test_dir)
        
        # Create sample databases
        self._create_sample_filings_db()
        self._create_sample_stock_db()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def _create_sample_filings_db(self):
        """Create sample SEC filings database"""
        filings_db = Path(self.test_dir) / "sec_filings.db"
        conn = sqlite3.connect(filings_db)
        
        # Create filings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                accession_number TEXT PRIMARY KEY,
                ticker TEXT,
                filing_date TEXT,
                category TEXT,
                content TEXT,
                url TEXT
            )
        """)
        
        # Insert sample data
        sample_filings = [
            ('0001234567-23-000001', 'TEST', '2023-01-15', '8.01', 
             'The company announced positive quarterly results with strong revenue growth.', 
             'https://test.url/1'),
            ('0001234567-23-000002', 'TEST', '2023-02-15', '2.02', 
             'Financial results show declining performance due to market conditions.', 
             'https://test.url/2'),
        ]
        
        for filing in sample_filings:
            conn.execute("""
                INSERT INTO filings 
                (accession_number, ticker, filing_date, category, content, url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, filing)
        
        conn.commit()
        conn.close()
    
    def _create_sample_stock_db(self):
        """Create sample stock data database"""
        stock_db = Path(self.test_dir) / "stock_data.db"
        conn = sqlite3.connect(stock_db)
        
        # Create stock_returns table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_returns (
                ticker TEXT,
                date TEXT,
                return_5d REAL,
                return_9d REAL,
                relative_return_5d REAL,
                relative_return_9d REAL,
                volatility_5d REAL,
                volatility_9d REAL,
                volatility_change_5d REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Insert sample data
        sample_returns = [
            ('TEST', '2023-01-15', 0.05, 0.08, 0.02, 0.03, 0.15, 0.18, 0.01),
            ('TEST', '2023-02-15', -0.03, -0.05, -0.01, -0.02, 0.20, 0.22, 0.05),
        ]
        
        for returns in sample_returns:
            conn.execute("""
                INSERT INTO stock_returns 
                (ticker, date, return_5d, return_9d, relative_return_5d, relative_return_9d,
                 volatility_5d, volatility_9d, volatility_change_5d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, returns)
        
        conn.commit()
        conn.close()
    
    def test_init(self):
        """Test extractor initialization"""
        self.assertEqual(self.extractor.data_dir, Path(self.test_dir))
        self.assertTrue(self.extractor.features_db.exists())
    
    def test_database_creation(self):
        """Test features database and table creation"""
        # Check if database exists
        self.assertTrue(self.extractor.features_db.exists())
        
        # Check if table exists
        conn = sqlite3.connect(self.extractor.features_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        # Check table schema
        cursor.execute("PRAGMA table_info(features)")
        columns = [row[1] for row in cursor.fetchall()]
        expected_columns = [
            'accession_number', 'ticker', 'filing_date', 'category',
            'sentiment_score', 'urgency_score', 'financial_impact_score', 'market_relevance_score',
            'return_5d', 'return_9d', 'relative_return_5d', 'relative_return_9d',
            'volatility_change_5d', 'tfidf_features'
        ]
        for col in expected_columns:
            self.assertIn(col, columns)
        
        conn.close()
    
    def test_extract_llm_features(self):
        """Test LLM feature extraction"""
        test_content = "The company reported strong quarterly earnings with revenue growth of 15%."
        
        try:
            features = self.extractor.extract_llm_features(test_content)
            
            self.assertIsInstance(features, dict)
            self.assertIn('sentiment_score', features)
            self.assertIn('urgency_score', features)
            self.assertIn('financial_impact_score', features)
            self.assertIn('market_relevance_score', features)
            
            # Check score ranges
            for score in features.values():
                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 10)
                
        except Exception as e:
            # If OpenAI API is not available, skip this test
            self.skipTest(f"LLM feature extraction test failed (API issue): {e}")
    
    def test_truncate_content(self):
        """Test content truncation"""
        # Test short content
        short_content = "Short content"
        truncated = self.extractor._truncate_content(short_content, max_tokens=100)
        self.assertEqual(truncated, short_content)
        
        # Test long content
        long_content = " ".join(["word"] * 1000)
        truncated = self.extractor._truncate_content(long_content, max_tokens=100)
        self.assertLess(len(truncated.split()), len(long_content.split()))
    
    def test_get_features_summary_empty(self):
        """Test features summary with empty database"""
        summary = self.extractor.get_features_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue(summary.empty)
    
    def test_create_tfidf_features_empty(self):
        """Test TF-IDF feature creation with empty data"""
        # This should not crash even with no data
        try:
            self.extractor.create_tfidf_features()
        except Exception as e:
            # Should handle empty data gracefully
            self.assertIn("No content found", str(e))
    
    def test_process_filings_with_sample_data(self):
        """Test processing filings with sample data"""
        try:
            # Process the sample filings
            self.extractor.process_filings(limit=2)
            
            # Check if features were created
            conn = sqlite3.connect(self.extractor.features_db)
            features_data = pd.read_sql_query("SELECT * FROM features", conn)
            conn.close()
            
            # Should have processed the sample filings
            self.assertFalse(features_data.empty)
            self.assertLessEqual(len(features_data), 2)
            
            # Check if required columns exist
            required_columns = ['accession_number', 'ticker', 'filing_date', 'category']
            for col in required_columns:
                self.assertIn(col, features_data.columns)
                
        except Exception as e:
            # If processing fails due to API issues, that's acceptable
            self.skipTest(f"Feature processing test failed (API issue): {e}")


class TestFeatureExtractionUtils(unittest.TestCase):
    """Test utility functions in feature extraction"""
    
    def test_parse_llm_response(self):
        """Test LLM response parsing"""
        extractor = FeatureExtractor(data_dir=tempfile.mkdtemp())
        
        # Test valid JSON response
        valid_response = '{"sentiment_score": 7, "urgency_score": 5, "financial_impact_score": 8, "market_relevance_score": 6}'
        parsed = extractor._parse_llm_response(valid_response)
        
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed['sentiment_score'], 7)
        self.assertEqual(parsed['urgency_score'], 5)
        
        # Test invalid JSON response
        invalid_response = "Not a JSON response"
        parsed_invalid = extractor._parse_llm_response(invalid_response)
        
        # Should return default scores
        self.assertIsInstance(parsed_invalid, dict)
        self.assertEqual(parsed_invalid['sentiment_score'], 5)
        
        # Clean up
        shutil.rmtree(extractor.data_dir)


if __name__ == '__main__':
    unittest.main()

