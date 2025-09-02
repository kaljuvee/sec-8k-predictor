"""
Unit tests for machine learning models module.
"""

import unittest
import tempfile
import shutil
import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import SEC8KPredictor


class TestSEC8KPredictor(unittest.TestCase):
    """Test cases for SEC8KPredictor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.predictor = SEC8KPredictor(data_dir=self.test_dir)
        
        # Create sample features database
        self._create_sample_features_db()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def _create_sample_features_db(self):
        """Create sample features database with test data"""
        features_db = Path(self.test_dir) / "features.db"
        conn = sqlite3.connect(features_db)
        
        # Create features table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                accession_number TEXT PRIMARY KEY,
                ticker TEXT,
                filing_date TEXT,
                category TEXT,
                sentiment_score REAL,
                urgency_score REAL,
                financial_impact_score REAL,
                market_relevance_score REAL,
                return_5d REAL,
                return_9d REAL,
                relative_return_5d REAL,
                relative_return_9d REAL,
                volatility_change_5d REAL,
                tfidf_features TEXT
            )
        """)
        
        # Create sample TF-IDF features (simplified)
        sample_tfidf = [0.1, 0.2, 0.0, 0.3, 0.1] + [0.0] * 995  # 1000 features total
        tfidf_json = json.dumps(sample_tfidf)
        
        # Insert sample data with sufficient samples for training
        sample_features = []
        for i in range(20):  # Create 20 samples
            category = "8.01" if i < 10 else "2.02"
            sentiment = 5 + np.random.normal(0, 2)
            urgency = 5 + np.random.normal(0, 2)
            financial = 5 + np.random.normal(0, 2)
            market = 5 + np.random.normal(0, 2)
            
            # Create some correlation between features and returns
            return_5d = (sentiment + financial - 10) * 0.01 + np.random.normal(0, 0.02)
            return_9d = return_5d * 1.2 + np.random.normal(0, 0.01)
            relative_return_5d = return_5d + np.random.normal(0, 0.005)
            relative_return_9d = return_9d + np.random.normal(0, 0.005)
            
            sample_features.append((
                f'000123456{i:02d}-23-00000{i}',  # accession_number
                'TEST',  # ticker
                f'2023-01-{i+1:02d}',  # filing_date
                category,  # category
                max(0, min(10, sentiment)),  # sentiment_score
                max(0, min(10, urgency)),  # urgency_score
                max(0, min(10, financial)),  # financial_impact_score
                max(0, min(10, market)),  # market_relevance_score
                return_5d,  # return_5d
                return_9d,  # return_9d
                relative_return_5d,  # relative_return_5d
                relative_return_9d,  # relative_return_9d
                abs(np.random.normal(0, 0.01)),  # volatility_change_5d
                tfidf_json  # tfidf_features
            ))
        
        for features in sample_features:
            conn.execute("""
                INSERT INTO features 
                (accession_number, ticker, filing_date, category,
                 sentiment_score, urgency_score, financial_impact_score, market_relevance_score,
                 return_5d, return_9d, relative_return_5d, relative_return_9d,
                 volatility_change_5d, tfidf_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, features)
        
        conn.commit()
        conn.close()
    
    def test_init(self):
        """Test predictor initialization"""
        self.assertEqual(self.predictor.data_dir, Path(self.test_dir))
        self.assertTrue(self.predictor.models_dir.exists())
        self.assertTrue(self.predictor.features_db.exists())
    
    def test_load_training_data(self):
        """Test loading training data"""
        X, y, feature_names = self.predictor.load_training_data("8.01", "relative_return_5d")
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(feature_names, list)
        
        if len(X) > 0:
            self.assertEqual(len(X), len(y))
            self.assertEqual(X.shape[1], len(feature_names))
            self.assertGreater(len(feature_names), 4)  # At least LLM features
    
    def test_train_classifier(self):
        """Test classifier training"""
        results = self.predictor.train_classifier("8.01", "relative_return_5d")
        
        if results:  # If training was successful
            self.assertIsInstance(results, dict)
            self.assertIn('category', results)
            self.assertIn('model_type', results)
            self.assertIn('accuracy', results)
            self.assertIn('n_samples', results)
            
            self.assertEqual(results['category'], "8.01")
            self.assertEqual(results['model_type'], 'classifier')
            self.assertGreaterEqual(results['accuracy'], 0)
            self.assertLessEqual(results['accuracy'], 1)
    
    def test_train_regressor(self):
        """Test regressor training"""
        results = self.predictor.train_regressor("8.01", "relative_return_5d")
        
        if results:  # If training was successful
            self.assertIsInstance(results, dict)
            self.assertIn('category', results)
            self.assertIn('model_type', results)
            self.assertIn('r2_score', results)
            self.assertIn('correlation', results)
            
            self.assertEqual(results['category'], "8.01")
            self.assertEqual(results['model_type'], 'regressor')
    
    def test_train_all_models(self):
        """Test training all models"""
        results = self.predictor.train_all_models(['relative_return_5d'])
        
        self.assertIsInstance(results, dict)
        self.assertIn('classifiers', results)
        self.assertIn('regressors', results)
        self.assertIn('summary', results)
        
        # Check summary
        summary = results['summary']
        self.assertIn('total_classifiers', summary)
        self.assertIn('total_regressors', summary)
    
    def test_get_model_summary_empty(self):
        """Test model summary with no trained models"""
        summary = self.predictor.get_model_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        # Should be empty if no models are trained
    
    def test_load_model_nonexistent(self):
        """Test loading non-existent model"""
        model, scaler = self.predictor.load_model("nonexistent", "classifier", "relative_return_5d")
        self.assertIsNone(model)
        self.assertIsNone(scaler)
    
    def test_predict_without_model(self):
        """Test prediction without trained model"""
        # Create dummy features
        features = np.array([[1, 2, 3, 4] + [0.1] * 1000])
        
        predictions = self.predictor.predict(features, "nonexistent", "classifier", "relative_return_5d")
        self.assertEqual(len(predictions), 0)


class TestModelTrainingIntegration(unittest.TestCase):
    """Integration tests for model training"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.predictor = SEC8KPredictor(data_dir=self.test_dir)
        
        # Create more comprehensive sample data
        self._create_comprehensive_features_db()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def _create_comprehensive_features_db(self):
        """Create comprehensive features database for integration testing"""
        features_db = Path(self.test_dir) / "features.db"
        conn = sqlite3.connect(features_db)
        
        # Create features table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                accession_number TEXT PRIMARY KEY,
                ticker TEXT,
                filing_date TEXT,
                category TEXT,
                sentiment_score REAL,
                urgency_score REAL,
                financial_impact_score REAL,
                market_relevance_score REAL,
                return_5d REAL,
                return_9d REAL,
                relative_return_5d REAL,
                relative_return_9d REAL,
                volatility_change_5d REAL,
                tfidf_features TEXT
            )
        """)
        
        # Create sample TF-IDF features
        sample_tfidf = [np.random.random() for _ in range(1000)]
        tfidf_json = json.dumps(sample_tfidf)
        
        # Insert more comprehensive sample data
        categories = ["8.01", "2.02", "1.01"]
        sample_features = []
        
        for cat_idx, category in enumerate(categories):
            for i in range(15):  # 15 samples per category
                # Create features with some patterns
                sentiment = 3 + cat_idx * 2 + np.random.normal(0, 1)
                urgency = 4 + cat_idx + np.random.normal(0, 1)
                financial = 5 + cat_idx * 1.5 + np.random.normal(0, 1)
                market = 6 - cat_idx + np.random.normal(0, 1)
                
                # Create returns with some correlation to features
                base_return = (sentiment + financial - 8) * 0.005
                return_5d = base_return + np.random.normal(0, 0.01)
                return_9d = return_5d * 1.1 + np.random.normal(0, 0.005)
                relative_return_5d = return_5d + np.random.normal(0, 0.003)
                relative_return_9d = return_9d + np.random.normal(0, 0.003)
                
                sample_features.append((
                    f'00012345{cat_idx}{i:02d}-23-00000{i}',  # accession_number
                    f'TEST{cat_idx}',  # ticker
                    f'2023-{cat_idx+1:02d}-{i+1:02d}',  # filing_date
                    category,  # category
                    max(0, min(10, sentiment)),  # sentiment_score
                    max(0, min(10, urgency)),  # urgency_score
                    max(0, min(10, financial)),  # financial_impact_score
                    max(0, min(10, market)),  # market_relevance_score
                    return_5d,  # return_5d
                    return_9d,  # return_9d
                    relative_return_5d,  # relative_return_5d
                    relative_return_9d,  # relative_return_9d
                    abs(np.random.normal(0, 0.01)),  # volatility_change_5d
                    tfidf_json  # tfidf_features
                ))
        
        for features in sample_features:
            conn.execute("""
                INSERT INTO features 
                (accession_number, ticker, filing_date, category,
                 sentiment_score, urgency_score, financial_impact_score, market_relevance_score,
                 return_5d, return_9d, relative_return_5d, relative_return_9d,
                 volatility_change_5d, tfidf_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, features)
        
        conn.commit()
        conn.close()
    
    def test_end_to_end_training(self):
        """Test end-to-end model training process"""
        # Train models for all categories
        results = self.predictor.train_all_models(['relative_return_5d'])
        
        # Verify results structure
        self.assertIn('classifiers', results)
        self.assertIn('regressors', results)
        self.assertIn('summary', results)
        
        # Check if any models were trained
        total_models = len(results['classifiers']) + len(results['regressors'])
        self.assertGreater(total_models, 0)
        
        # Verify model files were created
        model_files = list(self.predictor.models_dir.glob('*.joblib'))
        self.assertGreater(len(model_files), 0)
        
        # Test model summary
        summary = self.predictor.get_model_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        
        if not summary.empty:
            # Verify summary structure
            expected_columns = ['category', 'model_type', 'target_variable', 'n_samples']
            for col in expected_columns:
                self.assertIn(col, summary.columns)


if __name__ == '__main__':
    unittest.main()

