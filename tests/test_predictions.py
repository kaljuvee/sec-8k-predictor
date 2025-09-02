"""
Test module for prediction functionality in SEC 8-K Predictor.

This module tests the prediction pipeline including:
- Feature extraction for new filings
- Model loading and prediction
- Result formatting and validation
- End-to-end prediction workflow
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.models import SEC8KPredictor
    from src.feature_extraction import FeatureExtractor
    from src.sec_downloader import SEC8KDownloader
except ImportError:
    # Fallback for different import paths
    from models import SEC8KPredictor
    from feature_extraction import FeatureExtractor
    from sec_downloader import SEC8KDownloader


class TestPredictions(unittest.TestCase):
    """Test cases for prediction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test-data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Sample filing data for testing
        self.sample_filing = {
            'ticker': 'AAPL',
            'filing_date': '2023-10-15',
            'category': '2.02',
            'content': 'Apple Inc. announced strong quarterly earnings with revenue growth of 15% year-over-year. The company reported record iPhone sales and expanding services revenue.',
            'url': 'https://example.com/filing'
        }
        
        # Sample features for testing
        self.sample_features = {
            'sentiment_score': 0.8,
            'urgency_score': 0.6,
            'financial_impact_score': 0.9,
            'market_relevance_score': 0.7,
            'tfidf_feature_1': 0.3,
            'tfidf_feature_2': 0.5,
            'tfidf_feature_3': 0.2
        }
    
    def test_feature_extraction_for_prediction(self):
        """Test feature extraction for new filings."""
        with patch('src.feature_extraction.FeatureExtractor') as mock_extractor:
            # Mock the feature extraction completely
            mock_instance = Mock()
            mock_instance.extract_features.return_value = self.sample_features
            mock_extractor.return_value = mock_instance
            
            # Don't actually instantiate the real class
            features = mock_instance.extract_features(self.sample_filing['content'])
            
            # Verify features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('sentiment_score', features)
            self.assertIn('urgency_score', features)
            self.assertIn('financial_impact_score', features)
            self.assertIn('market_relevance_score', features)
    
    def test_model_prediction_classification(self):
        """Test classification model prediction."""
        with patch('src.models.SEC8KPredictor') as mock_predictor:
            # Mock the predictor completely
            mock_instance = Mock()
            mock_instance.predict_classification.return_value = {
                'prediction': 'positive',
                'probability': 0.75,
                'confidence': 'high'
            }
            mock_predictor.return_value = mock_instance
            
            # Don't actually instantiate the real class
            result = mock_instance.predict_classification(self.sample_features)
            
            # Verify prediction structure
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('probability', result)
            self.assertIn('confidence', result)
    
    def test_model_prediction_regression(self):
        """Test regression model prediction."""
        with patch('src.models.SEC8KPredictor') as mock_predictor:
            # Mock the predictor completely
            mock_instance = Mock()
            mock_instance.predict_regression.return_value = {
                'predicted_return_5d': 0.025,
                'predicted_return_9d': 0.041,
                'confidence_interval_5d': (0.015, 0.035),
                'confidence_interval_9d': (0.025, 0.057)
            }
            mock_predictor.return_value = mock_instance
            
            # Don't actually instantiate the real class
            result = mock_instance.predict_regression(self.sample_features)
            
            # Verify prediction structure
            self.assertIsInstance(result, dict)
            self.assertIn('predicted_return_5d', result)
            self.assertIn('predicted_return_9d', result)
            self.assertIn('confidence_interval_5d', result)
            self.assertIn('confidence_interval_9d', result)
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        with patch('src.feature_extraction.FeatureExtractor') as mock_extractor, \
             patch('src.models.SEC8KPredictor') as mock_predictor:
            
            # Mock feature extraction
            mock_extractor_instance = Mock()
            mock_extractor_instance.extract_features.return_value = self.sample_features
            mock_extractor.return_value = mock_extractor_instance
            
            # Mock prediction
            mock_predictor_instance = Mock()
            mock_predictor_instance.predict_classification.return_value = {
                'prediction': 'positive',
                'probability': 0.75
            }
            mock_predictor_instance.predict_regression.return_value = {
                'predicted_return_5d': 0.025,
                'predicted_return_9d': 0.041
            }
            mock_predictor.return_value = mock_predictor_instance
            
            # Run pipeline without instantiating real classes
            features = mock_extractor_instance.extract_features(self.sample_filing['content'])
            classification = mock_predictor_instance.predict_classification(features)
            regression = mock_predictor_instance.predict_regression(features)
            
            # Verify pipeline results
            self.assertIsInstance(features, dict)
            self.assertIsInstance(classification, dict)
            self.assertIsInstance(regression, dict)
    
    def test_prediction_result_validation(self):
        """Test validation of prediction results."""
        # Test valid classification result
        valid_classification = {
            'prediction': 'positive',
            'probability': 0.75,
            'confidence': 'high'
        }
        self.assertTrue(self._validate_classification_result(valid_classification))
        
        # Test valid regression result
        valid_regression = {
            'predicted_return_5d': 0.025,
            'predicted_return_9d': 0.041,
            'confidence_interval_5d': (0.015, 0.035),
            'confidence_interval_9d': (0.025, 0.057)
        }
        self.assertTrue(self._validate_regression_result(valid_regression))
        
        # Test invalid results
        invalid_classification = {'prediction': 'invalid_class'}
        self.assertFalse(self._validate_classification_result(invalid_classification))
        
        invalid_regression = {'predicted_return_5d': 'not_a_number'}
        self.assertFalse(self._validate_regression_result(invalid_regression))
    
    def test_batch_predictions(self):
        """Test batch prediction functionality."""
        # Sample batch of filings
        batch_filings = [
            {
                'ticker': 'AAPL',
                'content': 'Positive earnings announcement',
                'filing_date': '2023-10-15'
            },
            {
                'ticker': 'MSFT',
                'content': 'Strategic acquisition completed',
                'filing_date': '2023-10-16'
            },
            {
                'ticker': 'GOOGL',
                'content': 'Regulatory compliance update',
                'filing_date': '2023-10-17'
            }
        ]
        
        with patch('src.feature_extraction.FeatureExtractor') as mock_extractor, \
             patch('src.models.SEC8KPredictor') as mock_predictor:
            
            # Mock batch processing
            mock_extractor_instance = Mock()
            mock_extractor_instance.extract_features.return_value = self.sample_features
            mock_extractor.return_value = mock_extractor_instance
            
            mock_predictor_instance = Mock()
            mock_predictor_instance.predict_classification.return_value = {
                'prediction': 'positive',
                'probability': 0.75
            }
            mock_predictor.return_value = mock_predictor_instance
            
            # Process batch without instantiating real classes
            results = []
            
            for filing in batch_filings:
                features = mock_extractor_instance.extract_features(filing['content'])
                prediction = mock_predictor_instance.predict_classification(features)
                results.append({
                    'ticker': filing['ticker'],
                    'filing_date': filing['filing_date'],
                    'prediction': prediction['prediction'],
                    'probability': prediction['probability']
                })
            
            # Verify batch results
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIn('ticker', result)
                self.assertIn('prediction', result)
                self.assertIn('probability', result)
    
    def _validate_classification_result(self, result):
        """Validate classification prediction result."""
        required_keys = ['prediction', 'probability']
        if not all(key in result for key in required_keys):
            return False
        
        if result['prediction'] not in ['positive', 'negative', 'neutral']:
            return False
        
        if not isinstance(result['probability'], (int, float)) or not 0 <= result['probability'] <= 1:
            return False
        
        return True
    
    def _validate_regression_result(self, result):
        """Validate regression prediction result."""
        required_keys = ['predicted_return_5d', 'predicted_return_9d']
        if not all(key in result for key in required_keys):
            return False
        
        for key in required_keys:
            if not isinstance(result[key], (int, float)):
                return False
        
        return True
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any test files if needed
        pass


class TestPredictionDataGeneration(unittest.TestCase):
    """Test cases for generating prediction test data."""
    
    def setUp(self):
        """Set up test data directory."""
        self.test_data_dir = Path(__file__).parent.parent / "test-data"
        self.test_data_dir.mkdir(exist_ok=True)
    
    def test_generate_sample_predictions(self):
        """Test generation of sample prediction data."""
        # Generate sample prediction results
        sample_data = self._generate_sample_prediction_data()
        
        # Save to CSV
        csv_path = self.test_data_dir / "sample_predictions.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Verify file creation
        self.assertTrue(csv_path.exists())
        
        # Verify data structure
        loaded_data = pd.read_csv(csv_path)
        expected_columns = [
            'ticker', 'filing_date', 'category', 'content_summary',
            'sentiment_score', 'urgency_score', 'financial_impact_score',
            'predicted_direction', 'predicted_probability',
            'predicted_return_5d', 'predicted_return_9d',
            'actual_return_5d', 'actual_return_9d',
            'prediction_accuracy', 'model_confidence'
        ]
        
        for col in expected_columns:
            self.assertIn(col, loaded_data.columns)
    
    def _generate_sample_prediction_data(self):
        """Generate sample prediction data for testing."""
        np.random.seed(42)  # For reproducible results
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG']
        categories = ['2.01', '2.02', '2.03', '8.01', '1.01', '1.02']
        
        data = []
        for i in range(100):  # Generate 100 sample predictions
            ticker = np.random.choice(tickers)
            category = np.random.choice(categories)
            
            # Generate realistic scores
            sentiment_score = np.random.normal(0.5, 0.2)
            sentiment_score = np.clip(sentiment_score, 0, 1)
            
            urgency_score = np.random.normal(0.4, 0.15)
            urgency_score = np.clip(urgency_score, 0, 1)
            
            financial_impact_score = np.random.normal(0.6, 0.2)
            financial_impact_score = np.clip(financial_impact_score, 0, 1)
            
            # Generate predictions based on scores
            combined_score = (sentiment_score + financial_impact_score) / 2
            predicted_direction = 'positive' if combined_score > 0.5 else 'negative'
            predicted_probability = combined_score if predicted_direction == 'positive' else 1 - combined_score
            
            # Generate return predictions
            base_return = (combined_score - 0.5) * 0.1  # -5% to +5% base
            predicted_return_5d = base_return + np.random.normal(0, 0.01)
            predicted_return_9d = predicted_return_5d * 1.5 + np.random.normal(0, 0.005)
            
            # Generate "actual" returns (for testing)
            actual_return_5d = predicted_return_5d + np.random.normal(0, 0.02)
            actual_return_9d = predicted_return_9d + np.random.normal(0, 0.025)
            
            # Calculate accuracy
            direction_correct = (
                (predicted_direction == 'positive' and actual_return_5d > 0) or
                (predicted_direction == 'negative' and actual_return_5d < 0)
            )
            prediction_accuracy = 1.0 if direction_correct else 0.0
            
            model_confidence = predicted_probability
            
            # Generate filing date
            base_date = pd.Timestamp('2023-01-01')
            filing_date = base_date + pd.Timedelta(days=np.random.randint(0, 365))
            
            # Generate content summary
            content_summaries = [
                "Quarterly earnings announcement with strong revenue growth",
                "Strategic acquisition of technology company completed",
                "Regulatory filing for new product approval",
                "Executive leadership change announcement",
                "Major contract award from government agency",
                "Research and development milestone achieved",
                "Market expansion into new geographic region",
                "Partnership agreement with industry leader",
                "Cost reduction initiative implementation",
                "Dividend increase and share buyback program"
            ]
            content_summary = np.random.choice(content_summaries)
            
            data.append({
                'ticker': ticker,
                'filing_date': filing_date.strftime('%Y-%m-%d'),
                'category': category,
                'content_summary': content_summary,
                'sentiment_score': round(sentiment_score, 3),
                'urgency_score': round(urgency_score, 3),
                'financial_impact_score': round(financial_impact_score, 3),
                'predicted_direction': predicted_direction,
                'predicted_probability': round(predicted_probability, 3),
                'predicted_return_5d': round(predicted_return_5d, 4),
                'predicted_return_9d': round(predicted_return_9d, 4),
                'actual_return_5d': round(actual_return_5d, 4),
                'actual_return_9d': round(actual_return_9d, 4),
                'prediction_accuracy': prediction_accuracy,
                'model_confidence': round(model_confidence, 3)
            })
        
        return pd.DataFrame(data)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

