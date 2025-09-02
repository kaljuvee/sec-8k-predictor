"""
Test script for feature extraction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import FeatureExtractor
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test feature extraction with proper initialization"""
    
    # Initialize feature extractor (this will create the features database)
    extractor = FeatureExtractor(data_dir="../data")
    
    # Check if we have any filings to process
    conn = sqlite3.connect("../data/sec_filings.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM filings WHERE filing_date IS NOT NULL AND content IS NOT NULL")
    count = cursor.fetchone()[0]
    conn.close()
    
    logger.info(f"Found {count} filings available for processing")
    
    if count == 0:
        logger.warning("No filings found. Please run the SEC downloader first.")
        return
    
    # Process a small sample
    logger.info("Processing sample filings...")
    extractor.process_filings(limit=3)
    
    # Create TF-IDF features
    logger.info("Creating TF-IDF features...")
    extractor.create_tfidf_features()
    
    # Show summary
    summary = extractor.get_features_summary()
    print("\nFeatures Summary:")
    print(summary)
    
    # Test creating training dataset for a category
    if not summary.empty:
        test_category = summary.iloc[0]['category']
        if test_category:
            logger.info(f"Creating training dataset for category: {test_category}")
            X, y, feature_names = extractor.create_training_dataset(test_category)
            print(f"\nTraining dataset shape: {X.shape}")
            print(f"Target vector shape: {y.shape}")
            print(f"Number of features: {len(feature_names)}")

if __name__ == "__main__":
    main()

