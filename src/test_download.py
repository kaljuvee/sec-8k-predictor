"""
Test download script for a small sample of companies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sec_downloader import SEC8KDownloader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test download with a small sample"""
    
    # Initialize downloader
    downloader = SEC8KDownloader(data_dir="../data")
    
    # Test with a small sample of well-known companies
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "PG"]
    
    logger.info(f"Testing download with {len(test_tickers)} companies")
    
    # Download filings for 2023-2024 period (smaller timeframe for testing)
    downloader.download_8k_filings(
        tickers=test_tickers,
        start_date="2023-01-01", 
        end_date="2024-12-31",
        batch_size=5
    )
    
    # Generate summary
    summary = downloader.get_filings_summary()
    print("\n=== TEST DOWNLOAD SUMMARY ===")
    print(summary)
    
    # Save test summary
    summary.to_csv("../data/test_filings_summary.csv", index=False)
    
    # Show sample filings by category
    for category in ["2.02", "8.01", "2.01"]:
        filings = downloader.get_filings_by_category(category)
        if not filings.empty:
            print(f"\nSample {category} filings:")
            print(filings.head())

if __name__ == "__main__":
    main()

