"""
Download all S&P 500 8-K filings for the specified time period
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sec_downloader import SEC8KDownloader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Download all S&P 500 8-K filings"""
    
    # Initialize downloader
    downloader = SEC8KDownloader(data_dir="../data")
    
    # Get S&P 500 tickers
    logger.info("Getting S&P 500 tickers...")
    tickers = downloader.get_sp500_tickers()
    logger.info(f"Found {len(tickers)} S&P 500 companies")
    
    # Download filings for 2014-2024 period
    logger.info("Starting download of 8-K filings for 2014-2024...")
    downloader.download_8k_filings(
        tickers=tickers,
        start_date="2014-01-01",
        end_date="2024-12-31",
        batch_size=10
    )
    
    # Generate summary report
    logger.info("Generating summary report...")
    summary = downloader.get_filings_summary()
    
    # Save summary to CSV
    summary_path = "../data/filings_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Print overall statistics
    total_filings = summary['count'].sum()
    unique_tickers = summary['ticker'].nunique()
    categories = summary['category'].value_counts()
    
    print(f"\n=== DOWNLOAD SUMMARY ===")
    print(f"Total filings downloaded: {total_filings}")
    print(f"Unique companies: {unique_tickers}")
    print(f"\nFilings by category:")
    for category, count in categories.items():
        category_name = downloader.CATEGORIES.get(category, "Unknown/Other")
        print(f"  {category}: {count} ({category_name})")
    
    print(f"\nDetailed summary saved to: {summary_path}")
    print("Download completed successfully!")

if __name__ == "__main__":
    main()

