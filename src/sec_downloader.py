"""
SEC 8-K Filings Downloader

This module handles downloading SEC 8-K filings from the EDGAR database
and categorizing them by filing type.
"""

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests
from sec_edgar_downloader import Downloader
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEC8KDownloader:
    """Downloads and processes SEC 8-K filings"""
    
    # 8-K Categories mapping
    CATEGORIES = {
        "1.01": "Material definitive agreement",
        "1.02": "Termination of material definitive agreement", 
        "1.05": "Material cybersecurity incidents",
        "2.01": "Completion of acquisition or disposition",
        "2.02": "Results of operations and financial condition",
        "3.01": "Delisting notice",
        "5.01": "Changes in control",
        "5.02": "Departure/appointment of directors or officers",
        "5.03": "Amendments to charter or bylaws",
        "8.01": "Other events"
    }
    
    def __init__(self, data_dir: str = "data", company_name: str = "SEC8KPredictor"):
        """
        Initialize the downloader
        
        Args:
            data_dir: Directory to store downloaded filings
            company_name: Company name for SEC requests (required by EDGAR)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.filings_dir = self.data_dir / "filings"
        self.filings_dir.mkdir(exist_ok=True)
        
        # Initialize downloader
        self.downloader = Downloader(company_name, company_name + "@example.com", str(self.filings_dir))
        
        # Initialize database
        self.db_path = self.data_dir / "sec_filings.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing filing metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                filing_date DATE NOT NULL,
                accession_number TEXT UNIQUE NOT NULL,
                category TEXT,
                file_path TEXT,
                content TEXT,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            
            # Clean tickers (remove dots, etc.)
            cleaned_tickers = []
            for ticker in tickers:
                # Replace dots with dashes for SEC format
                cleaned_ticker = ticker.replace('.', '-')
                cleaned_tickers.append(cleaned_ticker)
                
            logger.info(f"Retrieved {len(cleaned_tickers)} S&P 500 tickers")
            return cleaned_tickers
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 tickers: {e}")
            # Fallback to a small sample for testing
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    def download_8k_filings(self, tickers: List[str], start_date: str = "2014-01-01", 
                           end_date: str = "2024-12-31", batch_size: int = 10) -> None:
        """
        Download 8-K filings for given tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            batch_size: Number of tickers to process in each batch
        """
        logger.info(f"Starting download of 8-K filings for {len(tickers)} tickers")
        
        # Process in batches to avoid overwhelming the server
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            for ticker in batch:
                try:
                    self._download_ticker_filings(ticker, start_date, end_date)
                    # Rate limiting - SEC requires 10 requests per second max
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error downloading filings for {ticker}: {e}")
                    continue
                    
            # Longer pause between batches
            if i + batch_size < len(tickers):
                logger.info("Pausing between batches...")
                time.sleep(1)
    
    def _download_ticker_filings(self, ticker: str, start_date: str, end_date: str) -> None:
        """Download 8-K filings for a specific ticker"""
        try:
            # Download 8-K filings
            num_downloaded = self.downloader.get("8-K", ticker, after=start_date, before=end_date)
            
            if num_downloaded > 0:
                logger.info(f"Downloaded {num_downloaded} 8-K filings for {ticker}")
                self._process_downloaded_filings(ticker)
            else:
                logger.info(f"No 8-K filings found for {ticker}")
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
    
    def _process_downloaded_filings(self, ticker: str) -> None:
        """Process downloaded filings and extract metadata"""
        ticker_dir = self.filings_dir / "sec-edgar-filings" / ticker / "8-K"
        
        if not ticker_dir.exists():
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for filing_dir in ticker_dir.iterdir():
            if filing_dir.is_dir():
                # Extract accession number from directory name
                accession_number = filing_dir.name
                
                # Check if already processed
                cursor.execute("SELECT id FROM filings WHERE accession_number = ?", (accession_number,))
                if cursor.fetchone():
                    continue
                
                # Find the main filing document
                filing_file = None
                for file in filing_dir.glob("*.txt"):
                    if "8-k" in file.name.lower():
                        filing_file = file
                        break
                
                if not filing_file:
                    # Try to find any .txt file
                    txt_files = list(filing_dir.glob("*.txt"))
                    if txt_files:
                        filing_file = txt_files[0]
                
                if filing_file:
                    try:
                        # Read filing content
                        with open(filing_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extract filing date and category
                        filing_date, category = self._extract_filing_metadata(content)
                        
                        # Store in database
                        cursor.execute("""
                            INSERT OR IGNORE INTO filings 
                            (ticker, filing_date, accession_number, category, file_path, content)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (ticker, filing_date, accession_number, category, str(filing_file), content))
                        
                    except Exception as e:
                        logger.error(f"Error processing filing {filing_file}: {e}")
        
        conn.commit()
        conn.close()
    
    def _extract_filing_metadata(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract filing date and category from filing content
        
        Args:
            content: Raw filing content
            
        Returns:
            Tuple of (filing_date, category)
        """
        filing_date = None
        category = None
        
        try:
            # Extract filing date
            date_pattern = r'FILED AS OF DATE:\s*(\d{8})'
            date_match = re.search(date_pattern, content)
            if date_match:
                date_str = date_match.group(1)
                filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                # Try alternative date pattern
                date_pattern = r'CONFORMED PERIOD OF REPORT:\s*(\d{8})'
                date_match = re.search(date_pattern, content)
                if date_match:
                    date_str = date_match.group(1)
                    filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Extract category from items mentioned
            for cat_code, description in self.CATEGORIES.items():
                if f"Item {cat_code}" in content or f"ITEM {cat_code}" in content:
                    category = cat_code
                    break
            
            # If no specific category found, check for common patterns
            if not category:
                if "Item 8.01" in content or "ITEM 8.01" in content:
                    category = "8.01"
                elif "earnings" in content.lower() or "results" in content.lower():
                    category = "2.02"
                elif "acquisition" in content.lower() or "merger" in content.lower():
                    category = "2.01"
                    
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return filing_date, category
    
    def get_filings_summary(self) -> pd.DataFrame:
        """Get summary of downloaded filings"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                ticker,
                category,
                COUNT(*) as count,
                MIN(filing_date) as earliest_date,
                MAX(filing_date) as latest_date
            FROM filings 
            WHERE filing_date IS NOT NULL
            GROUP BY ticker, category
            ORDER BY ticker, category
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_filings_by_category(self, category: str = None) -> pd.DataFrame:
        """Get filings filtered by category"""
        conn = sqlite3.connect(self.db_path)
        
        if category:
            query = """
                SELECT ticker, filing_date, accession_number, category, file_path
                FROM filings 
                WHERE category = ? AND filing_date IS NOT NULL
                ORDER BY filing_date DESC
            """
            df = pd.read_sql_query(query, conn, params=(category,))
        else:
            query = """
                SELECT ticker, filing_date, accession_number, category, file_path
                FROM filings 
                WHERE filing_date IS NOT NULL
                ORDER BY filing_date DESC
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df


if __name__ == "__main__":
    # Test the downloader
    downloader = SEC8KDownloader()
    
    # Get a small sample of tickers for testing
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("Testing SEC 8-K downloader...")
    downloader.download_8k_filings(test_tickers, "2023-01-01", "2023-12-31", batch_size=1)
    
    # Show summary
    summary = downloader.get_filings_summary()
    print("\nFilings Summary:")
    print(summary)

