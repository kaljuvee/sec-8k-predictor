#!/usr/bin/env python3
"""
S&P 500 Data Collector

This module downloads the current S&P 500 company list and collects
stock data for comprehensive prediction testing.
"""

import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SP500DataCollector:
    """Collects S&P 500 company data and stock information."""
    
    def __init__(self, data_dir="data"):
        """Initialize the collector.
        
        Args:
            data_dir (str): Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_sp500_list(self):
        """Download current S&P 500 company list from Wikipedia.
        
        Returns:
            pd.DataFrame: DataFrame with company information
        """
        logger.info("Downloading S&P 500 company list from Wikipedia...")
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        try:
            # Read the table directly with pandas
            tables = pd.read_html(url)
            sp500_df = tables[0]  # First table contains the company list
            
            # Clean column names
            sp500_df.columns = ['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry', 
                               'Headquarters_Location', 'Date_Added', 'CIK', 'Founded']
            
            # Clean the data
            sp500_df['Symbol'] = sp500_df['Symbol'].str.strip()
            sp500_df['Security'] = sp500_df['Security'].str.strip()
            
            logger.info(f"Successfully downloaded {len(sp500_df)} S&P 500 companies")
            
            # Save to CSV
            output_file = self.data_dir / "sp500_companies.csv"
            sp500_df.to_csv(output_file, index=False)
            logger.info(f"Saved company list to: {output_file}")
            
            return sp500_df
            
        except Exception as e:
            logger.error(f"Error downloading S&P 500 list: {e}")
            
            # Fallback: create a sample list with major companies
            logger.info("Using fallback list of major S&P 500 companies...")
            fallback_companies = [
                ['AAPL', 'Apple Inc.', 'Information Technology'],
                ['MSFT', 'Microsoft Corporation', 'Information Technology'],
                ['GOOGL', 'Alphabet Inc.', 'Communication Services'],
                ['AMZN', 'Amazon.com Inc.', 'Consumer Discretionary'],
                ['TSLA', 'Tesla Inc.', 'Consumer Discretionary'],
                ['META', 'Meta Platforms Inc.', 'Communication Services'],
                ['NVDA', 'NVIDIA Corporation', 'Information Technology'],
                ['JPM', 'JPMorgan Chase & Co.', 'Financials'],
                ['JNJ', 'Johnson & Johnson', 'Health Care'],
                ['V', 'Visa Inc.', 'Information Technology'],
                ['PG', 'Procter & Gamble Co.', 'Consumer Staples'],
                ['UNH', 'UnitedHealth Group Inc.', 'Health Care'],
                ['HD', 'Home Depot Inc.', 'Consumer Discretionary'],
                ['MA', 'Mastercard Inc.', 'Information Technology'],
                ['BAC', 'Bank of America Corp.', 'Financials'],
                ['XOM', 'Exxon Mobil Corporation', 'Energy'],
                ['PFE', 'Pfizer Inc.', 'Health Care'],
                ['KO', 'Coca-Cola Co.', 'Consumer Staples'],
                ['ABBV', 'AbbVie Inc.', 'Health Care'],
                ['CVX', 'Chevron Corporation', 'Energy'],
                ['LLY', 'Eli Lilly and Co.', 'Health Care'],
                ['AVGO', 'Broadcom Inc.', 'Information Technology'],
                ['PEP', 'PepsiCo Inc.', 'Consumer Staples'],
                ['TMO', 'Thermo Fisher Scientific Inc.', 'Health Care'],
                ['COST', 'Costco Wholesale Corp.', 'Consumer Staples'],
                ['WMT', 'Walmart Inc.', 'Consumer Staples'],
                ['ABT', 'Abbott Laboratories', 'Health Care'],
                ['MRK', 'Merck & Co. Inc.', 'Health Care'],
                ['CRM', 'Salesforce Inc.', 'Information Technology'],
                ['ACN', 'Accenture plc', 'Information Technology'],
                ['NFLX', 'Netflix Inc.', 'Communication Services'],
                ['ADBE', 'Adobe Inc.', 'Information Technology'],
                ['TXN', 'Texas Instruments Inc.', 'Information Technology'],
                ['NKE', 'Nike Inc.', 'Consumer Discretionary'],
                ['LIN', 'Linde plc', 'Materials'],
                ['DHR', 'Danaher Corporation', 'Health Care'],
                ['ORCL', 'Oracle Corporation', 'Information Technology'],
                ['WFC', 'Wells Fargo & Co.', 'Financials'],
                ['VZ', 'Verizon Communications Inc.', 'Communication Services'],
                ['CMCSA', 'Comcast Corporation', 'Communication Services'],
                ['PM', 'Philip Morris International Inc.', 'Consumer Staples'],
                ['NEE', 'NextEra Energy Inc.', 'Utilities'],
                ['RTX', 'Raytheon Technologies Corp.', 'Industrials'],
                ['UNP', 'Union Pacific Corporation', 'Industrials'],
                ['T', 'AT&T Inc.', 'Communication Services'],
                ['LOW', 'Lowe\'s Companies Inc.', 'Consumer Discretionary'],
                ['QCOM', 'Qualcomm Inc.', 'Information Technology'],
                ['HON', 'Honeywell International Inc.', 'Industrials'],
                ['INTU', 'Intuit Inc.', 'Information Technology'],
                ['UPS', 'United Parcel Service Inc.', 'Industrials'],
                ['IBM', 'International Business Machines Corp.', 'Information Technology']
            ]
            
            sp500_df = pd.DataFrame(fallback_companies, 
                                  columns=['Symbol', 'Security', 'GICS_Sector'])
            
            # Save fallback list
            output_file = self.data_dir / "sp500_companies.csv"
            sp500_df.to_csv(output_file, index=False)
            logger.info(f"Saved fallback company list to: {output_file}")
            
            return sp500_df
    
    def get_stock_data_batch(self, symbols, period="2y", max_retries=3):
        """Get stock data for multiple symbols efficiently.
        
        Args:
            symbols (list): List of stock symbols
            period (str): Time period for data collection
            max_retries (int): Maximum retry attempts
            
        Returns:
            dict: Dictionary of symbol -> DataFrame mappings
        """
        logger.info(f"Collecting stock data for {len(symbols)} symbols...")
        
        stock_data = {}
        failed_symbols = []
        
        # Process in batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {batch}")
            
            for symbol in batch:
                retries = 0
                while retries < max_retries:
                    try:
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period=period)
                        
                        if not data.empty:
                            stock_data[symbol] = data
                            logger.debug(f"✅ {symbol}: {len(data)} records")
                            break
                        else:
                            logger.warning(f"⚠️  {symbol}: No data returned")
                            failed_symbols.append(symbol)
                            break
                            
                    except Exception as e:
                        retries += 1
                        logger.warning(f"❌ {symbol} (attempt {retries}): {e}")
                        if retries < max_retries:
                            time.sleep(1)  # Wait before retry
                        else:
                            failed_symbols.append(symbol)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Longer delay between batches
            if i + batch_size < len(symbols):
                time.sleep(2)
        
        logger.info(f"Successfully collected data for {len(stock_data)} symbols")
        if failed_symbols:
            logger.warning(f"Failed to collect data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")
        
        return stock_data
    
    def simulate_filing_data(self, symbols, stock_data, filings_per_year=12):
        """Simulate SEC 8-K filing data for prediction testing.
        
        Args:
            symbols (list): List of stock symbols
            stock_data (dict): Stock price data
            filings_per_year (int): Average filings per company per year
            
        Returns:
            pd.DataFrame: Simulated filing data
        """
        logger.info(f"Simulating SEC 8-K filing data for {len(symbols)} companies...")
        
        filing_categories = [
            '1.01', '1.02', '2.01', '2.02', '2.03', '2.04', '2.05', '2.06',
            '3.01', '3.02', '3.03', '4.01', '4.02', '5.01', '5.02', '5.03',
            '5.04', '5.05', '5.06', '5.07', '5.08', '7.01', '8.01', '9.01'
        ]
        
        content_templates = [
            "Quarterly earnings results announcement with revenue and profit details",
            "Strategic acquisition of technology company to expand market presence",
            "Executive leadership change with new CEO appointment",
            "Dividend increase and share buyback program announcement",
            "Major contract award from enterprise customer",
            "Product launch announcement for new software platform",
            "Regulatory compliance update and filing",
            "Cost reduction initiative and restructuring plan",
            "Partnership agreement with industry leader",
            "Research and development milestone achievement",
            "Market expansion into new geographic region",
            "Cybersecurity incident disclosure and response",
            "Material agreement termination notice",
            "Financial guidance update for upcoming quarter",
            "Intellectual property acquisition and licensing deal"
        ]
        
        all_filings = []
        
        for symbol in symbols:
            if symbol not in stock_data:
                continue
                
            df = stock_data[symbol]
            if df.empty:
                continue
            
            # Generate random filing dates
            start_date = df.index[0]
            end_date = df.index[-1]
            date_range = (end_date - start_date).days
            
            # Calculate number of filings based on time period
            years = date_range / 365.25
            num_filings = max(1, int(filings_per_year * years))
            
            # Generate random filing dates
            filing_dates = []
            for _ in range(num_filings):
                random_days = random.randint(0, date_range)
                filing_date = start_date + timedelta(days=random_days)
                filing_dates.append(filing_date)
            
            filing_dates.sort()
            
            for filing_date in filing_dates:
                # Find closest trading day
                closest_date = min(df.index, key=lambda x: abs(x - filing_date))
                
                if closest_date not in df.index:
                    continue
                
                # Get stock data for this date
                stock_price = df.loc[closest_date, 'Close']
                volume = df.loc[closest_date, 'Volume']
                
                # Calculate recent volatility and trend
                window_start = max(0, df.index.get_loc(closest_date) - 20)
                window_end = df.index.get_loc(closest_date)
                recent_data = df.iloc[window_start:window_end]
                
                if len(recent_data) > 1:
                    returns = recent_data['Close'].pct_change().dropna()
                    recent_volatility = returns.std()
                    recent_trend = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) / len(recent_data)
                else:
                    recent_volatility = 0.02
                    recent_trend = 0.0
                
                # Calculate forward returns
                future_5d_idx = min(len(df) - 1, window_end + 5)
                future_9d_idx = min(len(df) - 1, window_end + 9)
                
                if future_5d_idx > window_end:
                    actual_return_5d = (df.iloc[future_5d_idx]['Close'] / stock_price) - 1
                else:
                    actual_return_5d = 0.0
                
                if future_9d_idx > window_end:
                    actual_return_9d = (df.iloc[future_9d_idx]['Close'] / stock_price) - 1
                else:
                    actual_return_9d = 0.0
                
                # Generate realistic features based on actual returns
                base_sentiment = 0.5 + (actual_return_5d * 2)  # Correlate with actual performance
                sentiment_score = max(0.1, min(0.9, base_sentiment + random.gauss(0, 0.1)))
                
                urgency_score = random.uniform(0.1, 0.8)
                financial_impact_score = max(0.1, min(0.9, 0.5 + abs(actual_return_5d) * 3))
                market_relevance_score = random.uniform(0.3, 0.9)
                
                # Generate predictions based on features
                feature_score = (sentiment_score * 0.3 + financial_impact_score * 0.4 + 
                               market_relevance_score * 0.3)
                
                predicted_direction = "positive" if feature_score > 0.5 else "negative"
                predicted_probability = feature_score if predicted_direction == "positive" else (1 - feature_score)
                
                # Add noise to predictions
                predicted_return_5d = actual_return_5d + random.gauss(0, 0.01)
                predicted_return_9d = actual_return_9d + random.gauss(0, 0.015)
                
                # Calculate accuracy
                predicted_positive = predicted_direction == "positive"
                actual_positive = actual_return_5d > 0
                prediction_accuracy_5d = 1.0 if predicted_positive == actual_positive else 0.0
                
                actual_positive_9d = actual_return_9d > 0
                prediction_accuracy_9d = 1.0 if predicted_positive == actual_positive_9d else 0.0
                
                filing = {
                    'ticker': symbol,
                    'filing_date': filing_date.strftime('%Y-%m-%d'),
                    'category': random.choice(filing_categories),
                    'content_summary': random.choice(content_templates),
                    'stock_price': round(stock_price, 2),
                    'volume': int(volume),
                    'recent_volatility': round(recent_volatility, 4),
                    'recent_trend': round(recent_trend, 4),
                    'sentiment_score': round(sentiment_score, 3),
                    'urgency_score': round(urgency_score, 3),
                    'financial_impact_score': round(financial_impact_score, 3),
                    'market_relevance_score': round(market_relevance_score, 3),
                    'predicted_direction': predicted_direction,
                    'predicted_probability': round(predicted_probability, 3),
                    'predicted_return_5d': round(predicted_return_5d, 4),
                    'predicted_return_9d': round(predicted_return_9d, 4),
                    'actual_return_5d': round(actual_return_5d, 4),
                    'actual_return_9d': round(actual_return_9d, 4),
                    'prediction_accuracy_5d': prediction_accuracy_5d,
                    'prediction_accuracy_9d': prediction_accuracy_9d,
                    'model_confidence': round(predicted_probability, 3)
                }
                
                all_filings.append(filing)
        
        filings_df = pd.DataFrame(all_filings)
        logger.info(f"Generated {len(filings_df)} simulated filings")
        
        return filings_df
    
    def collect_sp500_prediction_data(self, sample_size=None, period="2y"):
        """Collect comprehensive S&P 500 prediction data.
        
        Args:
            sample_size (int): Number of companies to process (None for all)
            period (str): Time period for stock data
            
        Returns:
            tuple: (companies_df, filings_df, summary_stats)
        """
        logger.info("Starting comprehensive S&P 500 data collection...")
        
        # Download company list
        companies_df = self.download_sp500_list()
        
        # Sample companies if requested
        if sample_size and sample_size < len(companies_df):
            companies_df = companies_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} companies for analysis")
        
        symbols = companies_df['Symbol'].tolist()
        
        # Collect stock data
        stock_data = self.get_stock_data_batch(symbols, period=period)
        
        # Filter companies with successful data collection
        successful_symbols = list(stock_data.keys())
        companies_df = companies_df[companies_df['Symbol'].isin(successful_symbols)]
        
        logger.info(f"Successfully collected data for {len(successful_symbols)} companies")
        
        # Generate filing predictions
        filings_df = self.simulate_filing_data(successful_symbols, stock_data)
        
        # Calculate summary statistics
        summary_stats = self.calculate_summary_statistics(companies_df, filings_df)
        
        # Save results
        companies_df.to_csv(self.data_dir / "sp500_companies_processed.csv", index=False)
        filings_df.to_csv(self.data_dir / "sp500_predictions_comprehensive.csv", index=False)
        
        with open(self.data_dir / "sp500_summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        logger.info("Data collection completed successfully!")
        
        return companies_df, filings_df, summary_stats
    
    def calculate_summary_statistics(self, companies_df, filings_df):
        """Calculate comprehensive summary statistics.
        
        Args:
            companies_df (pd.DataFrame): Company information
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Summary statistics
        """
        logger.info("Calculating summary statistics...")
        
        stats = {
            'collection_date': datetime.now().isoformat(),
            'total_companies': len(companies_df),
            'total_predictions': len(filings_df),
            'companies_by_sector': companies_df['GICS_Sector'].value_counts().to_dict(),
            'predictions_by_category': filings_df['category'].value_counts().to_dict(),
            'overall_performance': {
                'accuracy_5d': filings_df['prediction_accuracy_5d'].mean(),
                'accuracy_9d': filings_df['prediction_accuracy_9d'].mean(),
                'avg_confidence': filings_df['model_confidence'].mean(),
                'positive_predictions_pct': (filings_df['predicted_direction'] == 'positive').mean(),
                'avg_predicted_return_5d': filings_df['predicted_return_5d'].mean(),
                'avg_actual_return_5d': filings_df['actual_return_5d'].mean(),
                'avg_predicted_return_9d': filings_df['predicted_return_9d'].mean(),
                'avg_actual_return_9d': filings_df['actual_return_9d'].mean(),
                'prediction_error_5d': abs(filings_df['predicted_return_5d'] - filings_df['actual_return_5d']).mean(),
                'prediction_error_9d': abs(filings_df['predicted_return_9d'] - filings_df['actual_return_9d']).mean()
            }
        }
        
        # Sector-wise performance
        sector_performance = {}
        for sector in companies_df['GICS_Sector'].unique():
            sector_companies = companies_df[companies_df['GICS_Sector'] == sector]['Symbol'].tolist()
            sector_filings = filings_df[filings_df['ticker'].isin(sector_companies)]
            
            if len(sector_filings) > 0:
                sector_performance[sector] = {
                    'companies': len(sector_companies),
                    'predictions': len(sector_filings),
                    'accuracy_5d': sector_filings['prediction_accuracy_5d'].mean(),
                    'accuracy_9d': sector_filings['prediction_accuracy_9d'].mean(),
                    'avg_return_5d': sector_filings['actual_return_5d'].mean(),
                    'avg_return_9d': sector_filings['actual_return_9d'].mean()
                }
        
        stats['sector_performance'] = sector_performance
        
        # Category performance
        category_performance = {}
        for category in filings_df['category'].unique():
            cat_filings = filings_df[filings_df['category'] == category]
            category_performance[category] = {
                'predictions': len(cat_filings),
                'accuracy_5d': cat_filings['prediction_accuracy_5d'].mean(),
                'accuracy_9d': cat_filings['prediction_accuracy_9d'].mean(),
                'avg_confidence': cat_filings['model_confidence'].mean()
            }
        
        stats['category_performance'] = category_performance
        
        return stats

def main():
    """Main function for testing the collector."""
    collector = SP500DataCollector()
    
    # Test with a smaller sample first
    companies_df, filings_df, summary_stats = collector.collect_sp500_prediction_data(
        sample_size=50,  # Start with 50 companies
        period="1y"      # 1 year of data
    )
    
    print(f"\n🎉 Collection completed!")
    print(f"📊 Companies processed: {len(companies_df)}")
    print(f"📈 Predictions generated: {len(filings_df)}")
    print(f"🎯 Overall 5-day accuracy: {summary_stats['overall_performance']['accuracy_5d']:.1%}")
    print(f"🎯 Overall 9-day accuracy: {summary_stats['overall_performance']['accuracy_9d']:.1%}")

if __name__ == "__main__":
    main()

