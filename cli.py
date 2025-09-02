#!/usr/bin/env python3
"""
SEC 8-K Predictor CLI Application

Command-line interface for the SEC 8-K filing predictor system.
"""

import click
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.sec_downloader import SEC8KDownloader
from src.stock_data import StockDataCollector
from src.feature_extraction import FeatureExtractor
from src.models import SEC8KPredictor
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.option('--data-dir', default='data', help='Data directory path')
@click.pass_context
def cli(ctx, data_dir):
    """SEC 8-K Predictor - Predict stock returns from SEC filings"""
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True)

@cli.group()
@click.pass_context
def download(ctx):
    """Download SEC filings and stock data"""
    pass

@download.command('filings')
@click.option('--tickers', help='Comma-separated list of tickers (default: S&P 500)')
@click.option('--start-date', default='2014-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--batch-size', default=10, help='Batch size for downloads')
@click.option('--limit', type=int, help='Limit number of tickers (for testing)')
@click.pass_context
def download_filings(ctx, tickers, start_date, end_date, batch_size, limit):
    """Download SEC 8-K filings"""
    data_dir = ctx.obj['data_dir']
    
    click.echo(f"Downloading SEC 8-K filings to {data_dir}")
    
    downloader = SEC8KDownloader(data_dir=data_dir)
    
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(',')]
    else:
        click.echo("Getting S&P 500 tickers...")
        ticker_list = downloader.get_sp500_tickers()
        
    if limit:
        ticker_list = ticker_list[:limit]
        
    click.echo(f"Downloading filings for {len(ticker_list)} tickers")
    
    downloader.download_8k_filings(
        tickers=ticker_list,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size
    )
    
    # Show summary
    summary = downloader.get_filings_summary()
    click.echo(f"\nDownload completed!")
    click.echo(f"Total filings: {summary['count'].sum()}")
    click.echo(f"Companies: {summary['ticker'].nunique()}")

@download.command('stocks')
@click.option('--tickers', help='Comma-separated list of tickers (default: S&P 500)')
@click.option('--start-date', default='2014-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--batch-size', default=10, help='Batch size for downloads')
@click.option('--limit', type=int, help='Limit number of tickers (for testing)')
@click.pass_context
def download_stocks(ctx, tickers, start_date, end_date, batch_size, limit):
    """Download stock price data"""
    data_dir = ctx.obj['data_dir']
    
    click.echo(f"Downloading stock data to {data_dir}")
    
    collector = StockDataCollector(data_dir=data_dir)
    
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(',')]
    else:
        click.echo("Getting S&P 500 tickers...")
        ticker_list = collector.get_sp500_tickers()
        
    if limit:
        ticker_list = ticker_list[:limit]
        
    click.echo(f"Downloading stock data for {len(ticker_list)} tickers")
    
    collector.download_stock_data(
        tickers=ticker_list,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size
    )
    
    # Calculate returns
    click.echo("Calculating returns and volatility metrics...")
    collector.calculate_returns(ticker_list)
    
    # Show summary
    summary = collector.get_stock_data_summary()
    click.echo(f"\nDownload completed!")
    click.echo(f"Total records: {summary['records'].sum()}")
    click.echo(f"Companies: {len(summary)}")

@cli.group()
@click.pass_context
def features(ctx):
    """Extract features from filings"""
    pass

@features.command('extract')
@click.option('--limit', type=int, help='Limit number of filings to process')
@click.pass_context
def extract_features(ctx, limit):
    """Extract features from SEC filings"""
    data_dir = ctx.obj['data_dir']
    
    click.echo("Extracting features from SEC filings...")
    
    extractor = FeatureExtractor(data_dir=data_dir)
    
    # Process filings
    extractor.process_filings(limit=limit)
    
    # Create TF-IDF features
    click.echo("Creating TF-IDF features...")
    extractor.create_tfidf_features()
    
    # Show summary
    summary = extractor.get_features_summary()
    click.echo(f"\nFeature extraction completed!")
    click.echo(summary.to_string(index=False))

@features.command('summary')
@click.pass_context
def features_summary(ctx):
    """Show features summary"""
    data_dir = ctx.obj['data_dir']
    
    extractor = FeatureExtractor(data_dir=data_dir)
    summary = extractor.get_features_summary()
    
    if summary.empty:
        click.echo("No features found. Run 'features extract' first.")
    else:
        click.echo("Features Summary:")
        click.echo(summary.to_string(index=False))

@cli.group()
@click.pass_context
def models(ctx):
    """Train and manage ML models"""
    pass

@models.command('train')
@click.option('--categories', help='Comma-separated list of categories (default: all)')
@click.option('--targets', default='relative_return_5d,relative_return_9d', 
              help='Comma-separated list of target variables')
@click.pass_context
def train_models(ctx, categories, targets):
    """Train machine learning models"""
    data_dir = ctx.obj['data_dir']
    
    click.echo("Training machine learning models...")
    
    predictor = SEC8KPredictor(data_dir=data_dir)
    
    target_list = [t.strip() for t in targets.split(',')]
    
    if categories:
        # Train specific categories
        category_list = [c.strip() for c in categories.split(',')]
        results = {'classifiers': {}, 'regressors': {}, 'summary': {}}
        
        for category in category_list:
            for target in target_list:
                try:
                    classifier_results = predictor.train_classifier(category, target)
                    if classifier_results:
                        results['classifiers'][f"{category}_{target}"] = classifier_results
                    
                    regressor_results = predictor.train_regressor(category, target)
                    if regressor_results:
                        results['regressors'][f"{category}_{target}"] = regressor_results
                        
                except Exception as e:
                    click.echo(f"Error training {category} - {target}: {e}")
    else:
        # Train all available categories
        results = predictor.train_all_models(target_list)
    
    click.echo(f"\nTraining completed!")
    click.echo(f"Classifiers trained: {len(results['classifiers'])}")
    click.echo(f"Regressors trained: {len(results['regressors'])}")

@models.command('summary')
@click.pass_context
def models_summary(ctx):
    """Show models summary"""
    data_dir = ctx.obj['data_dir']
    
    predictor = SEC8KPredictor(data_dir=data_dir)
    summary = predictor.get_model_summary()
    
    if summary.empty:
        click.echo("No trained models found. Run 'models train' first.")
    else:
        click.echo("Models Summary:")
        click.echo(summary.to_string(index=False))

@models.command('predict')
@click.option('--category', required=True, help='8-K category')
@click.option('--model-type', type=click.Choice(['classifier', 'regressor']), 
              default='regressor', help='Model type')
@click.option('--target', default='relative_return_5d', help='Target variable')
@click.option('--ticker', required=True, help='Stock ticker')
@click.option('--filing-date', required=True, help='Filing date (YYYY-MM-DD)')
@click.pass_context
def predict(ctx, category, model_type, target, ticker, filing_date):
    """Make prediction for a specific filing"""
    data_dir = ctx.obj['data_dir']
    
    click.echo(f"Making prediction for {ticker} filing on {filing_date}")
    
    predictor = SEC8KPredictor(data_dir=data_dir)
    
    # This would require implementing a method to get features for a specific filing
    # For now, just show that the command structure is in place
    click.echo(f"Category: {category}")
    click.echo(f"Model type: {model_type}")
    click.echo(f"Target: {target}")
    click.echo("Note: Prediction functionality requires additional implementation")

@cli.group()
@click.pass_context
def status(ctx):
    """Check system status"""
    pass

@status.command('data')
@click.pass_context
def data_status(ctx):
    """Show data status"""
    data_dir = ctx.obj['data_dir']
    data_path = Path(data_dir)
    
    click.echo(f"Data Directory: {data_path.absolute()}")
    click.echo(f"Exists: {data_path.exists()}")
    
    if data_path.exists():
        # Check databases
        databases = {
            'SEC Filings': data_path / 'sec_filings.db',
            'Stock Data': data_path / 'stock_data.db',
            'Features': data_path / 'features.db'
        }
        
        click.echo("\nDatabases:")
        for name, db_path in databases.items():
            size = db_path.stat().st_size if db_path.exists() else 0
            click.echo(f"  {name}: {'✓' if db_path.exists() else '✗'} ({size:,} bytes)")
        
        # Check models
        models_dir = data_path / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.joblib'))
            click.echo(f"\nTrained Models: {len(model_files)}")
        else:
            click.echo("\nTrained Models: 0")

@cli.command()
@click.option('--sample-size', default=5, help='Number of companies for sample run')
@click.pass_context
def quickstart(ctx, sample_size):
    """Run a quick end-to-end test with sample data"""
    data_dir = ctx.obj['data_dir']
    
    click.echo(f"Running quickstart with {sample_size} companies...")
    
    # Sample tickers
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"][:sample_size]
    
    try:
        # 1. Download filings
        click.echo("1. Downloading SEC filings...")
        downloader = SEC8KDownloader(data_dir=data_dir)
        downloader.download_8k_filings(
            tickers=sample_tickers,
            start_date="2023-01-01",
            end_date="2024-12-31",
            batch_size=sample_size
        )
        
        # 2. Download stock data
        click.echo("2. Downloading stock data...")
        collector = StockDataCollector(data_dir=data_dir)
        collector.download_stock_data(
            tickers=sample_tickers,
            start_date="2023-01-01",
            end_date="2024-12-31",
            batch_size=sample_size
        )
        collector.calculate_returns(sample_tickers)
        
        # 3. Extract features
        click.echo("3. Extracting features...")
        extractor = FeatureExtractor(data_dir=data_dir)
        extractor.process_filings(limit=10)
        extractor.create_tfidf_features()
        
        # 4. Train models
        click.echo("4. Training models...")
        predictor = SEC8KPredictor(data_dir=data_dir)
        results = predictor.train_all_models()
        
        click.echo(f"\nQuickstart completed!")
        click.echo(f"Classifiers trained: {len(results['classifiers'])}")
        click.echo(f"Regressors trained: {len(results['regressors'])}")
        
    except Exception as e:
        click.echo(f"Error during quickstart: {e}")
        logger.error(f"Quickstart error: {e}")

if __name__ == '__main__':
    cli()

