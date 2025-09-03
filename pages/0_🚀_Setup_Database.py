"""
Database Setup Page for SEC 8-K Predictor

This page allows users to download and populate the databases with SEC filings and stock data.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import sqlite3
import pandas as pd
import time
import threading
from datetime import datetime, timedelta

# Add src directory to path
current_dir = Path(__file__).parent.parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from sec_downloader import SEC8KDownloader
    from stock_data import StockDataCollector
    from feature_extraction import FeatureExtractor
    from models import SEC8KPredictor
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure all required packages are installed: pip install -r requirements.txt")
    st.stop()

st.set_page_config(
    page_title="Database Setup",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Database Setup")
st.markdown("Set up your SEC 8-K Predictor databases with fresh data.")

# Initialize session state
if 'setup_status' not in st.session_state:
    st.session_state.setup_status = {
        'sec_filings': False,
        'stock_data': False,
        'features': False,
        'models': False
    }

if 'setup_progress' not in st.session_state:
    st.session_state.setup_progress = {
        'current_step': '',
        'progress': 0,
        'log': []
    }

def check_database_status():
    """Check the current status of databases"""
    data_dir = Path("data")
    
    status = {
        'sec_filings': False,
        'stock_data': False,
        'features': False,
        'models': False
    }
    
    # Check SEC filings database
    sec_db = data_dir / "sec_filings.db"
    if sec_db.exists():
        try:
            conn = sqlite3.connect(sec_db)
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM filings", conn)['count'][0]
            conn.close()
            if count > 0:
                status['sec_filings'] = count
        except:
            pass
    
    # Check stock data database
    stock_db = data_dir / "stock_data.db"
    if stock_db.exists():
        try:
            conn = sqlite3.connect(stock_db)
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_prices", conn)['count'][0]
            conn.close()
            if count > 0:
                status['stock_data'] = count
        except:
            pass
    
    # Check features database
    features_db = data_dir / "features.db"
    if features_db.exists():
        try:
            conn = sqlite3.connect(features_db)
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM features", conn)['count'][0]
            conn.close()
            if count > 0:
                status['features'] = count
        except:
            pass
    
    # Check models directory
    models_dir = data_dir / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.joblib"))
        if model_files:
            status['models'] = len(model_files)
    
    return status

def update_progress(step, progress, message):
    """Update progress in session state"""
    st.session_state.setup_progress['current_step'] = step
    st.session_state.setup_progress['progress'] = progress
    st.session_state.setup_progress['log'].append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

def download_sec_filings(tickers, start_date, end_date, limit):
    """Download SEC filings in background"""
    try:
        update_progress("SEC Filings", 10, f"Starting SEC filings download for {len(tickers)} tickers")
        
        downloader = SEC8KDownloader(data_dir="data")
        
        update_progress("SEC Filings", 30, "Downloading 8-K filings from EDGAR...")
        downloader.download_8k_filings(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=3,
            limit=limit
        )
        
        # Get summary
        summary = downloader.get_filings_summary()
        total_filings = len(summary)
        
        update_progress("SEC Filings", 100, f"✅ Downloaded {total_filings} SEC filings successfully")
        st.session_state.setup_status['sec_filings'] = True
        
    except Exception as e:
        update_progress("SEC Filings", 0, f"❌ Error downloading SEC filings: {str(e)}")

def download_stock_data(tickers, start_date, end_date, limit):
    """Download stock data in background"""
    try:
        update_progress("Stock Data", 10, f"Starting stock data download for {len(tickers)} tickers")
        
        collector = StockDataCollector(data_dir="data")
        
        update_progress("Stock Data", 30, "Downloading stock prices from Yahoo Finance...")
        collector.download_stock_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=5,
            limit=limit
        )
        
        update_progress("Stock Data", 70, "Calculating returns and volatility...")
        collector.calculate_returns(tickers[:limit] if limit else tickers)
        
        # Get summary
        summary = collector.get_stock_data_summary()
        total_records = len(summary)
        
        update_progress("Stock Data", 100, f"✅ Downloaded {total_records} stock data records successfully")
        st.session_state.setup_status['stock_data'] = True
        
    except Exception as e:
        update_progress("Stock Data", 0, f"❌ Error downloading stock data: {str(e)}")

def extract_features(limit):
    """Extract features in background"""
    try:
        update_progress("Features", 10, "Starting feature extraction")
        
        extractor = FeatureExtractor(data_dir="data")
        
        update_progress("Features", 30, "Processing filings with LLM analysis...")
        extractor.process_filings(limit=limit)
        
        update_progress("Features", 70, "Creating TF-IDF features...")
        extractor.create_tfidf_features()
        
        # Get summary
        summary = extractor.get_features_summary()
        total_features = len(summary)
        
        update_progress("Features", 100, f"✅ Extracted features for {total_features} filings successfully")
        st.session_state.setup_status['features'] = True
        
    except Exception as e:
        update_progress("Features", 0, f"❌ Error extracting features: {str(e)}")

def train_models():
    """Train models in background"""
    try:
        update_progress("Models", 10, "Starting model training")
        
        predictor = SEC8KPredictor(data_dir="data")
        
        update_progress("Models", 50, "Training Random Forest models...")
        results = predictor.train_all_models(['relative_return_5d'])
        
        total_models = len(results['classifiers']) + len(results['regressors'])
        
        update_progress("Models", 100, f"✅ Trained {total_models} models successfully")
        st.session_state.setup_status['models'] = True
        
    except Exception as e:
        update_progress("Models", 0, f"❌ Error training models: {str(e)}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Database Status")
    
    # Check current status
    status = check_database_status()
    
    # Display status
    status_cols = st.columns(4)
    
    with status_cols[0]:
        if status['sec_filings']:
            st.success(f"SEC Filings\n{status['sec_filings']} records")
        else:
            st.error("SEC Filings\nNot available")
    
    with status_cols[1]:
        if status['stock_data']:
            st.success(f"Stock Data\n{status['stock_data']} records")
        else:
            st.error("Stock Data\nNot available")
    
    with status_cols[2]:
        if status['features']:
            st.success(f"Features\n{status['features']} records")
        else:
            st.error("Features\nNot available")
    
    with status_cols[3]:
        if status['models']:
            st.success(f"Models\n{status['models']} files")
        else:
            st.error("Models\nNot available")

with col2:
    st.header("Quick Setup")
    
    if st.button("🚀 Quick Demo Setup", type="primary", use_container_width=True):
        st.info("Starting quick demo setup with sample data...")
        
        # Quick setup with minimal data
        sample_tickers = ['AAPL', 'MSFT', 'GOOGL']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Start background processes
        threading.Thread(target=download_sec_filings, args=(sample_tickers, start_date, end_date, 3)).start()

st.header("Custom Setup")

# Configuration options
with st.expander("⚙️ Configuration Options", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Selection")
        
        # Ticker selection
        ticker_option = st.radio(
            "Select tickers:",
            ["Sample (AAPL, MSFT, GOOGL)", "Top 10 S&P 500", "Custom list"]
        )
        
        if ticker_option == "Custom list":
            custom_tickers = st.text_input(
                "Enter tickers (comma-separated):",
                value="AAPL,MSFT,GOOGL,TSLA,AMZN"
            )
            tickers = [t.strip().upper() for t in custom_tickers.split(',')]
        elif ticker_option == "Top 10 S&P 500":
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ']
        else:
            tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        st.write(f"Selected tickers: {', '.join(tickers)}")
        
        # Date range
        end_date = st.date_input("End date", value=datetime.now())
        start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=365))
        
        # Limits
        filing_limit = st.number_input("Max companies for SEC filings", min_value=1, max_value=50, value=len(tickers))
        feature_limit = st.number_input("Max filings for feature extraction", min_value=1, max_value=100, value=20)
    
    with col2:
        st.subheader("API Configuration")
        
        # OpenAI API key
        openai_key = st.text_input(
            "OpenAI API Key (for feature extraction):",
            type="password",
            help="Required for LLM-based feature extraction"
        )
        
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            st.success("✅ OpenAI API key configured")
        else:
            st.warning("⚠️ OpenAI API key required for feature extraction")
        
        st.subheader("Setup Steps")
        
        step1 = st.checkbox("1. Download SEC filings", value=True)
        step2 = st.checkbox("2. Download stock data", value=True)
        step3 = st.checkbox("3. Extract features", value=True, disabled=not openai_key)
        step4 = st.checkbox("4. Train models", value=True)

# Setup buttons
st.header("Run Setup")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Reset All Databases", type="secondary", use_container_width=True):
        # Clear databases
        data_dir = Path("data")
        for db_file in data_dir.glob("*.db"):
            try:
                db_file.unlink()
                st.success(f"Deleted {db_file.name}")
            except:
                st.error(f"Failed to delete {db_file.name}")
        
        # Clear models
        models_dir = data_dir / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.joblib"):
                try:
                    model_file.unlink()
                    st.success(f"Deleted {model_file.name}")
                except:
                    st.error(f"Failed to delete {model_file.name}")
        
        st.session_state.setup_status = {
            'sec_filings': False,
            'stock_data': False,
            'features': False,
            'models': False
        }
        st.rerun()

with col2:
    if st.button("▶️ Run Selected Steps", type="primary", use_container_width=True):
        if not any([step1, step2, step3, step4]):
            st.error("Please select at least one setup step")
        else:
            st.info("Starting setup process...")
            
            # Convert dates to strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Run selected steps in sequence
            if step1:
                threading.Thread(target=download_sec_filings, args=(tickers, start_date_str, end_date_str, filing_limit)).start()
            
            if step2:
                threading.Thread(target=download_stock_data, args=(tickers, start_date_str, end_date_str, filing_limit)).start()
            
            if step3 and openai_key:
                threading.Thread(target=extract_features, args=(feature_limit,)).start()
            
            if step4:
                threading.Thread(target=train_models).start()

with col3:
    if st.button("📊 Check Status", use_container_width=True):
        st.rerun()

# Progress monitoring
if st.session_state.setup_progress['current_step']:
    st.header("Setup Progress")
    
    # Progress bar
    progress_bar = st.progress(st.session_state.setup_progress['progress'] / 100)
    st.write(f"Current step: {st.session_state.setup_progress['current_step']}")
    
    # Progress log
    if st.session_state.setup_progress['log']:
        with st.expander("📋 Setup Log", expanded=True):
            for log_entry in st.session_state.setup_progress['log'][-10:]:  # Show last 10 entries
                st.text(log_entry)

# Auto-refresh during setup
if st.session_state.setup_progress['current_step']:
    time.sleep(2)
    st.rerun()

# Tips and information
st.header("💡 Setup Tips")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Quick Start:**
    1. Use the "Quick Demo Setup" for fastest results
    2. Ensure you have a valid OpenAI API key
    3. Start with sample data (3 companies)
    4. Monitor the progress log for any issues
    """)

with col2:
    st.warning("""
    **Important Notes:**
    - SEC filing download may take 5-10 minutes
    - Feature extraction requires OpenAI API credits
    - Stock data download is usually fastest
    - Model training needs sufficient data (10+ samples)
    """)

# Data directory info
st.header("📁 Data Directory Structure")

data_dir = Path("data")
if data_dir.exists():
    st.code(f"""
data/
├── sec_filings.db      # SEC 8-K filings database
├── stock_data.db       # Stock price and returns database  
├── features.db         # Extracted features database
└── models/             # Trained ML model files
    ├── *.joblib        # Model files
    └── training_results.json
    """)
else:
    st.info("Data directory will be created during setup.")

