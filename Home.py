"""
SEC 8-K Predictor Streamlit Application

Main application file for the Streamlit web interface.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure page
st.set_page_config(
    page_title="SEC 8-K Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("📈 SEC 8-K Predictor")
st.markdown("### Predict Stock Returns from SEC 8-K Filings")

# Check if databases exist
import sqlite3
from pathlib import Path

data_dir = Path("data")
databases_exist = (
    (data_dir / "sec_filings.db").exists() and
    (data_dir / "stock_data.db").exists()
)

if not databases_exist:
    st.warning("""
    🚀 **Welcome to SEC 8-K Predictor!** 
    
    To get started, please set up your databases first by visiting the **🚀 Setup Database** page in the sidebar.
    
    This will download SEC filings and stock data to populate your local databases.
    """)
    
    if st.button("🚀 Go to Database Setup", type="primary"):
        st.switch_page("pages/0_🚀_Setup_Database.py")
else:
    st.success("✅ Databases are ready! Explore the application using the sidebar navigation.")

st.markdown("""
Welcome to the SEC 8-K Predictor! This application uses machine learning to predict stock returns 
based on SEC 8-K filing content and market data.

**Features:**
- 🚀 **Setup Database**: Download and populate databases with fresh data
- 📊 **Data Overview**: Explore SEC filings and stock data
- 🔍 **Feature Analysis**: Analyze extracted features from filings
- 🤖 **Model Performance**: View machine learning model results
- 🎯 **Predictions**: Make predictions on new filings
- 📈 **Visualizations**: Interactive charts and graphs

**Navigation:**
Use the sidebar to navigate between different pages of the application.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page to explore different features of the SEC 8-K Predictor.")

# Data status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Status")

try:
    from src.sec_downloader import SEC8KDownloader
    from src.stock_data import StockDataCollector
    from src.feature_extraction import FeatureExtractor
    from src.models import SEC8KPredictor
    
    data_dir = "data"
    
    # Check data availability
    filings_db = Path(data_dir) / "sec_filings.db"
    stock_db = Path(data_dir) / "stock_data.db"
    features_db = Path(data_dir) / "features.db"
    models_dir = Path(data_dir) / "models"
    
    st.sidebar.markdown(f"**SEC Filings:** {'✅' if filings_db.exists() else '❌'}")
    st.sidebar.markdown(f"**Stock Data:** {'✅' if stock_db.exists() else '❌'}")
    st.sidebar.markdown(f"**Features:** {'✅' if features_db.exists() else '❌'}")
    st.sidebar.markdown(f"**Models:** {'✅' if models_dir.exists() and list(models_dir.glob('*.joblib')) else '❌'}")
    
except Exception as e:
    st.sidebar.error(f"Error checking data status: {e}")

# Quick actions
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Refresh Data Status"):
    st.rerun()

# Instructions
st.markdown("---")
st.markdown("### Getting Started")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **1. Data Collection**
    - Download SEC 8-K filings
    - Collect stock price data
    - Extract features from filings
    """)

with col2:
    st.markdown("""
    **2. Model Training**
    - Train machine learning models
    - Evaluate model performance
    - Make predictions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SEC 8-K Predictor | Built with Streamlit and scikit-learn</p>
</div>
""", unsafe_allow_html=True)

