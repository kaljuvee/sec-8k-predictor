"""
SEC 8-K Predictor Streamlit Application

Main application file for the Streamlit web interface.
"""

import streamlit as st
import sys
from pathlib import Path
import sqlite3
import os

# Configure page first (must be first Streamlit command)
st.set_page_config(
    page_title="SEC 8-K Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Main page content
st.title("📈 SEC 8-K Predictor")
st.markdown("### Predict Stock Returns from SEC 8-K Filings")

# Check if databases exist
data_dir = Path("data")
filings_db = data_dir / "filings.db"
stock_db = data_dir / "stock_data.db"
features_db = data_dir / "features.db"

# Database status
col1, col2, col3 = st.columns(3)

with col1:
    if filings_db.exists():
        st.success("✅ SEC Filings Database")
        try:
            conn = sqlite3.connect(filings_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM filings")
            count = cursor.fetchone()[0]
            st.metric("SEC Filings", f"{count:,}")
            conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
    else:
        st.warning("⚠️ SEC Filings Database Missing")
        st.info("Use the Setup Database page to download SEC filings")

with col2:
    if stock_db.exists():
        st.success("✅ Stock Data Database")
        try:
            conn = sqlite3.connect(stock_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM stock_data")
            count = cursor.fetchone()[0]
            st.metric("Stock Records", f"{count:,}")
            conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
    else:
        st.warning("⚠️ Stock Data Database Missing")
        st.info("Use the Setup Database page to download stock data")

with col3:
    if features_db.exists():
        st.success("✅ Features Database")
        try:
            conn = sqlite3.connect(features_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM features")
            count = cursor.fetchone()[0]
            st.metric("Feature Records", f"{count:,}")
            conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
    else:
        st.warning("⚠️ Features Database Missing")
        st.info("Use the Setup Database page to extract features")

# System overview
st.markdown("---")
st.markdown("## 🎯 System Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Features
    - **SEC 8-K Filing Analysis**: Automated download and processing
    - **Stock Data Integration**: Real-time price and volume data
    - **Feature Extraction**: LLM-based content analysis
    - **Machine Learning Models**: Random Forest classifiers and regressors
    - **Prediction Interface**: Interactive prediction tools
    - **Backtesting Framework**: Historical performance analysis
    """)

with col2:
    st.markdown("""
    ### 🚀 Getting Started
    1. **Setup Database**: Download SEC filings and stock data
    2. **Extract Features**: Process filings with LLM analysis
    3. **Train Models**: Build prediction models
    4. **Make Predictions**: Predict stock returns from new filings
    5. **Analyze Performance**: Review model accuracy and insights
    """)

# Quick actions
st.markdown("---")
st.markdown("## ⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 Setup Database", use_container_width=True):
        st.switch_page("pages/0_🚀_Setup_Database.py")

with col2:
    if st.button("📊 View Data", use_container_width=True):
        st.switch_page("pages/1_📊_Data_Overview.py")

with col3:
    if st.button("🔍 Analyze Features", use_container_width=True):
        st.switch_page("pages/2_🔍_Feature_Analysis.py")

with col4:
    if st.button("🎯 Make Predictions", use_container_width=True):
        st.switch_page("pages/4_🎯_Predictions.py")

# Status summary
st.markdown("---")
st.markdown("## 📋 System Status")

# Check overall system readiness
databases_ready = all([filings_db.exists(), stock_db.exists(), features_db.exists()])

if databases_ready:
    st.success("🎉 **System Ready!** All databases are available. You can start making predictions.")
else:
    st.info("🔧 **Setup Required**: Please use the Setup Database page to initialize the system.")
    
    missing = []
    if not filings_db.exists():
        missing.append("SEC Filings")
    if not stock_db.exists():
        missing.append("Stock Data")
    if not features_db.exists():
        missing.append("Features")
    
    st.warning(f"Missing: {', '.join(missing)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SEC 8-K Predictor | Built with Streamlit | 
    <a href='https://github.com/kaljuvee/sec-8k-predictor' target='_blank'>GitHub Repository</a>
    </p>
</div>
""", unsafe_allow_html=True)

