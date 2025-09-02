"""
Data Overview Page

Shows overview of SEC filings and stock data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.sec_downloader import SEC8KDownloader
from src.stock_data import StockDataCollector

st.title("📊 Data Overview")
st.markdown("Explore the SEC filings and stock data in the system.")

# Data directory
data_dir = "data"

# Tabs for different data views
tab1, tab2, tab3 = st.tabs(["SEC Filings", "Stock Data", "Data Quality"])

with tab1:
    st.header("SEC 8-K Filings")
    
    try:
        downloader = SEC8KDownloader(data_dir=data_dir)
        filings_summary = downloader.get_filings_summary()
        
        if not filings_summary.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_filings = filings_summary['count'].sum()
                st.metric("Total Filings", f"{total_filings:,}")
            
            with col2:
                unique_companies = filings_summary['ticker'].nunique()
                st.metric("Companies", unique_companies)
            
            with col3:
                categories = filings_summary['category'].nunique()
                st.metric("Categories", categories)
            
            with col4:
                date_range = f"{filings_summary['earliest_date'].min()} to {filings_summary['latest_date'].max()}"
                st.metric("Date Range", date_range)
            
            # Filings by category
            st.subheader("Filings by Category")
            category_counts = filings_summary.groupby('category')['count'].sum().reset_index()
            category_counts = category_counts.sort_values('count', ascending=False)
            
            fig_cat = px.bar(
                category_counts, 
                x='category', 
                y='count',
                title="Number of Filings by 8-K Category",
                labels={'category': '8-K Category', 'count': 'Number of Filings'}
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Filings by company (top 20)
            st.subheader("Top 20 Companies by Filing Count")
            company_counts = filings_summary.groupby('ticker')['count'].sum().reset_index()
            company_counts = company_counts.sort_values('count', ascending=False).head(20)
            
            fig_comp = px.bar(
                company_counts,
                x='ticker',
                y='count',
                title="Top 20 Companies by Number of Filings",
                labels={'ticker': 'Company Ticker', 'count': 'Number of Filings'}
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Filings Summary")
            st.dataframe(filings_summary, use_container_width=True)
            
        else:
            st.warning("No SEC filings data found. Please download filings first.")
            
    except Exception as e:
        st.error(f"Error loading SEC filings data: {e}")

with tab2:
    st.header("Stock Price Data")
    
    try:
        collector = StockDataCollector(data_dir=data_dir)
        stock_summary = collector.get_stock_data_summary()
        returns_summary = collector.get_returns_summary()
        
        if not stock_summary.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_records = stock_summary['records'].sum()
                st.metric("Total Records", f"{total_records:,}")
            
            with col2:
                unique_stocks = len(stock_summary)
                st.metric("Stocks", unique_stocks)
            
            with col3:
                avg_volume = stock_summary['avg_volume'].mean()
                st.metric("Avg Daily Volume", f"{avg_volume:,.0f}")
            
            with col4:
                date_range = f"{stock_summary['start_date'].min()} to {stock_summary['end_date'].max()}"
                st.metric("Date Range", date_range)
            
            # Returns analysis
            if not returns_summary.empty:
                st.subheader("Returns Analysis")
                
                # Average returns by stock
                fig_returns = px.scatter(
                    returns_summary,
                    x='avg_5d_return',
                    y='avg_relative_5d_return',
                    hover_data=['ticker'],
                    title="Average 5-Day Returns vs Relative Returns",
                    labels={
                        'avg_5d_return': 'Average 5-Day Return',
                        'avg_relative_5d_return': 'Average Relative 5-Day Return'
                    }
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # Volatility distribution
                fig_vol = px.histogram(
                    returns_summary,
                    x='avg_volatility',
                    title="Distribution of Average Volatility",
                    labels={'avg_volatility': 'Average Volatility'}
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Detailed tables
            st.subheader("Stock Data Summary")
            st.dataframe(stock_summary, use_container_width=True)
            
            if not returns_summary.empty:
                st.subheader("Returns Summary")
                st.dataframe(returns_summary, use_container_width=True)
            
        else:
            st.warning("No stock data found. Please download stock data first.")
            
    except Exception as e:
        st.error(f"Error loading stock data: {e}")

with tab3:
    st.header("Data Quality Assessment")
    
    try:
        # Check data completeness
        filings_db = Path(data_dir) / "sec_filings.db"
        stock_db = Path(data_dir) / "stock_data.db"
        features_db = Path(data_dir) / "features.db"
        
        quality_data = []
        
        if filings_db.exists():
            conn = sqlite3.connect(filings_db)
            
            # Filings with content
            total_filings = pd.read_sql_query("SELECT COUNT(*) as count FROM filings", conn)['count'][0]
            filings_with_content = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM filings WHERE content IS NOT NULL", conn
            )['count'][0]
            
            quality_data.append({
                'Dataset': 'SEC Filings',
                'Total Records': total_filings,
                'Complete Records': filings_with_content,
                'Completeness %': (filings_with_content / total_filings * 100) if total_filings > 0 else 0
            })
            
            conn.close()
        
        if stock_db.exists():
            conn = sqlite3.connect(stock_db)
            
            # Stock data completeness
            total_stock_records = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_prices", conn)['count'][0]
            returns_records = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_returns", conn)['count'][0]
            
            quality_data.append({
                'Dataset': 'Stock Prices',
                'Total Records': total_stock_records,
                'Complete Records': total_stock_records,
                'Completeness %': 100.0
            })
            
            quality_data.append({
                'Dataset': 'Stock Returns',
                'Total Records': returns_records,
                'Complete Records': returns_records,
                'Completeness %': 100.0
            })
            
            conn.close()
        
        if features_db.exists():
            conn = sqlite3.connect(features_db)
            
            # Features completeness
            total_features = pd.read_sql_query("SELECT COUNT(*) as count FROM features", conn)['count'][0]
            complete_features = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM features WHERE tfidf_features IS NOT NULL", conn
            )['count'][0]
            
            quality_data.append({
                'Dataset': 'Features',
                'Total Records': total_features,
                'Complete Records': complete_features,
                'Completeness %': (complete_features / total_features * 100) if total_features > 0 else 0
            })
            
            conn.close()
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            
            # Quality metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_completeness = quality_df['Completeness %'].mean()
                st.metric("Average Completeness", f"{avg_completeness:.1f}%")
            
            with col2:
                total_records = quality_df['Total Records'].sum()
                st.metric("Total Records", f"{total_records:,}")
            
            with col3:
                complete_records = quality_df['Complete Records'].sum()
                st.metric("Complete Records", f"{complete_records:,}")
            
            # Quality chart
            fig_quality = px.bar(
                quality_df,
                x='Dataset',
                y='Completeness %',
                title="Data Completeness by Dataset",
                labels={'Completeness %': 'Completeness (%)'}
            )
            fig_quality.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_quality, use_container_width=True)
            
            # Quality table
            st.subheader("Data Quality Summary")
            st.dataframe(quality_df, use_container_width=True)
            
        else:
            st.warning("No data quality information available.")
            
    except Exception as e:
        st.error(f"Error assessing data quality: {e}")

# Download section
st.markdown("---")
st.header("Data Management")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Download Data")
    st.markdown("Use the CLI to download data:")
    st.code("""
# Download SEC filings (sample)
python cli.py download filings --limit 10

# Download stock data (sample)
python cli.py download stocks --limit 10

# Extract features
python cli.py features extract --limit 10
    """)

with col2:
    st.subheader("Quick Start")
    st.markdown("Run end-to-end test:")
    st.code("""
# Run quickstart with sample data
python cli.py quickstart --sample-size 5
    """)

