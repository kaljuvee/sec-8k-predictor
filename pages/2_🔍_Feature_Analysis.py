"""
Feature Analysis Page

Analyze extracted features from SEC filings.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.feature_extraction import FeatureExtractor

st.title("🔍 Feature Analysis")
st.markdown("Analyze features extracted from SEC 8-K filings.")

# Data directory
data_dir = "data"

try:
    extractor = FeatureExtractor(data_dir=data_dir)
    features_summary = extractor.get_features_summary()
    
    if not features_summary.empty:
        # Summary metrics
        st.header("Feature Extraction Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_filings = features_summary['total_filings'].sum()
            st.metric("Total Filings", f"{total_filings:,}")
        
        with col2:
            with_targets = features_summary['with_targets'].sum()
            st.metric("With Targets", f"{with_targets:,}")
        
        with col3:
            categories = len(features_summary)
            st.metric("Categories", categories)
        
        with col4:
            completion_rate = (with_targets / total_filings * 100) if total_filings > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Features by category
        st.subheader("Features by Category")
        
        fig_cat = px.bar(
            features_summary,
            x='category',
            y='total_filings',
            title="Number of Processed Filings by Category",
            labels={'category': '8-K Category', 'total_filings': 'Number of Filings'}
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # LLM Feature Analysis
        st.header("LLM Feature Analysis")
        
        # Average scores by category
        score_columns = ['avg_sentiment', 'avg_urgency', 'avg_financial_impact', 'avg_market_relevance']
        
        if all(col in features_summary.columns for col in score_columns):
            # Radar chart for average scores
            categories_list = features_summary['category'].tolist()
            
            fig_radar = go.Figure()
            
            for idx, row in features_summary.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['avg_sentiment'], row['avg_urgency'], 
                       row['avg_financial_impact'], row['avg_market_relevance']],
                    theta=['Sentiment', 'Urgency', 'Financial Impact', 'Market Relevance'],
                    fill='toself',
                    name=f"Category {row['category']}"
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=True,
                title="Average LLM Scores by Category"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Individual score distributions
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sent = px.bar(
                    features_summary,
                    x='category',
                    y='avg_sentiment',
                    title="Average Sentiment Score by Category",
                    labels={'avg_sentiment': 'Average Sentiment Score'}
                )
                st.plotly_chart(fig_sent, use_container_width=True)
                
                fig_fin = px.bar(
                    features_summary,
                    x='category',
                    y='avg_financial_impact',
                    title="Average Financial Impact Score by Category",
                    labels={'avg_financial_impact': 'Average Financial Impact Score'}
                )
                st.plotly_chart(fig_fin, use_container_width=True)
            
            with col2:
                fig_urg = px.bar(
                    features_summary,
                    x='category',
                    y='avg_urgency',
                    title="Average Urgency Score by Category",
                    labels={'avg_urgency': 'Average Urgency Score'}
                )
                st.plotly_chart(fig_urg, use_container_width=True)
                
                fig_mkt = px.bar(
                    features_summary,
                    x='category',
                    y='avg_market_relevance',
                    title="Average Market Relevance Score by Category",
                    labels={'avg_market_relevance': 'Average Market Relevance Score'}
                )
                st.plotly_chart(fig_mkt, use_container_width=True)
        
        # Target Variable Analysis
        st.header("Target Variable Analysis")
        
        # Load detailed features data for target analysis
        conn = sqlite3.connect(Path(data_dir) / "features.db")
        
        target_query = """
            SELECT category, return_5d, return_9d, relative_return_5d, relative_return_9d, volatility_change_5d
            FROM features 
            WHERE return_5d IS NOT NULL
        """
        
        target_data = pd.read_sql_query(target_query, conn)
        conn.close()
        
        if not target_data.empty:
            # Target distributions
            col1, col2 = st.columns(2)
            
            with col1:
                fig_ret5 = px.histogram(
                    target_data,
                    x='relative_return_5d',
                    color='category',
                    title="Distribution of 5-Day Relative Returns",
                    labels={'relative_return_5d': '5-Day Relative Return'}
                )
                st.plotly_chart(fig_ret5, use_container_width=True)
                
                fig_vol = px.histogram(
                    target_data,
                    x='volatility_change_5d',
                    color='category',
                    title="Distribution of Volatility Changes",
                    labels={'volatility_change_5d': 'Volatility Change'}
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col2:
                fig_ret9 = px.histogram(
                    target_data,
                    x='relative_return_9d',
                    color='category',
                    title="Distribution of 9-Day Relative Returns",
                    labels={'relative_return_9d': '9-Day Relative Return'}
                )
                st.plotly_chart(fig_ret9, use_container_width=True)
                
                # Box plot of returns by category
                fig_box = px.box(
                    target_data,
                    x='category',
                    y='relative_return_5d',
                    title="5-Day Relative Returns by Category",
                    labels={'relative_return_5d': '5-Day Relative Return'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Correlation Analysis")
            
            # Calculate correlations
            numeric_cols = ['return_5d', 'return_9d', 'relative_return_5d', 'relative_return_9d', 'volatility_change_5d']
            corr_data = target_data[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_data,
                title="Correlation Matrix of Target Variables",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Summary statistics
            st.subheader("Target Variable Statistics")
            st.dataframe(target_data.describe(), use_container_width=True)
        
        # Feature Importance (if available)
        st.header("Feature Importance Analysis")
        
        # Check if we have trained models with feature importance
        models_dir = Path(data_dir) / "models"
        results_file = models_dir / "training_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                training_results = json.load(f)
            
            # Extract feature importance from models
            importance_data = []
            
            for model_key, model_results in training_results.get('regressors', {}).items():
                if 'feature_importance' in model_results:
                    for feat in model_results['feature_importance']:
                        importance_data.append({
                            'model': model_key,
                            'feature': feat['feature'],
                            'importance': feat['importance']
                        })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                
                # Top features across all models
                avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
                avg_importance = avg_importance.sort_values('importance', ascending=False).head(20)
                
                fig_imp = px.bar(
                    avg_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 20 Most Important Features (Average Across Models)",
                    labels={'importance': 'Average Importance', 'feature': 'Feature'}
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Feature importance by model
                st.subheader("Feature Importance by Model")
                
                selected_model = st.selectbox(
                    "Select Model",
                    options=importance_df['model'].unique()
                )
                
                if selected_model:
                    model_importance = importance_df[importance_df['model'] == selected_model]
                    model_importance = model_importance.sort_values('importance', ascending=False).head(15)
                    
                    fig_model_imp = px.bar(
                        model_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f"Top 15 Features for {selected_model}",
                        labels={'importance': 'Importance', 'feature': 'Feature'}
                    )
                    st.plotly_chart(fig_model_imp, use_container_width=True)
            else:
                st.info("No feature importance data available. Train models first.")
        else:
            st.info("No trained models found. Train models to see feature importance.")
        
        # Detailed features table
        st.header("Detailed Features Summary")
        st.dataframe(features_summary, use_container_width=True)
        
    else:
        st.warning("No features data found. Please extract features first.")
        
        st.markdown("### Extract Features")
        st.markdown("Use the CLI to extract features from SEC filings:")
        st.code("""
# Extract features from filings
python cli.py features extract --limit 50

# Check features summary
python cli.py features summary
        """)

except Exception as e:
    st.error(f"Error loading features data: {e}")

# Feature extraction controls
st.markdown("---")
st.header("Feature Extraction Controls")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Extract New Features")
    
    limit = st.number_input("Number of filings to process", min_value=1, max_value=1000, value=10)
    
    if st.button("Extract Features"):
        with st.spinner("Extracting features..."):
            try:
                extractor = FeatureExtractor(data_dir=data_dir)
                extractor.process_filings(limit=limit)
                extractor.create_tfidf_features()
                st.success(f"Successfully extracted features from {limit} filings!")
                st.rerun()
            except Exception as e:
                st.error(f"Error extracting features: {e}")

with col2:
    st.subheader("Feature Extraction Tips")
    st.markdown("""
    - Start with a small number of filings for testing
    - Feature extraction uses OpenAI API and may take time
    - TF-IDF features are created after LLM feature extraction
    - Features are stored in the features database
    """)

