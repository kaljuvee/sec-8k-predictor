"""
Predictions Page

Make predictions on SEC filings and analyze results.
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
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import SEC8KPredictor
from src.feature_extraction import FeatureExtractor

st.title("🎯 Predictions")
st.markdown("Make predictions on SEC filings and analyze results.")

# Data directory
data_dir = "data"

# Tabs for different prediction views
tab1, tab2, tab3 = st.tabs(["Make Predictions", "Prediction History", "Backtesting"])

with tab1:
    st.header("Make New Predictions")
    
    try:
        # Get available models
        predictor = SEC8KPredictor(data_dir=data_dir)
        model_summary = predictor.get_model_summary()
        
        if not model_summary.empty:
            # Model selection
            col1, col2 = st.columns(2)
            
            with col1:
                available_categories = model_summary['category'].unique()
                selected_category = st.selectbox(
                    "Select 8-K Category",
                    options=available_categories
                )
                
                model_types = model_summary[model_summary['category'] == selected_category]['model_type'].unique()
                selected_model_type = st.selectbox(
                    "Select Model Type",
                    options=model_types
                )
            
            with col2:
                target_vars = model_summary[
                    (model_summary['category'] == selected_category) & 
                    (model_summary['model_type'] == selected_model_type)
                ]['target_variable'].unique()
                
                selected_target = st.selectbox(
                    "Select Target Variable",
                    options=target_vars
                )
            
            # Get available filings for prediction
            conn = sqlite3.connect(Path(data_dir) / "features.db")
            
            filings_query = """
                SELECT ticker, filing_date, accession_number, category
                FROM features 
                WHERE category = ? AND tfidf_features IS NOT NULL
                ORDER BY filing_date DESC
                LIMIT 50
            """
            
            available_filings = pd.read_sql_query(filings_query, conn, params=(selected_category,))
            conn.close()
            
            if not available_filings.empty:
                st.subheader("Select Filing for Prediction")
                
                # Filing selection
                filing_options = []
                for _, row in available_filings.iterrows():
                    filing_options.append(f"{row['ticker']} - {row['filing_date']} ({row['accession_number'][:10]}...)")
                
                selected_filing_idx = st.selectbox(
                    "Select Filing",
                    options=range(len(filing_options)),
                    format_func=lambda x: filing_options[x]
                )
                
                selected_filing = available_filings.iloc[selected_filing_idx]
                
                # Show filing details
                st.markdown("**Selected Filing Details:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Ticker", selected_filing['ticker'])
                
                with col2:
                    st.metric("Filing Date", selected_filing['filing_date'])
                
                with col3:
                    st.metric("Category", selected_filing['category'])
                
                # Make prediction button
                if st.button("Make Prediction", type="primary"):
                    with st.spinner("Making prediction..."):
                        try:
                            # Load features for the selected filing
                            conn = sqlite3.connect(Path(data_dir) / "features.db")
                            
                            features_query = """
                                SELECT sentiment_score, urgency_score, financial_impact_score, 
                                       market_relevance_score, tfidf_features, return_5d, return_9d,
                                       relative_return_5d, relative_return_9d
                                FROM features 
                                WHERE accession_number = ?
                            """
                            
                            filing_features = pd.read_sql_query(
                                features_query, 
                                conn, 
                                params=(selected_filing['accession_number'],)
                            )
                            conn.close()
                            
                            if not filing_features.empty:
                                # Prepare feature vector
                                row = filing_features.iloc[0]
                                
                                llm_features = [
                                    row['sentiment_score'],
                                    row['urgency_score'],
                                    row['financial_impact_score'],
                                    row['market_relevance_score']
                                ]
                                
                                tfidf_features = json.loads(row['tfidf_features'])
                                combined_features = np.array([llm_features + tfidf_features])
                                
                                # Make prediction
                                prediction = predictor.predict(
                                    combined_features,
                                    selected_category,
                                    selected_model_type,
                                    selected_target
                                )
                                
                                if len(prediction) > 0:
                                    pred_value = prediction[0]
                                    
                                    # Show prediction results
                                    st.success("Prediction completed!")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Predicted Value", f"{pred_value:.4f}")
                                    
                                    with col2:
                                        if selected_model_type == 'classifier':
                                            direction = "Positive" if pred_value > 0.5 else "Negative"
                                            confidence = max(pred_value, 1 - pred_value)
                                            st.metric("Direction", direction)
                                            st.metric("Confidence", f"{confidence:.2%}")
                                        else:
                                            direction = "Positive" if pred_value > 0 else "Negative"
                                            st.metric("Direction", direction)
                                    
                                    with col3:
                                        # Show actual value if available
                                        actual_value = row[selected_target]
                                        if pd.notna(actual_value):
                                            st.metric("Actual Value", f"{actual_value:.4f}")
                                            error = abs(pred_value - actual_value)
                                            st.metric("Prediction Error", f"{error:.4f}")
                                    
                                    # Visualization
                                    if pd.notna(actual_value):
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Bar(
                                            x=['Predicted', 'Actual'],
                                            y=[pred_value, actual_value],
                                            marker_color=['blue', 'red']
                                        ))
                                        
                                        fig.update_layout(
                                            title="Predicted vs Actual Value",
                                            yaxis_title=selected_target.replace('_', ' ').title()
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Feature contribution (simplified)
                                    st.subheader("Feature Values")
                                    
                                    feature_data = pd.DataFrame({
                                        'Feature': ['Sentiment', 'Urgency', 'Financial Impact', 'Market Relevance'],
                                        'Value': llm_features
                                    })
                                    
                                    fig_features = px.bar(
                                        feature_data,
                                        x='Feature',
                                        y='Value',
                                        title="LLM Feature Values for This Filing"
                                    )
                                    st.plotly_chart(fig_features, use_container_width=True)
                                    
                                else:
                                    st.error("Failed to make prediction. Model may not be available.")
                            else:
                                st.error("No features found for selected filing.")
                                
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
            else:
                st.warning(f"No filings available for category {selected_category}")
        else:
            st.warning("No trained models found. Please train models first.")
            
    except Exception as e:
        st.error(f"Error loading prediction interface: {e}")

with tab2:
    st.header("Prediction History")
    
    # This would store prediction history in a database table
    # For now, show a placeholder
    st.info("Prediction history feature would be implemented here.")
    
    st.markdown("""
    **Planned Features:**
    - Store all predictions made through the interface
    - Track prediction accuracy over time
    - Compare different model performances
    - Export prediction results
    """)

with tab3:
    st.header("Backtesting Analysis")
    
    try:
        # Load features with actual returns for backtesting
        conn = sqlite3.connect(Path(data_dir) / "features.db")
        
        backtest_query = """
            SELECT category, ticker, filing_date, 
                   sentiment_score, urgency_score, financial_impact_score, market_relevance_score,
                   return_5d, return_9d, relative_return_5d, relative_return_9d
            FROM features 
            WHERE relative_return_5d IS NOT NULL
            ORDER BY filing_date DESC
        """
        
        backtest_data = pd.read_sql_query(backtest_query, conn)
        conn.close()
        
        if not backtest_data.empty:
            st.subheader("Backtesting Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                backtest_categories = backtest_data['category'].unique()
                selected_backtest_category = st.selectbox(
                    "Category for Backtesting",
                    options=backtest_categories
                )
                
                backtest_target = st.selectbox(
                    "Target Variable for Backtesting",
                    options=['relative_return_5d', 'relative_return_9d']
                )
            
            with col2:
                # Date range for backtesting
                min_date = pd.to_datetime(backtest_data['filing_date']).min()
                max_date = pd.to_datetime(backtest_data['filing_date']).max()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Filter data for backtesting
            category_data = backtest_data[
                (backtest_data['category'] == selected_backtest_category) &
                (pd.to_datetime(backtest_data['filing_date']) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(backtest_data['filing_date']) <= pd.to_datetime(end_date))
            ]
            
            if len(category_data) > 0:
                st.subheader("Backtesting Results")
                
                # Simple correlation analysis
                numeric_features = ['sentiment_score', 'urgency_score', 'financial_impact_score', 'market_relevance_score']
                
                correlations = []
                for feature in numeric_features:
                    corr = category_data[feature].corr(category_data[backtest_target])
                    correlations.append({
                        'Feature': feature.replace('_', ' ').title(),
                        'Correlation': corr
                    })
                
                corr_df = pd.DataFrame(correlations)
                
                # Correlation chart
                fig_corr = px.bar(
                    corr_df,
                    x='Feature',
                    y='Correlation',
                    title=f"Feature Correlations with {backtest_target.replace('_', ' ').title()}",
                    color='Correlation',
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Time series analysis
                category_data['filing_date'] = pd.to_datetime(category_data['filing_date'])
                category_data = category_data.sort_values('filing_date')
                
                fig_ts = px.scatter(
                    category_data,
                    x='filing_date',
                    y=backtest_target,
                    color='sentiment_score',
                    title=f"{backtest_target.replace('_', ' ').title()} Over Time",
                    labels={'filing_date': 'Filing Date'}
                )
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Performance by sentiment ranges
                category_data['sentiment_range'] = pd.cut(
                    category_data['sentiment_score'], 
                    bins=[0, 3, 7, 10], 
                    labels=['Negative (0-3)', 'Neutral (3-7)', 'Positive (7-10)']
                )
                
                sentiment_performance = category_data.groupby('sentiment_range')[backtest_target].agg(['mean', 'std', 'count']).reset_index()
                
                fig_sent_perf = px.bar(
                    sentiment_performance,
                    x='sentiment_range',
                    y='mean',
                    error_y='std',
                    title=f"Average {backtest_target.replace('_', ' ').title()} by Sentiment Range",
                    labels={'mean': f'Average {backtest_target.replace("_", " ").title()}'}
                )
                st.plotly_chart(fig_sent_perf, use_container_width=True)
                
                # Summary statistics
                st.subheader("Backtesting Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Filings", len(category_data))
                    st.metric("Date Range", f"{len(category_data)} filings")
                
                with col2:
                    avg_return = category_data[backtest_target].mean()
                    st.metric("Average Return", f"{avg_return:.4f}")
                    
                    volatility = category_data[backtest_target].std()
                    st.metric("Volatility", f"{volatility:.4f}")
                
                with col3:
                    positive_returns = (category_data[backtest_target] > 0).sum()
                    positive_rate = positive_returns / len(category_data)
                    st.metric("Positive Returns", f"{positive_rate:.1%}")
                
                # Detailed data
                st.subheader("Detailed Backtesting Data")
                st.dataframe(category_data[['ticker', 'filing_date'] + numeric_features + [backtest_target]], use_container_width=True)
                
            else:
                st.warning("No data available for the selected category and date range.")
        else:
            st.warning("No backtesting data available. Ensure features have been extracted with target variables.")
            
    except Exception as e:
        st.error(f"Error loading backtesting data: {e}")

# Prediction tips
st.markdown("---")
st.header("Prediction Tips")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Making Good Predictions")
    st.markdown("""
    - Ensure models are trained on sufficient data
    - Check model performance metrics before using
    - Consider the filing category and its typical impact
    - Review feature values for reasonableness
    """)

with col2:
    st.subheader("Interpreting Results")
    st.markdown("""
    - Classifier predictions: probability of positive return
    - Regressor predictions: expected return magnitude
    - Compare with actual values when available
    - Consider prediction confidence/uncertainty
    """)

