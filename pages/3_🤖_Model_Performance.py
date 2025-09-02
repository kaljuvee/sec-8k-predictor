"""
Model Performance Page

View machine learning model results and performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import SEC8KPredictor

st.title("🤖 Model Performance")
st.markdown("View machine learning model results and performance metrics.")

# Data directory
data_dir = "data"

try:
    predictor = SEC8KPredictor(data_dir=data_dir)
    model_summary = predictor.get_model_summary()
    
    # Check for training results
    results_file = Path(data_dir) / "models" / "training_results.json"
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            training_results = json.load(f)
        
        # Summary metrics
        st.header("Model Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_classifiers = len(training_results.get('classifiers', {}))
            st.metric("Classifiers", total_classifiers)
        
        with col2:
            total_regressors = len(training_results.get('regressors', {}))
            st.metric("Regressors", total_regressors)
        
        with col3:
            categories_processed = training_results.get('summary', {}).get('categories_processed', 0)
            st.metric("Categories", categories_processed)
        
        with col4:
            training_date = training_results.get('summary', {}).get('training_date', 'Unknown')
            st.metric("Last Training", training_date[:10] if training_date != 'Unknown' else 'Unknown')
        
        # Model performance comparison
        if not model_summary.empty:
            st.header("Model Performance Comparison")
            
            # Tabs for different model types
            tab1, tab2 = st.tabs(["Classifiers", "Regressors"])
            
            with tab1:
                classifier_data = model_summary[model_summary['model_type'] == 'classifier']
                
                if not classifier_data.empty:
                    # Accuracy comparison
                    fig_acc = px.bar(
                        classifier_data,
                        x='category',
                        y='accuracy',
                        color='target_variable',
                        title="Classifier Accuracy by Category",
                        labels={'accuracy': 'Accuracy', 'category': '8-K Category'}
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # Cross-validation scores
                    fig_cv = px.bar(
                        classifier_data,
                        x='category',
                        y='cv_score',
                        color='target_variable',
                        title="Cross-Validation Scores by Category",
                        labels={'cv_score': 'CV Score', 'category': '8-K Category'}
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)
                    
                    # Sample size vs performance
                    fig_sample = px.scatter(
                        classifier_data,
                        x='n_samples',
                        y='accuracy',
                        color='category',
                        size='accuracy',
                        title="Sample Size vs Accuracy",
                        labels={'n_samples': 'Number of Samples', 'accuracy': 'Accuracy'}
                    )
                    st.plotly_chart(fig_sample, use_container_width=True)
                    
                    # Detailed classifier results
                    st.subheader("Detailed Classifier Results")
                    
                    for _, row in classifier_data.iterrows():
                        with st.expander(f"Category {row['category']} - {row['target_variable']}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Accuracy", f"{row['accuracy']:.3f}")
                                st.metric("Samples", f"{row['n_samples']:,}")
                            
                            with col2:
                                st.metric("CV Score", f"{row['cv_score']:.3f}")
                            
                            # Get detailed results from training_results
                            model_key = f"{row['category']}_{row['target_variable']}"
                            if model_key in training_results.get('classifiers', {}):
                                detailed = training_results['classifiers'][model_key]
                                
                                # Confusion matrix
                                if 'confusion_matrix' in detailed:
                                    conf_matrix = np.array(detailed['confusion_matrix'])
                                    fig_conf = px.imshow(
                                        conf_matrix,
                                        title="Confusion Matrix",
                                        labels=dict(x="Predicted", y="Actual"),
                                        x=['Negative', 'Positive'],
                                        y=['Negative', 'Positive']
                                    )
                                    st.plotly_chart(fig_conf, use_container_width=True)
                else:
                    st.info("No classifier models found.")
            
            with tab2:
                regressor_data = model_summary[model_summary['model_type'] == 'regressor']
                
                if not regressor_data.empty:
                    # R² comparison
                    fig_r2 = px.bar(
                        regressor_data,
                        x='category',
                        y='r2_score',
                        color='target_variable',
                        title="R² Score by Category",
                        labels={'r2_score': 'R² Score', 'category': '8-K Category'}
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
                    
                    # Correlation comparison
                    fig_corr = px.bar(
                        regressor_data,
                        x='category',
                        y='correlation',
                        color='target_variable',
                        title="Correlation (Actual vs Predicted) by Category",
                        labels={'correlation': 'Correlation', 'category': '8-K Category'}
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Cross-validation scores
                    fig_cv_reg = px.bar(
                        regressor_data,
                        x='category',
                        y='cv_score',
                        color='target_variable',
                        title="Cross-Validation R² Scores by Category",
                        labels={'cv_score': 'CV R² Score', 'category': '8-K Category'}
                    )
                    st.plotly_chart(fig_cv_reg, use_container_width=True)
                    
                    # Performance vs sample size
                    fig_perf = px.scatter(
                        regressor_data,
                        x='n_samples',
                        y='r2_score',
                        color='category',
                        size='correlation',
                        title="Sample Size vs R² Score",
                        labels={'n_samples': 'Number of Samples', 'r2_score': 'R² Score'}
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Detailed regressor results
                    st.subheader("Detailed Regressor Results")
                    
                    for _, row in regressor_data.iterrows():
                        with st.expander(f"Category {row['category']} - {row['target_variable']}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("R² Score", f"{row['r2_score']:.3f}")
                                st.metric("Correlation", f"{row['correlation']:.3f}")
                            
                            with col2:
                                st.metric("CV Score", f"{row['cv_score']:.3f}")
                                st.metric("Samples", f"{row['n_samples']:,}")
                            
                            # Get detailed results
                            model_key = f"{row['category']}_{row['target_variable']}"
                            if model_key in training_results.get('regressors', {}):
                                detailed = training_results['regressors'][model_key]
                                
                                with col3:
                                    if 'mse' in detailed:
                                        st.metric("MSE", f"{detailed['mse']:.6f}")
                                    if 'mae' in detailed:
                                        st.metric("MAE", f"{detailed['mae']:.6f}")
                else:
                    st.info("No regressor models found.")
        
        # Feature importance analysis
        st.header("Feature Importance Analysis")
        
        # Aggregate feature importance across models
        all_importance = []
        
        for model_type in ['classifiers', 'regressors']:
            for model_key, model_results in training_results.get(model_type, {}).items():
                if 'feature_importance' in model_results:
                    for feat in model_results['feature_importance']:
                        all_importance.append({
                            'model_type': model_type[:-1],  # Remove 's'
                            'model': model_key,
                            'feature': feat['feature'],
                            'importance': feat['importance']
                        })
        
        if all_importance:
            importance_df = pd.DataFrame(all_importance)
            
            # Top features overall
            top_features = importance_df.groupby('feature')['importance'].mean().reset_index()
            top_features = top_features.sort_values('importance', ascending=False).head(15)
            
            fig_top = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features (Average Across All Models)",
                labels={'importance': 'Average Importance', 'feature': 'Feature'}
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
            # Feature importance by model type
            col1, col2 = st.columns(2)
            
            with col1:
                classifier_importance = importance_df[importance_df['model_type'] == 'classifier']
                if not classifier_importance.empty:
                    top_class_features = classifier_importance.groupby('feature')['importance'].mean().reset_index()
                    top_class_features = top_class_features.sort_values('importance', ascending=False).head(10)
                    
                    fig_class = px.bar(
                        top_class_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Features for Classifiers",
                        labels={'importance': 'Average Importance'}
                    )
                    st.plotly_chart(fig_class, use_container_width=True)
            
            with col2:
                regressor_importance = importance_df[importance_df['model_type'] == 'regressor']
                if not regressor_importance.empty:
                    top_reg_features = regressor_importance.groupby('feature')['importance'].mean().reset_index()
                    top_reg_features = top_reg_features.sort_values('importance', ascending=False).head(10)
                    
                    fig_reg = px.bar(
                        top_reg_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Features for Regressors",
                        labels={'importance': 'Average Importance'}
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
        
        # Model summary table
        st.header("Model Summary Table")
        if not model_summary.empty:
            st.dataframe(model_summary, use_container_width=True)
        
    else:
        st.warning("No trained models found.")
        
        st.markdown("### Train Models")
        st.markdown("Use the CLI to train machine learning models:")
        st.code("""
# Train models for all available categories
python cli.py models train

# Train models for specific categories
python cli.py models train --categories "2.02,8.01"

# Check model summary
python cli.py models summary
        """)

except Exception as e:
    st.error(f"Error loading model performance data: {e}")

# Model training controls
st.markdown("---")
st.header("Model Training Controls")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Train New Models")
    
    # Get available categories
    try:
        import sqlite3
        conn = sqlite3.connect(Path(data_dir) / "features.db")
        available_categories = pd.read_sql_query(
            "SELECT DISTINCT category FROM features WHERE category IS NOT NULL", 
            conn
        )['category'].tolist()
        conn.close()
        
        selected_categories = st.multiselect(
            "Select categories to train",
            options=available_categories,
            default=available_categories[:3] if len(available_categories) >= 3 else available_categories
        )
        
        target_variables = st.multiselect(
            "Select target variables",
            options=['relative_return_5d', 'relative_return_9d'],
            default=['relative_return_5d']
        )
        
        if st.button("Train Models"):
            if selected_categories and target_variables:
                with st.spinner("Training models..."):
                    try:
                        predictor = SEC8KPredictor(data_dir=data_dir)
                        
                        # Train models for selected categories
                        results = {'classifiers': {}, 'regressors': {}}
                        
                        for category in selected_categories:
                            for target in target_variables:
                                try:
                                    classifier_results = predictor.train_classifier(category, target)
                                    if classifier_results:
                                        results['classifiers'][f"{category}_{target}"] = classifier_results
                                    
                                    regressor_results = predictor.train_regressor(category, target)
                                    if regressor_results:
                                        results['regressors'][f"{category}_{target}"] = regressor_results
                                        
                                except Exception as e:
                                    st.error(f"Error training {category} - {target}: {e}")
                        
                        st.success(f"Training completed! Trained {len(results['classifiers'])} classifiers and {len(results['regressors'])} regressors.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during training: {e}")
            else:
                st.warning("Please select categories and target variables.")
                
    except Exception as e:
        st.error(f"Error loading categories: {e}")

with col2:
    st.subheader("Training Tips")
    st.markdown("""
    - Ensure you have extracted features before training models
    - Models require sufficient data (>10 samples recommended)
    - Training may take several minutes depending on data size
    - Both classifiers and regressors are trained for each category
    - Models are automatically saved for later use
    """)

