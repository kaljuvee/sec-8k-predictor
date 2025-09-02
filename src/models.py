"""
Machine Learning Models for SEC 8-K Prediction

This module implements Random Forest classifier and regressor models
for predicting stock returns based on SEC 8-K filings.
"""

import os
import sqlite3
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEC8KPredictor:
    """Machine learning models for SEC 8-K filing prediction"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the predictor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Database paths
        self.features_db = self.data_dir / "features.db"
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # 8-K Categories
        self.categories = ["1.01", "1.02", "1.05", "2.01", "2.02", "3.01", "5.01", "5.02", "5.03", "8.01"]
        
    def load_training_data(self, category: str, target_variable: str = 'relative_return_5d') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load training data for a specific category
        
        Args:
            category: 8-K category
            target_variable: Target variable name
            
        Returns:
            Tuple of (feature_matrix, target_vector, feature_names)
        """
        conn = sqlite3.connect(self.features_db)
        
        query = """
            SELECT sentiment_score, urgency_score, financial_impact_score, market_relevance_score,
                   tfidf_features, {}
            FROM features 
            WHERE category = ? AND tfidf_features IS NOT NULL AND {} IS NOT NULL
        """.format(target_variable, target_variable)
        
        data = pd.read_sql_query(query, conn, params=(category,))
        conn.close()
        
        if data.empty:
            logger.warning(f"No data found for category {category}")
            return np.array([]), np.array([]), []
        
        # Prepare feature matrix
        features = []
        targets = []
        
        for _, row in data.iterrows():
            # LLM features
            llm_features = [
                row['sentiment_score'],
                row['urgency_score'], 
                row['financial_impact_score'],
                row['market_relevance_score']
            ]
            
            # TF-IDF features
            try:
                tfidf_features = json.loads(row['tfidf_features'])
            except:
                logger.warning(f"Invalid TF-IDF features for row, skipping")
                continue
            
            # Combine features
            combined_features = llm_features + tfidf_features
            features.append(combined_features)
            targets.append(row[target_variable])
        
        if not features:
            return np.array([]), np.array([]), []
        
        feature_matrix = np.array(features)
        target_vector = np.array(targets)
        
        # Create feature names
        feature_names = ['sentiment_score', 'urgency_score', 'financial_impact_score', 'market_relevance_score']
        feature_names.extend([f'tfidf_{i}' for i in range(len(tfidf_features))])
        
        logger.info(f"Loaded data for {category}: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        
        return feature_matrix, target_vector, feature_names
    
    def train_classifier(self, category: str, target_variable: str = 'relative_return_5d') -> Dict[str, Any]:
        """
        Train Random Forest classifier for a category
        
        Args:
            category: 8-K category
            target_variable: Target variable name
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training classifier for category {category}")
        
        # Load data
        X, y, feature_names = self.load_training_data(category, target_variable)
        
        if len(X) == 0:
            logger.warning(f"No data available for category {category}")
            return {}
        
        # Convert regression target to classification labels
        # Positive returns = 1, negative returns = 0
        y_class = (y > 0).astype(int)
        
        # Check if we have both classes
        if len(np.unique(y_class)) < 2:
            logger.warning(f"Only one class found for category {category}")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_classifier.predict(X_test_scaled)
        y_pred_proba = rf_classifier.predict_proba(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model and scaler
        model_key = f"{category}_classifier_{target_variable}"
        self.models[model_key] = rf_classifier
        self.scalers[model_key] = scaler
        
        # Save to disk
        model_path = self.models_dir / f"{model_key}.joblib"
        scaler_path = self.models_dir / f"{model_key}_scaler.joblib"
        
        joblib.dump(rf_classifier, model_path)
        joblib.dump(scaler, scaler_path)
        
        results = {
            'category': category,
            'model_type': 'classifier',
            'target_variable': target_variable,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance.head(10).to_dict('records'),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
        
        logger.info(f"Classifier trained for {category}: Accuracy = {accuracy:.3f}")
        
        return results
    
    def train_regressor(self, category: str, target_variable: str = 'relative_return_5d') -> Dict[str, Any]:
        """
        Train Random Forest regressor for a category
        
        Args:
            category: 8-K category
            target_variable: Target variable name
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training regressor for category {category}")
        
        # Load data
        X, y, feature_names = self.load_training_data(category, target_variable)
        
        if len(X) == 0:
            logger.warning(f"No data available for category {category}")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_regressor.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_regressor.predict(X_test_scaled)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_regressor, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Correlation between actual and predicted
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        
        # Store model and scaler
        model_key = f"{category}_regressor_{target_variable}"
        self.models[model_key] = rf_regressor
        self.scalers[model_key] = scaler
        
        # Save to disk
        model_path = self.models_dir / f"{model_key}.joblib"
        scaler_path = self.models_dir / f"{model_key}_scaler.joblib"
        
        joblib.dump(rf_regressor, model_path)
        joblib.dump(scaler, scaler_path)
        
        results = {
            'category': category,
            'model_type': 'regressor',
            'target_variable': target_variable,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'correlation': correlation,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.head(10).to_dict('records'),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
        
        logger.info(f"Regressor trained for {category}: R² = {r2:.3f}, Correlation = {correlation:.3f}")
        
        return results
    
    def train_all_models(self, target_variables: List[str] = None) -> Dict[str, Any]:
        """
        Train models for all categories and target variables
        
        Args:
            target_variables: List of target variables to train on
            
        Returns:
            Dictionary with all training results
        """
        if target_variables is None:
            target_variables = ['relative_return_5d', 'relative_return_9d']
        
        logger.info(f"Training models for all categories with targets: {target_variables}")
        
        all_results = {
            'classifiers': {},
            'regressors': {},
            'summary': {}
        }
        
        # Get available categories from data
        conn = sqlite3.connect(self.features_db)
        categories_query = "SELECT DISTINCT category FROM features WHERE category IS NOT NULL"
        available_categories = pd.read_sql_query(categories_query, conn)['category'].tolist()
        conn.close()
        
        logger.info(f"Available categories: {available_categories}")
        
        for category in available_categories:
            for target_var in target_variables:
                try:
                    # Train classifier
                    classifier_results = self.train_classifier(category, target_var)
                    if classifier_results:
                        key = f"{category}_{target_var}"
                        all_results['classifiers'][key] = classifier_results
                    
                    # Train regressor
                    regressor_results = self.train_regressor(category, target_var)
                    if regressor_results:
                        key = f"{category}_{target_var}"
                        all_results['regressors'][key] = regressor_results
                        
                except Exception as e:
                    logger.error(f"Error training models for {category} - {target_var}: {e}")
                    continue
        
        # Generate summary
        all_results['summary'] = {
            'total_classifiers': len(all_results['classifiers']),
            'total_regressors': len(all_results['regressors']),
            'categories_processed': len(available_categories),
            'target_variables': target_variables,
            'training_date': datetime.now().isoformat()
        }
        
        # Save results
        results_path = self.models_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Training completed. Results saved to {results_path}")
        
        return all_results
    
    def load_model(self, category: str, model_type: str, target_variable: str = 'relative_return_5d'):
        """
        Load a trained model from disk
        
        Args:
            category: 8-K category
            model_type: 'classifier' or 'regressor'
            target_variable: Target variable name
            
        Returns:
            Tuple of (model, scaler)
        """
        model_key = f"{category}_{model_type}_{target_variable}"
        model_path = self.models_dir / f"{model_key}.joblib"
        scaler_path = self.models_dir / f"{model_key}_scaler.joblib"
        
        if not model_path.exists() or not scaler_path.exists():
            logger.error(f"Model files not found for {model_key}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    def predict(self, features: np.ndarray, category: str, model_type: str, 
                target_variable: str = 'relative_return_5d') -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            features: Feature matrix
            category: 8-K category
            model_type: 'classifier' or 'regressor'
            target_variable: Target variable name
            
        Returns:
            Predictions array
        """
        model, scaler = self.load_model(category, model_type, target_variable)
        
        if model is None or scaler is None:
            logger.error(f"Could not load model for {category} {model_type}")
            return np.array([])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        
        return predictions
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        results_path = self.models_dir / "training_results.json"
        
        if not results_path.exists():
            logger.warning("No training results found")
            return pd.DataFrame()
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Create summary DataFrame
        summary_data = []
        
        for key, result in results['classifiers'].items():
            summary_data.append({
                'category': result['category'],
                'model_type': 'classifier',
                'target_variable': result['target_variable'],
                'n_samples': result['n_samples'],
                'accuracy': result.get('accuracy', 0),
                'cv_score': result.get('cv_mean', 0)
            })
        
        for key, result in results['regressors'].items():
            summary_data.append({
                'category': result['category'],
                'model_type': 'regressor',
                'target_variable': result['target_variable'],
                'n_samples': result['n_samples'],
                'r2_score': result.get('r2_score', 0),
                'correlation': result.get('correlation', 0),
                'cv_score': result.get('cv_mean', 0)
            })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test the models
    predictor = SEC8KPredictor(data_dir="../data")
    
    print("Testing SEC 8-K predictor models...")
    
    # Train models for available categories
    results = predictor.train_all_models()
    
    print(f"\nTraining Summary:")
    print(f"Classifiers trained: {results['summary']['total_classifiers']}")
    print(f"Regressors trained: {results['summary']['total_regressors']}")
    
    # Show model summary
    summary = predictor.get_model_summary()
    if not summary.empty:
        print("\nModel Summary:")
        print(summary)

