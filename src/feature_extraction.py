"""
Feature Extraction and Labeling System

This module handles extracting features from SEC 8-K filings using OpenAI LLM,
text preprocessing with spaCy, TF-IDF vectorization, and calculating target variables.
"""

import os
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import openai
import logging
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts features from SEC 8-K filings and creates labeled datasets"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the feature extractor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Increase max length for large documents
            self.nlp.max_length = 2000000
        except OSError:
            logger.error("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI()
        
        # Database paths
        self.filings_db = self.data_dir / "sec_filings.db"
        self.stock_db = self.data_dir / "stock_data.db"
        self.features_db = self.data_dir / "features.db"
        
        # Initialize features database
        self._init_features_database()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
    def _init_features_database(self):
        """Initialize database for storing extracted features"""
        conn = sqlite3.connect(self.features_db)
        cursor = conn.cursor()
        
        # Features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                filing_date DATE NOT NULL,
                accession_number TEXT NOT NULL,
                category TEXT,
                processed_content TEXT,
                sentiment_score REAL,
                urgency_score REAL,
                financial_impact_score REAL,
                market_relevance_score REAL,
                llm_features TEXT,  -- JSON string of LLM extracted features
                tfidf_features TEXT,  -- JSON string of TF-IDF features
                return_5d REAL,
                return_9d REAL,
                relative_return_5d REAL,
                relative_return_9d REAL,
                volatility_change_5d REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(accession_number)
            )
        """)
        
        # Training datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                feature_matrix TEXT,  -- JSON string of feature matrix
                target_vector TEXT,   -- JSON string of target vector
                feature_names TEXT,   -- JSON string of feature names
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, dataset_name)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using spaCy
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        try:
            # Truncate very long texts to avoid memory issues
            if len(text) > 500000:  # 500K characters limit
                text = text[:500000]
            
            # Clean text
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip().lower()
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract lemmatized tokens, excluding stop words and punctuation
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ]
            
            return ' '.join(tokens)
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            # Fallback to simple processing
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()
    
    def extract_llm_features(self, content: str, category: str) -> Dict[str, Any]:
        """
        Extract features using OpenAI LLM
        
        Args:
            content: Filing content
            category: 8-K category
            
        Returns:
            Dictionary of extracted features
        """
        try:
            prompt = f"""
            Analyze the following SEC 8-K filing content and extract key features for stock price prediction.
            
            Filing Category: {category}
            Content: {content[:4000]}  # Limit content length
            
            Please provide scores (0-10) for the following aspects:
            1. Sentiment Score: How positive/negative is the news (0=very negative, 5=neutral, 10=very positive)
            2. Urgency Score: How urgent/immediate is the impact (0=low urgency, 10=high urgency)
            3. Financial Impact Score: Expected magnitude of financial impact (0=minimal, 10=major)
            4. Market Relevance Score: How relevant to broader market (0=company-specific, 10=market-wide impact)
            
            Also extract:
            5. Key Topics: List of main topics/themes (max 5)
            6. Financial Metrics Mentioned: Any specific numbers, percentages, or financial metrics
            7. Forward-Looking Statements: Any guidance or future projections mentioned
            
            Return the response as a JSON object with these keys:
            sentiment_score, urgency_score, financial_impact_score, market_relevance_score, 
            key_topics, financial_metrics, forward_looking_statements
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Ensure all required fields are present
            default_result = {
                "sentiment_score": 5.0,
                "urgency_score": 5.0,
                "financial_impact_score": 5.0,
                "market_relevance_score": 5.0,
                "key_topics": [],
                "financial_metrics": [],
                "forward_looking_statements": []
            }
            
            # Update with actual results
            default_result.update(result)
            
            return default_result
            
        except Exception as e:
            logger.error(f"Error extracting LLM features: {e}")
            # Return default values
            return {
                "sentiment_score": 5.0,
                "urgency_score": 5.0,
                "financial_impact_score": 5.0,
                "market_relevance_score": 5.0,
                "key_topics": [],
                "financial_metrics": [],
                "forward_looking_statements": []
            }
    
    def calculate_target_variables(self, ticker: str, filing_date: str) -> Dict[str, float]:
        """
        Calculate target variables (returns and volatility changes)
        
        Args:
            ticker: Stock ticker
            filing_date: Filing date
            
        Returns:
            Dictionary of target variables
        """
        try:
            conn = sqlite3.connect(self.stock_db)
            
            # Convert filing date to datetime
            filing_dt = pd.to_datetime(filing_date)
            
            # Get a window of data around the filing date
            start_date = (filing_dt - timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (filing_dt + timedelta(days=15)).strftime('%Y-%m-%d')
            
            query = """
                SELECT date, return_5d, return_9d, relative_return_5d, relative_return_9d, 
                       volatility_5d, spy_volatility_5d
                FROM stock_returns 
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            data = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
            conn.close()
            
            if data.empty:
                logger.warning(f"No stock data found for {ticker} around {filing_date}")
                return {}
            
            # Find the closest trading day to filing date
            data['date'] = pd.to_datetime(data['date'])
            data['days_from_filing'] = (data['date'] - filing_dt).dt.days
            
            # Get the row closest to filing date (preferably after)
            future_data = data[data['days_from_filing'] >= 0]
            if not future_data.empty:
                target_row = future_data.iloc[0]
            else:
                target_row = data.loc[data['days_from_filing'].abs().idxmin()]
            
            # Calculate volatility change
            pre_filing_data = data[data['days_from_filing'] < 0]
            if not pre_filing_data.empty:
                pre_volatility = pre_filing_data['volatility_5d'].mean()
                pre_spy_volatility = pre_filing_data['spy_volatility_5d'].mean()
                volatility_change = target_row['volatility_5d'] - pre_volatility
                relative_volatility_change = volatility_change - (target_row['spy_volatility_5d'] - pre_spy_volatility)
            else:
                relative_volatility_change = 0.0
            
            return {
                'return_5d': target_row['return_5d'],
                'return_9d': target_row['return_9d'],
                'relative_return_5d': target_row['relative_return_5d'],
                'relative_return_9d': target_row['relative_return_9d'],
                'volatility_change_5d': relative_volatility_change
            }
            
        except Exception as e:
            logger.error(f"Error calculating target variables for {ticker} on {filing_date}: {e}")
            return {}
    
    def process_filings(self, limit: int = None) -> None:
        """
        Process SEC filings to extract features and calculate targets
        
        Args:
            limit: Maximum number of filings to process (None for all)
        """
        logger.info("Starting feature extraction from SEC filings...")
        
        # Get processed accession numbers from features database
        features_conn = sqlite3.connect(self.features_db)
        processed_query = "SELECT accession_number FROM features"
        processed_df = pd.read_sql_query(processed_query, features_conn)
        processed_accessions = set(processed_df['accession_number'].tolist()) if not processed_df.empty else set()
        features_conn.close()
        
        # Get unprocessed filings from filings database
        filings_conn = sqlite3.connect(self.filings_db)
        
        query = """
            SELECT ticker, filing_date, accession_number, category, content
            FROM filings
            WHERE filing_date IS NOT NULL 
            AND content IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit * 2}"  # Get more to account for already processed
        
        all_filings = pd.read_sql_query(query, filings_conn)
        filings_conn.close()
        
        # Filter out already processed filings
        filings = all_filings[~all_filings['accession_number'].isin(processed_accessions)]
        
        if limit and len(filings) > limit:
            filings = filings.head(limit)
        
        logger.info(f"Processing {len(filings)} filings...")
        
        for idx, filing in filings.iterrows():
            try:
                self._process_single_filing(filing)
                
                # Rate limiting for OpenAI API
                time.sleep(0.5)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(filings)} filings")
                    
            except Exception as e:
                logger.error(f"Error processing filing {filing['accession_number']}: {e}")
                continue
        
        logger.info("Feature extraction completed")
    
    def _process_single_filing(self, filing: pd.Series) -> None:
        """Process a single filing"""
        
        # Preprocess content
        processed_content = self.preprocess_text(filing['content'])
        
        # Extract LLM features
        llm_features = self.extract_llm_features(filing['content'], filing['category'])
        
        # Calculate target variables
        targets = self.calculate_target_variables(filing['ticker'], filing['filing_date'])
        
        # Store in features database
        conn = sqlite3.connect(self.features_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO features (
                ticker, filing_date, accession_number, category, processed_content,
                sentiment_score, urgency_score, financial_impact_score, market_relevance_score,
                llm_features, return_5d, return_9d, relative_return_5d, relative_return_9d,
                volatility_change_5d
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filing['ticker'], filing['filing_date'], filing['accession_number'],
            filing['category'], processed_content,
            llm_features.get('sentiment_score', 5.0),
            llm_features.get('urgency_score', 5.0),
            llm_features.get('financial_impact_score', 5.0),
            llm_features.get('market_relevance_score', 5.0),
            json.dumps(llm_features),
            targets.get('return_5d'),
            targets.get('return_9d'),
            targets.get('relative_return_5d'),
            targets.get('relative_return_9d'),
            targets.get('volatility_change_5d')
        ))
        
        conn.commit()
        conn.close()
    
    def create_tfidf_features(self, category: str = None) -> None:
        """
        Create TF-IDF features for processed filings
        
        Args:
            category: Specific category to process (None for all)
        """
        logger.info(f"Creating TF-IDF features for category: {category or 'all'}")
        
        conn = sqlite3.connect(self.features_db)
        
        # Get processed filings
        if category:
            query = "SELECT id, processed_content FROM features WHERE category = ? AND processed_content IS NOT NULL"
            data = pd.read_sql_query(query, conn, params=(category,))
        else:
            query = "SELECT id, processed_content FROM features WHERE processed_content IS NOT NULL"
            data = pd.read_sql_query(query, conn)
        
        if data.empty:
            logger.warning("No processed content found")
            conn.close()
            return
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(data['processed_content'])
        
        # Update database with TF-IDF features
        cursor = conn.cursor()
        
        for idx, row in data.iterrows():
            tfidf_features = tfidf_matrix[idx].toarray()[0].tolist()
            
            cursor.execute("""
                UPDATE features 
                SET tfidf_features = ?
                WHERE id = ?
            """, (json.dumps(tfidf_features), row['id']))
        
        conn.commit()
        conn.close()
        
        logger.info(f"TF-IDF features created for {len(data)} filings")
    
    def create_training_dataset(self, category: str, target_variable: str = 'relative_return_5d') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create training dataset for a specific category
        
        Args:
            category: 8-K category
            target_variable: Target variable name
            
        Returns:
            Tuple of (feature_matrix, target_vector, feature_names)
        """
        logger.info(f"Creating training dataset for category {category}, target: {target_variable}")
        
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
            tfidf_features = json.loads(row['tfidf_features'])
            
            # Combine features
            combined_features = llm_features + tfidf_features
            features.append(combined_features)
            targets.append(row[target_variable])
        
        feature_matrix = np.array(features)
        target_vector = np.array(targets)
        
        # Create feature names
        feature_names = ['sentiment_score', 'urgency_score', 'financial_impact_score', 'market_relevance_score']
        feature_names.extend([f'tfidf_{i}' for i in range(len(tfidf_features))])
        
        # Store dataset
        self._store_training_dataset(category, target_variable, feature_matrix, target_vector, feature_names)
        
        logger.info(f"Created dataset for {category}: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        
        return feature_matrix, target_vector, feature_names
    
    def _store_training_dataset(self, category: str, dataset_name: str, 
                               feature_matrix: np.ndarray, target_vector: np.ndarray, 
                               feature_names: List[str]) -> None:
        """Store training dataset in database"""
        
        conn = sqlite3.connect(self.features_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO training_datasets 
            (category, dataset_name, feature_matrix, target_vector, feature_names)
            VALUES (?, ?, ?, ?, ?)
        """, (
            category,
            dataset_name,
            json.dumps(feature_matrix.tolist()),
            json.dumps(target_vector.tolist()),
            json.dumps(feature_names)
        ))
        
        conn.commit()
        conn.close()
    
    def get_features_summary(self) -> pd.DataFrame:
        """Get summary of extracted features"""
        conn = sqlite3.connect(self.features_db)
        
        query = """
            SELECT 
                category,
                COUNT(*) as total_filings,
                COUNT(CASE WHEN return_5d IS NOT NULL THEN 1 END) as with_targets,
                AVG(sentiment_score) as avg_sentiment,
                AVG(urgency_score) as avg_urgency,
                AVG(financial_impact_score) as avg_financial_impact,
                AVG(market_relevance_score) as avg_market_relevance
            FROM features 
            GROUP BY category
            ORDER BY total_filings DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor(data_dir="../data")
    
    print("Testing feature extraction...")
    
    # Process a small sample of filings
    extractor.process_filings(limit=5)
    
    # Create TF-IDF features
    extractor.create_tfidf_features()
    
    # Show summary
    summary = extractor.get_features_summary()
    print("\nFeatures Summary:")
    print(summary)

