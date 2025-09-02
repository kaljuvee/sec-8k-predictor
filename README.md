# SEC 8-K Predictor

A comprehensive machine learning system for predicting stock returns based on SEC 8-K filing content and market data.

## 🎯 Overview

The SEC 8-K Predictor is an end-to-end system that:
- Downloads and processes SEC 8-K filings from EDGAR
- Collects corresponding stock price data
- Extracts features using LLM analysis and TF-IDF
- Trains machine learning models for return prediction
- Provides both CLI and web interfaces for interaction

## 🚀 Features

### Data Collection
- **SEC 8-K Filings**: Automated download from EDGAR database
- **Stock Data**: Historical price and volume data via yfinance
- **S&P 500 Coverage**: Focus on major market constituents
- **Multiple Categories**: Support for all 8-K item categories

### Feature Engineering
- **LLM Analysis**: Sentiment, urgency, financial impact, and market relevance scoring
- **TF-IDF Features**: Text vectorization for content analysis
- **Return Calculations**: 5-day and 9-day absolute and relative returns
- **Volatility Metrics**: Risk-adjusted performance measures

### Machine Learning
- **Random Forest Models**: Both classification and regression
- **Category-Specific**: Separate models for each 8-K category
- **Performance Metrics**: Accuracy, R², correlation, and cross-validation
- **Feature Importance**: Analysis of predictive factors

### Interfaces
- **Command Line Interface**: Full system control via CLI
- **Streamlit Web App**: Interactive dashboards and visualizations
- **RESTful Design**: Modular architecture for easy extension

## 📋 Requirements

### System Requirements
- Python 3.11+
- 8GB+ RAM (for large-scale processing)
- 10GB+ disk space (for data storage)
- Internet connection (for data downloads)

### API Keys
- **OpenAI API Key**: Required for LLM feature extraction
- Set as environment variable: `OPENAI_API_KEY`

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/kaljuvee/sec-8k-predictor.git
cd sec-8k-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/kaljuvee/sec-8k-predictor.git
cd sec-8k-predictor
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run Home.py
```

### 3. Setup Databases (In-App)
1. Open the Streamlit application in your browser
2. Navigate to **🚀 Setup Database** page in the sidebar
3. Configure your OpenAI API key for feature extraction
4. Choose between:
   - **Quick Demo Setup**: Sample data (3 companies, fast setup)
   - **Custom Setup**: Configure tickers, date ranges, and data limits
5. Click "Run Selected Steps" to download and process data

### 4. Explore Features
Once databases are populated, explore:
- **📊 Data Overview**: View SEC filings and stock data
- **🔍 Feature Analysis**: Analyze LLM-extracted features
- **🤖 Model Performance**: Train and evaluate ML models
- **🎯 Predictions**: Make predictions on new filings

## 📊 Web Interface

The Streamlit application provides an intuitive interface with:
- **Database Setup**: Automated data download and processing
- **Interactive Dashboards**: Real-time data visualization
- **Model Training**: Point-and-click ML model training
- **Prediction Interface**: Easy-to-use prediction tools

## 🖥️ CLI Interface (Alternative)

For advanced users, the CLI provides full system control:
```

### Available Pages
- **📊 Data Overview**: Explore SEC filings and stock data
- **🔍 Feature Analysis**: Analyze extracted features
- **🤖 Model Performance**: View ML model results
- **🎯 Predictions**: Make predictions on filings

## 💻 CLI Reference

### Data Download
```bash
# Download SEC filings
python cli.py download filings [OPTIONS]
  --tickers TEXT          Comma-separated tickers
  --start-date TEXT       Start date (YYYY-MM-DD)
  --end-date TEXT         End date (YYYY-MM-DD)
  --batch-size INTEGER    Batch size
  --limit INTEGER         Limit number of tickers

# Download stock data
python cli.py download stocks [OPTIONS]
  # Same options as filings
```

### Feature Extraction
```bash
# Extract features from filings
python cli.py features extract [OPTIONS]
  --limit INTEGER         Number of filings to process

# View features summary
python cli.py features summary
```

### Model Training
```bash
# Train all models
python cli.py models train

# Train specific categories
python cli.py models train --categories "8.01,2.02"

# View model summary
python cli.py models summary
```

### System Status
```bash
# Check data status
python cli.py status data

# Run quickstart demo
python cli.py quickstart --sample-size 5
```

## 🏗️ Architecture

### Data Flow
```
SEC EDGAR → SEC Filings DB → Feature Extraction → Features DB
     ↓              ↓                ↓               ↓
Stock APIs → Stock Data DB → Return Calculation → ML Training → Trained Models
                                                        ↓
                                                   Predictions
```

### Components

#### 1. SEC Downloader (`src/sec_downloader.py`)
- Downloads 8-K filings from EDGAR
- Extracts filing categories and content
- Stores in SQLite database

#### 2. Stock Data Collector (`src/stock_data.py`)
- Fetches historical stock data
- Calculates returns and volatility
- Normalizes against market (SPY)

#### 3. Feature Extractor (`src/feature_extraction.py`)
- LLM-based content analysis
- TF-IDF vectorization
- Target variable alignment

#### 4. ML Models (`src/models.py`)
- Random Forest classifiers and regressors
- Category-specific training
- Performance evaluation

#### 5. CLI Interface (`cli.py`)
- Command-line access to all functionality
- Batch processing capabilities
- Status monitoring

#### 6. Web Interface (`app.py`, `pages/`)
- Interactive Streamlit dashboard
- Data visualization with Plotly
- Real-time predictions

## 📈 Model Performance

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: R², MAE, MSE, Correlation
- **Cross-Validation**: 5-fold CV for robustness
- **Feature Importance**: Top predictive features

### Expected Performance
- **Baseline Accuracy**: ~55-65% (better than random)
- **R² Scores**: 0.1-0.3 (typical for financial prediction)
- **Best Categories**: Material agreements (1.01), Results (2.02)

## 🧪 Testing

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Tests
```bash
python run_tests.py test_sec_downloader
python run_tests.py test_models
```

### Integration Test
```bash
python test_integration.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
python -c "import openai; print(openai.api_key)"
```

#### 2. Database Locked
```bash
# Check for running processes
ps aux | grep python

# Remove lock files if safe
rm data/*.db-wal data/*.db-shm
```

#### 3. Memory Issues
```bash
# Reduce batch sizes
python cli.py download filings --batch-size 5 --limit 10
```

## 📚 API Reference

### SEC8KDownloader
```python
from src.sec_downloader import SEC8KDownloader

downloader = SEC8KDownloader(data_dir="data")
downloader.download_8k_filings(
    tickers=["AAPL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### StockDataCollector
```python
from src.stock_data import StockDataCollector

collector = StockDataCollector(data_dir="data")
collector.download_stock_data(
    tickers=["AAPL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### FeatureExtractor
```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(data_dir="data")
extractor.process_filings(limit=100)
extractor.create_tfidf_features()
```

### SEC8KPredictor
```python
from src.models import SEC8KPredictor

predictor = SEC8KPredictor(data_dir="data")
results = predictor.train_all_models()
predictions = predictor.predict(features, "8.01", "regressor")
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run code formatting
black src/ tests/

# Run tests
python run_tests.py
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **SEC EDGAR**: For providing public access to filing data
- **OpenAI**: For LLM capabilities in feature extraction
- **Yahoo Finance**: For stock market data
- **Streamlit**: For the web interface framework
- **scikit-learn**: For machine learning algorithms

---

**Built with ❤️ for financial data science**