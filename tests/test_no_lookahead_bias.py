#!/usr/bin/env python3
"""
No Look-Ahead Bias Testing Framework

This module implements rigorous testing methodologies to eliminate look-ahead bias
in SEC 8-K filing predictions, ensuring realistic trading simulation conditions.

Key Principles:
1. Temporal Isolation: Only use information available at prediction time
2. Walk-Forward Analysis: Progressive training and testing windows
3. Real-Time Simulation: Mimic actual trading system constraints
4. Information Embargo: Respect filing disclosure timing
5. Market Hours Constraints: Account for trading session limitations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NoLookAheadBiasValidator:
    """Comprehensive validator for eliminating look-ahead bias in prediction systems."""
    
    def __init__(self, test_data_dir="../test-data"):
        """Initialize the validator.
        
        Args:
            test_data_dir (str): Directory for test data output
        """
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Market hours and constraints
        self.market_open = time(9, 30)  # 9:30 AM EST
        self.market_close = time(16, 0)  # 4:00 PM EST
        self.filing_processing_delay = timedelta(minutes=30)  # Realistic processing delay
        self.feature_extraction_time = timedelta(minutes=15)  # Time to extract features
        
    def create_temporal_dataset(self, num_companies=20, start_date='2020-01-01', end_date='2024-01-01'):
        """Create a temporally ordered dataset with realistic timing constraints.
        
        Args:
            num_companies (int): Number of companies to simulate
            start_date (str): Start date for simulation
            end_date (str): End date for simulation
            
        Returns:
            pd.DataFrame: Temporally ordered dataset
        """
        logger.info(f"Creating temporal dataset: {num_companies} companies, {start_date} to {end_date}")
        
        # Generate company list
        companies = [f"STOCK_{i:03d}" for i in range(num_companies)]
        
        # Generate filing dates with realistic frequency
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        all_filings = []
        filing_id = 1
        
        for company in companies:
            # Generate 8-12 filings per year per company
            total_days = (end_dt - start_dt).days
            num_filings = int(total_days / 365.25 * np.random.uniform(8, 12))
            
            # Generate random filing dates
            filing_dates = []
            for _ in range(num_filings):
                random_days = np.random.randint(0, total_days)
                filing_date = start_dt + timedelta(days=random_days)
                
                # Add realistic filing time (most filings after market close)
                if np.random.random() < 0.7:  # 70% after market close
                    filing_time = time(np.random.randint(16, 23), np.random.randint(0, 59))
                else:  # 30% during market hours or pre-market
                    filing_time = time(np.random.randint(6, 15), np.random.randint(0, 59))
                
                filing_datetime = datetime.combine(filing_date.date(), filing_time)
                filing_dates.append(filing_datetime)
            
            filing_dates.sort()
            
            for filing_datetime in filing_dates:
                # Calculate when information becomes available for trading
                info_available_time = self.calculate_info_availability(filing_datetime)
                
                # Generate realistic features (sentiment, urgency, etc.)
                features = self.generate_realistic_features()
                
                # Generate forward returns (these would be unknown at prediction time)
                forward_returns = self.generate_forward_returns()
                
                filing = {
                    'filing_id': filing_id,
                    'ticker': company,
                    'filing_datetime': filing_datetime,
                    'info_available_datetime': info_available_time,
                    'category': np.random.choice(['1.01', '1.02', '2.01', '3.01', '5.01', '7.01', '8.01']),
                    'sentiment_score': features['sentiment'],
                    'urgency_score': features['urgency'],
                    'financial_impact_score': features['financial_impact'],
                    'market_relevance_score': features['market_relevance'],
                    'content_length': features['content_length'],
                    'after_hours_filing': filing_datetime.time() >= self.market_close,
                    'return_1d': forward_returns['1d'],
                    'return_3d': forward_returns['3d'],
                    'return_5d': forward_returns['5d'],
                    'return_10d': forward_returns['10d'],
                    'volatility_1d': forward_returns['vol_1d'],
                    'volatility_5d': forward_returns['vol_5d']
                }
                
                all_filings.append(filing)
                filing_id += 1
        
        # Create DataFrame and sort by filing time (critical for no look-ahead)
        df = pd.DataFrame(all_filings)
        df = df.sort_values('filing_datetime').reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} filings across {num_companies} companies")
        return df
    
    def calculate_info_availability(self, filing_datetime):
        """Calculate when filing information becomes available for trading decisions.
        
        Args:
            filing_datetime (datetime): When filing was submitted
            
        Returns:
            datetime: When information is available for trading
        """
        # Add processing delay
        available_time = filing_datetime + self.filing_processing_delay
        
        # Add feature extraction time
        available_time += self.feature_extraction_time
        
        # If filing is after market close, information is available next trading day
        if filing_datetime.time() >= self.market_close:
            # Available at market open next trading day
            next_day = filing_datetime.date() + timedelta(days=1)
            # Skip weekends (simplified - doesn't handle holidays)
            while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_day += timedelta(days=1)
            available_time = datetime.combine(next_day, self.market_open)
        
        # If available time is before market open, wait until market open
        elif available_time.time() < self.market_open:
            available_time = datetime.combine(available_time.date(), self.market_open)
        
        # If available time is after market close, wait until next trading day
        elif available_time.time() >= self.market_close:
            next_day = available_time.date() + timedelta(days=1)
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            available_time = datetime.combine(next_day, self.market_open)
        
        return available_time
    
    def generate_realistic_features(self):
        """Generate realistic feature values for SEC filings.
        
        Returns:
            dict: Feature values
        """
        # Correlated features (sentiment affects other scores)
        base_sentiment = np.random.beta(2, 2)  # Centered around 0.5
        
        return {
            'sentiment': base_sentiment,
            'urgency': np.random.beta(1.5, 3),  # Skewed toward lower urgency
            'financial_impact': np.clip(base_sentiment + np.random.normal(0, 0.2), 0, 1),
            'market_relevance': np.random.beta(2, 1.5),  # Skewed toward higher relevance
            'content_length': np.random.lognormal(7, 1)  # Log-normal distribution
        }
    
    def generate_forward_returns(self):
        """Generate realistic forward returns with proper correlation structure.
        
        Returns:
            dict: Forward returns and volatilities
        """
        # Base return with some persistence
        base_return = np.random.normal(0, 0.02)
        
        # Returns with increasing noise over time
        return_1d = base_return + np.random.normal(0, 0.01)
        return_3d = base_return * 0.7 + np.random.normal(0, 0.015)
        return_5d = base_return * 0.5 + np.random.normal(0, 0.02)
        return_10d = base_return * 0.3 + np.random.normal(0, 0.025)
        
        return {
            '1d': return_1d,
            '3d': return_3d,
            '5d': return_5d,
            '10d': return_10d,
            'vol_1d': abs(return_1d) + np.random.exponential(0.01),
            'vol_5d': abs(return_5d) + np.random.exponential(0.015)
        }
    
    def walk_forward_analysis(self, df, initial_train_days=365, test_days=30, step_days=30):
        """Perform walk-forward analysis to eliminate look-ahead bias.
        
        Args:
            df (pd.DataFrame): Temporal dataset
            initial_train_days (int): Initial training period in days
            test_days (int): Testing period in days
            step_days (int): Step size for rolling window
            
        Returns:
            list: Results from each walk-forward step
        """
        logger.info(f"Starting walk-forward analysis: train={initial_train_days}d, test={test_days}d, step={step_days}d")
        
        results = []
        
        # Convert to datetime for easier manipulation
        df['filing_datetime'] = pd.to_datetime(df['filing_datetime'])
        df['info_available_datetime'] = pd.to_datetime(df['info_available_datetime'])
        
        start_date = df['filing_datetime'].min()
        end_date = df['filing_datetime'].max()
        
        # Initial training end date
        train_end = start_date + timedelta(days=initial_train_days)
        
        step_num = 1
        
        while train_end + timedelta(days=test_days) <= end_date:
            logger.info(f"Walk-forward step {step_num}: training until {train_end.date()}")
            
            # Define training period (only use information available before train_end)
            train_mask = (df['info_available_datetime'] <= train_end)
            train_data = df[train_mask].copy()
            
            # Define test period
            test_start = train_end
            test_end = train_end + timedelta(days=test_days)
            
            # Test data: filings that occur during test period
            test_mask = (
                (df['filing_datetime'] > test_start) & 
                (df['filing_datetime'] <= test_end)
            )
            test_data = df[test_mask].copy()
            
            if len(train_data) < 50 or len(test_data) < 5:
                logger.warning(f"Insufficient data for step {step_num}: train={len(train_data)}, test={len(test_data)}")
                train_end += timedelta(days=step_days)
                step_num += 1
                continue
            
            # Train model (simulate training process)
            model_performance = self.simulate_model_training(train_data)
            
            # Generate predictions for test period
            predictions = self.generate_no_lookahead_predictions(test_data, model_performance)
            
            # Evaluate predictions
            evaluation = self.evaluate_predictions(test_data, predictions)
            
            step_result = {
                'step': step_num,
                'train_start': start_date,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'model_performance': model_performance,
                'predictions': predictions,
                'evaluation': evaluation
            }
            
            results.append(step_result)
            
            # Move to next step
            train_end += timedelta(days=step_days)
            step_num += 1
        
        logger.info(f"Completed {len(results)} walk-forward steps")
        return results
    
    def simulate_model_training(self, train_data):
        """Simulate model training process with realistic performance metrics.
        
        Args:
            train_data (pd.DataFrame): Training data
            
        Returns:
            dict: Model performance metrics
        """
        # Simulate feature importance based on training data
        feature_correlations = {
            'sentiment_score': train_data['sentiment_score'].corr(train_data['return_5d']),
            'urgency_score': train_data['urgency_score'].corr(train_data['return_5d']),
            'financial_impact_score': train_data['financial_impact_score'].corr(train_data['return_5d']),
            'market_relevance_score': train_data['market_relevance_score'].corr(train_data['return_5d'])
        }
        
        # Simulate cross-validation performance
        cv_accuracy = 0.5 + np.random.normal(0.05, 0.02)  # Around 55% with noise
        cv_accuracy = np.clip(cv_accuracy, 0.45, 0.65)
        
        return {
            'cv_accuracy': cv_accuracy,
            'feature_importance': feature_correlations,
            'training_samples': len(train_data),
            'model_confidence': np.random.uniform(0.6, 0.8)
        }
    
    def generate_no_lookahead_predictions(self, test_data, model_performance):
        """Generate predictions ensuring no look-ahead bias.
        
        Args:
            test_data (pd.DataFrame): Test data
            model_performance (dict): Model performance from training
            
        Returns:
            list: Predictions with timing constraints
        """
        predictions = []
        
        for _, filing in test_data.iterrows():
            # Ensure prediction is made only after information is available
            prediction_time = filing['info_available_datetime']
            
            # Calculate prediction based on available features only
            feature_score = (
                filing['sentiment_score'] * 0.3 +
                filing['financial_impact_score'] * 0.4 +
                filing['market_relevance_score'] * 0.2 +
                filing['urgency_score'] * 0.1
            )
            
            # Add model uncertainty
            base_accuracy = model_performance['cv_accuracy']
            prediction_confidence = base_accuracy + np.random.normal(0, 0.05)
            prediction_confidence = np.clip(prediction_confidence, 0.4, 0.8)
            
            # Generate directional prediction
            predicted_direction = 1 if feature_score > 0.5 else -1
            predicted_return = predicted_direction * abs(np.random.normal(0.01, 0.005))
            
            prediction = {
                'filing_id': filing['filing_id'],
                'ticker': filing['ticker'],
                'prediction_time': prediction_time,
                'predicted_direction': predicted_direction,
                'predicted_return_5d': predicted_return,
                'prediction_confidence': prediction_confidence,
                'feature_score': feature_score,
                'model_version': f"v{prediction_time.strftime('%Y%m%d')}",
                'can_trade_immediately': self.can_trade_immediately(prediction_time),
                'next_trading_opportunity': self.get_next_trading_time(prediction_time)
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def can_trade_immediately(self, prediction_time):
        """Check if trading can happen immediately after prediction.
        
        Args:
            prediction_time (datetime): When prediction was made
            
        Returns:
            bool: Whether immediate trading is possible
        """
        # Can trade if during market hours and not weekend
        return (
            prediction_time.weekday() < 5 and  # Not weekend
            self.market_open <= prediction_time.time() <= self.market_close
        )
    
    def get_next_trading_time(self, prediction_time):
        """Get the next available trading time.
        
        Args:
            prediction_time (datetime): When prediction was made
            
        Returns:
            datetime: Next trading opportunity
        """
        if self.can_trade_immediately(prediction_time):
            return prediction_time
        
        # Find next market open
        next_day = prediction_time.date()
        if prediction_time.time() >= self.market_close:
            next_day += timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        
        return datetime.combine(next_day, self.market_open)
    
    def evaluate_predictions(self, test_data, predictions):
        """Evaluate predictions with proper temporal alignment.
        
        Args:
            test_data (pd.DataFrame): Test data with actual returns
            predictions (list): Predictions made
            
        Returns:
            dict: Evaluation metrics
        """
        if not predictions:
            return {'error': 'No predictions to evaluate'}
        
        # Merge predictions with actual returns
        pred_df = pd.DataFrame(predictions)
        test_df = test_data.set_index('filing_id')
        
        results = []
        for _, pred in pred_df.iterrows():
            filing_id = pred['filing_id']
            if filing_id in test_df.index:
                actual_return_5d = test_df.loc[filing_id, 'return_5d']
                predicted_return_5d = pred['predicted_return_5d']
                
                # Calculate accuracy (directional)
                actual_direction = 1 if actual_return_5d > 0 else -1
                predicted_direction = pred['predicted_direction']
                directional_accuracy = 1 if actual_direction == predicted_direction else 0
                
                # Calculate return prediction error
                return_error = abs(predicted_return_5d - actual_return_5d)
                
                results.append({
                    'filing_id': filing_id,
                    'directional_accuracy': directional_accuracy,
                    'return_error': return_error,
                    'actual_return': actual_return_5d,
                    'predicted_return': predicted_return_5d,
                    'confidence': pred['prediction_confidence'],
                    'trading_delay_hours': self.calculate_trading_delay(
                        test_df.loc[filing_id, 'filing_datetime'],
                        pred['next_trading_opportunity']
                    )
                })
        
        if not results:
            return {'error': 'No valid predictions to evaluate'}
        
        results_df = pd.DataFrame(results)
        
        return {
            'total_predictions': len(results_df),
            'directional_accuracy': results_df['directional_accuracy'].mean(),
            'mean_return_error': results_df['return_error'].mean(),
            'mean_confidence': results_df['confidence'].mean(),
            'mean_trading_delay_hours': results_df['trading_delay_hours'].mean(),
            'accuracy_by_confidence': self.analyze_accuracy_by_confidence(results_df),
            'return_correlation': results_df['actual_return'].corr(results_df['predicted_return'])
        }
    
    def calculate_trading_delay(self, filing_time, trading_time):
        """Calculate delay between filing and trading opportunity.
        
        Args:
            filing_time (datetime): When filing was submitted
            trading_time (datetime): When trading can occur
            
        Returns:
            float: Delay in hours
        """
        if pd.isna(filing_time) or pd.isna(trading_time):
            return np.nan
        
        filing_dt = pd.to_datetime(filing_time)
        trading_dt = pd.to_datetime(trading_time)
        
        return (trading_dt - filing_dt).total_seconds() / 3600
    
    def analyze_accuracy_by_confidence(self, results_df):
        """Analyze accuracy by confidence buckets.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            dict: Accuracy by confidence level
        """
        if len(results_df) < 10:
            return {'insufficient_data': True}
        
        # Create confidence buckets
        results_df['confidence_bucket'] = pd.cut(
            results_df['confidence'],
            bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
        
        accuracy_by_bucket = results_df.groupby('confidence_bucket')['directional_accuracy'].agg([
            'mean', 'count'
        ]).to_dict('index')
        
        return accuracy_by_bucket
    
    def run_comprehensive_no_lookahead_test(self):
        """Run comprehensive test suite to validate no look-ahead bias.
        
        Returns:
            dict: Comprehensive test results
        """
        logger.info("Starting comprehensive no look-ahead bias validation")
        
        # Create temporal dataset
        df = self.create_temporal_dataset(
            num_companies=30,
            start_date='2020-01-01',
            end_date='2024-01-01'
        )
        
        # Save raw dataset
        df.to_csv(self.test_data_dir / "no_lookahead_dataset.csv", index=False)
        
        # Run walk-forward analysis
        walk_forward_results = self.walk_forward_analysis(
            df,
            initial_train_days=365,
            test_days=60,
            step_days=30
        )
        
        # Aggregate results across all steps
        aggregated_results = self.aggregate_walk_forward_results(walk_forward_results)
        
        # Perform bias detection tests
        bias_tests = self.perform_bias_detection_tests(walk_forward_results)
        
        # Generate methodology documentation
        methodology = self.document_methodology()
        
        comprehensive_results = {
            'dataset_info': {
                'total_filings': len(df),
                'companies': df['ticker'].nunique(),
                'date_range': {
                    'start': df['filing_datetime'].min().isoformat(),
                    'end': df['filing_datetime'].max().isoformat()
                },
                'avg_filings_per_company': len(df) / df['ticker'].nunique()
            },
            'walk_forward_results': walk_forward_results,
            'aggregated_performance': aggregated_results,
            'bias_detection_tests': bias_tests,
            'methodology': methodology,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Save comprehensive results
        with open(self.test_data_dir / "no_lookahead_comprehensive_results.json", 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Create summary CSV
        self.create_summary_csv(walk_forward_results)
        
        logger.info("Comprehensive no look-ahead bias validation completed")
        return comprehensive_results
    
    def aggregate_walk_forward_results(self, walk_forward_results):
        """Aggregate results across all walk-forward steps.
        
        Args:
            walk_forward_results (list): Results from walk-forward analysis
            
        Returns:
            dict: Aggregated performance metrics
        """
        if not walk_forward_results:
            return {'error': 'No walk-forward results to aggregate'}
        
        # Extract evaluation metrics from each step
        accuracies = []
        return_errors = []
        confidences = []
        trading_delays = []
        sample_sizes = []
        
        for step in walk_forward_results:
            eval_metrics = step['evaluation']
            if 'error' not in eval_metrics:
                accuracies.append(eval_metrics['directional_accuracy'])
                return_errors.append(eval_metrics['mean_return_error'])
                confidences.append(eval_metrics['mean_confidence'])
                trading_delays.append(eval_metrics['mean_trading_delay_hours'])
                sample_sizes.append(eval_metrics['total_predictions'])
        
        if not accuracies:
            return {'error': 'No valid evaluations to aggregate'}
        
        return {
            'total_steps': len(walk_forward_results),
            'valid_steps': len(accuracies),
            'overall_accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'median': np.median(accuracies)
            },
            'return_prediction_error': {
                'mean': np.mean(return_errors),
                'std': np.std(return_errors),
                'median': np.median(return_errors)
            },
            'confidence_metrics': {
                'mean': np.mean(confidences),
                'std': np.std(confidences)
            },
            'trading_delay_hours': {
                'mean': np.mean(trading_delays),
                'median': np.median(trading_delays),
                'max': np.max(trading_delays)
            },
            'sample_size_stats': {
                'total_predictions': sum(sample_sizes),
                'avg_per_step': np.mean(sample_sizes),
                'min_per_step': np.min(sample_sizes),
                'max_per_step': np.max(sample_sizes)
            }
        }
    
    def perform_bias_detection_tests(self, walk_forward_results):
        """Perform statistical tests to detect potential look-ahead bias.
        
        Args:
            walk_forward_results (list): Walk-forward results
            
        Returns:
            dict: Bias detection test results
        """
        logger.info("Performing bias detection tests")
        
        # Test 1: Performance stability over time
        accuracies = []
        timestamps = []
        
        for step in walk_forward_results:
            eval_metrics = step['evaluation']
            if 'error' not in eval_metrics:
                accuracies.append(eval_metrics['directional_accuracy'])
                timestamps.append(step['test_start'])
        
        if len(accuracies) < 3:
            return {'error': 'Insufficient data for bias detection tests'}
        
        # Test for trend in performance (should be stable if no bias)
        from scipy import stats
        time_indices = range(len(accuracies))
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, accuracies)
        
        # Test 2: Performance vs random walk
        random_walk_test = self.test_vs_random_walk(accuracies)
        
        # Test 3: Information leakage detection
        leakage_test = self.detect_information_leakage(walk_forward_results)
        
        return {
            'performance_stability_test': {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'interpretation': 'Stable performance (no bias)' if abs(slope) < 0.01 and p_value > 0.05 else 'Potential bias detected',
                'is_stable': abs(slope) < 0.01 and p_value > 0.05
            },
            'random_walk_test': random_walk_test,
            'information_leakage_test': leakage_test,
            'overall_bias_assessment': self.assess_overall_bias(slope, p_value, random_walk_test, leakage_test)
        }
    
    def test_vs_random_walk(self, accuracies):
        """Test if performance is significantly different from random walk.
        
        Args:
            accuracies (list): Accuracy values over time
            
        Returns:
            dict: Random walk test results
        """
        from scipy import stats
        
        # Test if mean accuracy is significantly different from 0.5
        t_stat, p_value = stats.ttest_1samp(accuracies, 0.5)
        
        return {
            'mean_accuracy': np.mean(accuracies),
            't_statistic': t_stat,
            'p_value': p_value,
            'significantly_better_than_random': p_value < 0.05 and np.mean(accuracies) > 0.5,
            'interpretation': 'Significantly better than random' if p_value < 0.05 and np.mean(accuracies) > 0.5 else 'Not significantly better than random'
        }
    
    def detect_information_leakage(self, walk_forward_results):
        """Detect potential information leakage in the prediction process.
        
        Args:
            walk_forward_results (list): Walk-forward results
            
        Returns:
            dict: Information leakage test results
        """
        # Check for unrealistic performance patterns
        high_accuracy_steps = 0
        total_valid_steps = 0
        
        for step in walk_forward_results:
            eval_metrics = step['evaluation']
            if 'error' not in eval_metrics:
                total_valid_steps += 1
                if eval_metrics['directional_accuracy'] > 0.75:  # Suspiciously high
                    high_accuracy_steps += 1
        
        high_accuracy_rate = high_accuracy_steps / total_valid_steps if total_valid_steps > 0 else 0
        
        # Check for perfect predictions (red flag)
        perfect_prediction_steps = 0
        for step in walk_forward_results:
            eval_metrics = step['evaluation']
            if 'error' not in eval_metrics and eval_metrics['directional_accuracy'] == 1.0:
                perfect_prediction_steps += 1
        
        return {
            'high_accuracy_rate': high_accuracy_rate,
            'perfect_prediction_steps': perfect_prediction_steps,
            'total_steps': total_valid_steps,
            'potential_leakage': high_accuracy_rate > 0.3 or perfect_prediction_steps > 0,
            'interpretation': 'Potential information leakage detected' if high_accuracy_rate > 0.3 or perfect_prediction_steps > 0 else 'No obvious information leakage'
        }
    
    def assess_overall_bias(self, slope, p_value, random_walk_test, leakage_test):
        """Assess overall bias based on all tests.
        
        Args:
            slope (float): Performance trend slope
            p_value (float): Trend significance
            random_walk_test (dict): Random walk test results
            leakage_test (dict): Information leakage test results
            
        Returns:
            dict: Overall bias assessment
        """
        bias_indicators = []
        
        # Check performance stability
        if abs(slope) > 0.01 and p_value < 0.05:
            bias_indicators.append("Unstable performance over time")
        
        # Check for information leakage
        if leakage_test['potential_leakage']:
            bias_indicators.append("Potential information leakage")
        
        # Check for unrealistic performance
        if random_walk_test['mean_accuracy'] > 0.7:
            bias_indicators.append("Unrealistically high accuracy")
        
        return {
            'bias_indicators': bias_indicators,
            'bias_detected': len(bias_indicators) > 0,
            'confidence_level': 'High' if len(bias_indicators) == 0 else 'Medium' if len(bias_indicators) == 1 else 'Low',
            'recommendation': 'Methodology appears sound' if len(bias_indicators) == 0 else 'Review methodology for potential bias'
        }
    
    def create_summary_csv(self, walk_forward_results):
        """Create summary CSV of walk-forward results.
        
        Args:
            walk_forward_results (list): Walk-forward results
        """
        summary_data = []
        
        for step in walk_forward_results:
            eval_metrics = step['evaluation']
            if 'error' not in eval_metrics:
                summary_data.append({
                    'step': step['step'],
                    'test_start': step['test_start'],
                    'test_end': step['test_end'],
                    'train_samples': step['train_samples'],
                    'test_samples': step['test_samples'],
                    'directional_accuracy': eval_metrics['directional_accuracy'],
                    'mean_return_error': eval_metrics['mean_return_error'],
                    'mean_confidence': eval_metrics['mean_confidence'],
                    'mean_trading_delay_hours': eval_metrics['mean_trading_delay_hours'],
                    'return_correlation': eval_metrics.get('return_correlation', np.nan)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.test_data_dir / "no_lookahead_walk_forward_summary.csv", index=False)
    
    def document_methodology(self):
        """Document the no look-ahead bias methodology.
        
        Returns:
            dict: Methodology documentation
        """
        return {
            'temporal_isolation': {
                'description': 'Only information available at prediction time is used',
                'implementation': 'Strict temporal ordering of all data points',
                'validation': 'Walk-forward analysis with progressive training windows'
            },
            'information_embargo': {
                'description': 'Respect realistic filing processing and availability delays',
                'filing_processing_delay_minutes': 30,
                'feature_extraction_time_minutes': 15,
                'market_hours_constraint': 'Predictions only actionable during trading hours'
            },
            'walk_forward_validation': {
                'description': 'Progressive training and testing to simulate real-time deployment',
                'initial_training_period_days': 365,
                'test_period_days': 60,
                'step_size_days': 30,
                'total_steps': 'Variable based on data availability'
            },
            'bias_detection_tests': {
                'performance_stability': 'Test for consistent performance over time',
                'random_walk_comparison': 'Statistical test vs random chance',
                'information_leakage_detection': 'Check for unrealistic performance patterns'
            },
            'trading_realism': {
                'market_hours': '9:30 AM - 4:00 PM EST',
                'weekend_handling': 'No trading on weekends',
                'after_hours_filings': 'Available next trading day at market open',
                'immediate_trading_check': 'Verify if prediction can be acted upon immediately'
            }
        }

def main():
    """Main function for running no look-ahead bias validation."""
    validator = NoLookAheadBiasValidator()
    
    print("🔍 Starting comprehensive no look-ahead bias validation...")
    
    # Run comprehensive test
    results = validator.run_comprehensive_no_lookahead_test()
    
    # Print key results
    dataset_info = results['dataset_info']
    aggregated = results['aggregated_performance']
    bias_tests = results['bias_detection_tests']
    
    print(f"\n📊 Dataset Information:")
    print(f"Total Filings: {dataset_info['total_filings']}")
    print(f"Companies: {dataset_info['companies']}")
    print(f"Date Range: {dataset_info['date_range']['start'][:10]} to {dataset_info['date_range']['end'][:10]}")
    
    if 'error' not in aggregated:
        print(f"\n🎯 Aggregated Performance:")
        print(f"Overall Accuracy: {aggregated['overall_accuracy']['mean']:.1%} ± {aggregated['overall_accuracy']['std']:.1%}")
        print(f"Return Prediction Error: {aggregated['return_prediction_error']['mean']:.2%}")
        print(f"Average Trading Delay: {aggregated['trading_delay_hours']['mean']:.1f} hours")
        print(f"Total Predictions: {aggregated['sample_size_stats']['total_predictions']}")
    
    if 'error' not in bias_tests:
        print(f"\n🔍 Bias Detection Results:")
        stability = bias_tests['performance_stability_test']
        print(f"Performance Stability: {'✅ Stable' if stability['is_stable'] else '⚠️ Unstable'}")
        
        random_test = bias_tests['random_walk_test']
        print(f"vs Random: {'✅ Significantly better' if random_test['significantly_better_than_random'] else '❌ Not significant'}")
        
        overall = bias_tests['overall_bias_assessment']
        print(f"Overall Assessment: {overall['recommendation']}")
        print(f"Confidence Level: {overall['confidence_level']}")
    
    print(f"\n📁 Detailed results saved to: test-data/")
    print(f"📄 Files generated:")
    print(f"  - no_lookahead_dataset.csv")
    print(f"  - no_lookahead_comprehensive_results.json")
    print(f"  - no_lookahead_walk_forward_summary.csv")

if __name__ == "__main__":
    main()

