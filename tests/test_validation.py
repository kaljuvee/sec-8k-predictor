#!/usr/bin/env python3
"""
Independent Validation Framework for SEC 8-K Prediction Results

This module provides comprehensive validation of all prediction results to detect:
1. Data leakage and look-ahead bias
2. Calculation errors in returns and metrics
3. Suspicious patterns indicating bugs
4. Statistical inconsistencies

The validation is completely independent of the main prediction system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsValidator:
    """Independent validator for all prediction system results."""
    
    def __init__(self, test_data_dir="../test-data"):
        """Initialize the validator.
        
        Args:
            test_data_dir (str): Directory containing test data
        """
        self.test_data_dir = Path(test_data_dir)
        self.validation_results = {}
        self.critical_issues = []
        self.warnings = []
        
    def validate_all_results(self):
        """Run comprehensive validation of all results.
        
        Returns:
            dict: Comprehensive validation report
        """
        logger.info("Starting comprehensive results validation")
        
        # Load all data files
        data_files = self.load_all_data_files()
        
        # Validate each component
        self.validate_forward_returns(data_files)
        self.validate_prediction_accuracy(data_files)
        self.validate_temporal_consistency(data_files)
        self.validate_statistical_properties(data_files)
        self.validate_no_lookahead_bias(data_files)
        self.detect_suspicious_patterns(data_files)
        
        # Generate comprehensive report
        validation_report = self.generate_validation_report()
        
        # Save validation results
        self.save_validation_results(validation_report)
        
        logger.info("Comprehensive validation completed")
        return validation_report
    
    def load_all_data_files(self):
        """Load all available data files for validation.
        
        Returns:
            dict: Dictionary of loaded dataframes
        """
        logger.info("Loading all data files for validation")
        
        data_files = {}
        
        # List of expected files
        file_patterns = [
            'no_lookahead_dataset.csv',
            'no_lookahead_walk_forward_summary.csv',
            'sp500_comprehensive_predictions.csv',
            'sp500_comprehensive_summary.csv',
            'real_predictions_*.csv'
        ]
        
        # Load CSV files
        for file_path in self.test_data_dir.glob('*.csv'):
            try:
                df = pd.read_csv(file_path)
                data_files[file_path.name] = df
                logger.info(f"Loaded {file_path.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
        
        # Load JSON files
        for file_path in self.test_data_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                data_files[file_path.name] = data
                logger.info(f"Loaded {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
        
        return data_files
    
    def validate_forward_returns(self, data_files):
        """Validate forward return calculations.
        
        Args:
            data_files (dict): Loaded data files
        """
        logger.info("Validating forward return calculations")
        
        issues = []
        
        # Check no-lookahead dataset
        if 'no_lookahead_dataset.csv' in data_files:
            df = data_files['no_lookahead_dataset.csv']
            
            # Validate return relationships
            if 'return_5d' in df.columns and 'return_9d' in df.columns:
                # Check if 5d and 9d returns are identical (suspicious)
                identical_returns = (df['return_5d'] == df['return_9d']).sum()
                if identical_returns > len(df) * 0.1:  # More than 10% identical
                    issues.append(f"Suspicious: {identical_returns} identical 5d and 9d returns")
                
                # Check return magnitudes
                return_5d_stats = df['return_5d'].describe()
                return_9d_stats = df['return_9d'].describe()
                
                # 9d returns should generally have higher volatility
                if return_9d_stats['std'] <= return_5d_stats['std']:
                    issues.append("Warning: 9d returns have lower volatility than 5d returns")
                
                # Check for unrealistic returns
                extreme_5d = (abs(df['return_5d']) > 0.5).sum()
                extreme_9d = (abs(df['return_9d']) > 0.5).sum()
                
                if extreme_5d > 0:
                    issues.append(f"Warning: {extreme_5d} extreme 5d returns (>50%)")
                if extreme_9d > 0:
                    issues.append(f"Warning: {extreme_9d} extreme 9d returns (>50%)")
        
        # Validate SP500 predictions
        if 'sp500_comprehensive_predictions.csv' in data_files:
            df = data_files['sp500_comprehensive_predictions.csv']
            
            # Check for identical predicted vs actual returns
            if 'predicted_return_5d' in df.columns and 'actual_return_5d' in df.columns:
                identical_pred_actual = (df['predicted_return_5d'] == df['actual_return_5d']).sum()
                if identical_pred_actual > 0:
                    issues.append(f"CRITICAL: {identical_pred_actual} identical predicted and actual returns")
                
                # Check prediction vs actual correlation (should not be perfect)
                correlation = df['predicted_return_5d'].corr(df['actual_return_5d'])
                if correlation > 0.95:
                    issues.append(f"CRITICAL: Suspiciously high prediction correlation: {correlation:.3f}")
        
        self.validation_results['forward_returns'] = {
            'issues': issues,
            'status': 'CRITICAL' if any('CRITICAL' in issue for issue in issues) else 'WARNING' if issues else 'PASS'
        }
        
        if issues:
            logger.warning(f"Forward returns validation found {len(issues)} issues")
            for issue in issues:
                if 'CRITICAL' in issue:
                    self.critical_issues.append(issue)
                else:
                    self.warnings.append(issue)
    
    def validate_prediction_accuracy(self, data_files):
        """Validate prediction accuracy calculations.
        
        Args:
            data_files (dict): Loaded data files
        """
        logger.info("Validating prediction accuracy calculations")
        
        issues = []
        
        # Check walk-forward summary
        if 'no_lookahead_walk_forward_summary.csv' in data_files:
            df = data_files['no_lookahead_walk_forward_summary.csv']
            
            # Validate accuracy ranges
            if 'directional_accuracy' in df.columns:
                accuracy_stats = df['directional_accuracy'].describe()
                
                # Check for unrealistic accuracy
                if accuracy_stats['max'] > 0.9:
                    issues.append(f"CRITICAL: Unrealistic max accuracy: {accuracy_stats['max']:.3f}")
                
                if accuracy_stats['min'] < 0.1:
                    issues.append(f"Warning: Very low min accuracy: {accuracy_stats['min']:.3f}")
                
                # Check for constant accuracy (suspicious)
                if accuracy_stats['std'] < 0.01:
                    issues.append("CRITICAL: Accuracy has no variation (constant values)")
                
                # Validate accuracy is between 0 and 1
                invalid_accuracy = ((df['directional_accuracy'] < 0) | (df['directional_accuracy'] > 1)).sum()
                if invalid_accuracy > 0:
                    issues.append(f"CRITICAL: {invalid_accuracy} invalid accuracy values")
        
        # Validate SP500 summary
        if 'sp500_comprehensive_summary.csv' in data_files:
            df = data_files['sp500_comprehensive_summary.csv']
            
            # Check for reasonable accuracy ranges
            accuracy_columns = [col for col in df.columns if 'accuracy' in col.lower()]
            for col in accuracy_columns:
                if col in df.columns:
                    invalid_values = ((df[col] < 0) | (df[col] > 1)).sum()
                    if invalid_values > 0:
                        issues.append(f"CRITICAL: {invalid_values} invalid values in {col}")
        
        self.validation_results['prediction_accuracy'] = {
            'issues': issues,
            'status': 'CRITICAL' if any('CRITICAL' in issue for issue in issues) else 'WARNING' if issues else 'PASS'
        }
        
        if issues:
            logger.warning(f"Prediction accuracy validation found {len(issues)} issues")
            for issue in issues:
                if 'CRITICAL' in issue:
                    self.critical_issues.append(issue)
                else:
                    self.warnings.append(issue)
    
    def validate_temporal_consistency(self, data_files):
        """Validate temporal consistency and ordering.
        
        Args:
            data_files (dict): Loaded data files
        """
        logger.info("Validating temporal consistency")
        
        issues = []
        
        # Check no-lookahead dataset
        if 'no_lookahead_dataset.csv' in data_files:
            df = data_files['no_lookahead_dataset.csv']
            
            # Convert datetime columns
            datetime_columns = ['filing_datetime', 'info_available_datetime']
            for col in datetime_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        issues.append(f"CRITICAL: Cannot parse datetime column {col}")
                        continue
            
            # Validate temporal ordering
            if 'filing_datetime' in df.columns and 'info_available_datetime' in df.columns:
                # Info available time should be after filing time
                invalid_ordering = (df['info_available_datetime'] <= df['filing_datetime']).sum()
                if invalid_ordering > 0:
                    issues.append(f"CRITICAL: {invalid_ordering} cases where info_available <= filing_datetime")
                
                # Check for reasonable delays
                delays = (df['info_available_datetime'] - df['filing_datetime']).dt.total_seconds() / 3600
                delay_stats = delays.describe()
                
                if delay_stats['min'] < 0.5:  # Less than 30 minutes
                    issues.append("Warning: Some delays are less than 30 minutes")
                
                if delay_stats['max'] > 72:  # More than 3 days
                    issues.append("Warning: Some delays are more than 3 days")
        
        # Check walk-forward temporal consistency
        if 'no_lookahead_walk_forward_summary.csv' in data_files:
            df = data_files['no_lookahead_walk_forward_summary.csv']
            
            # Convert datetime columns
            for col in ['test_start', 'test_end']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        issues.append(f"CRITICAL: Cannot parse datetime column {col}")
                        continue
            
            # Validate test period ordering
            if 'test_start' in df.columns and 'test_end' in df.columns:
                invalid_periods = (df['test_end'] <= df['test_start']).sum()
                if invalid_periods > 0:
                    issues.append(f"CRITICAL: {invalid_periods} invalid test periods (end <= start)")
                
                # Check for overlapping periods (should not happen in walk-forward)
                df_sorted = df.sort_values('test_start')
                overlaps = 0
                for i in range(1, len(df_sorted)):
                    if df_sorted.iloc[i]['test_start'] < df_sorted.iloc[i-1]['test_end']:
                        overlaps += 1
                
                if overlaps > 0:
                    issues.append(f"CRITICAL: {overlaps} overlapping test periods detected")
        
        self.validation_results['temporal_consistency'] = {
            'issues': issues,
            'status': 'CRITICAL' if any('CRITICAL' in issue for issue in issues) else 'WARNING' if issues else 'PASS'
        }
        
        if issues:
            logger.warning(f"Temporal consistency validation found {len(issues)} issues")
            for issue in issues:
                if 'CRITICAL' in issue:
                    self.critical_issues.append(issue)
                else:
                    self.warnings.append(issue)
    
    def validate_statistical_properties(self, data_files):
        """Validate statistical properties of the data.
        
        Args:
            data_files (dict): Loaded data files
        """
        logger.info("Validating statistical properties")
        
        issues = []
        
        # Check return distributions
        if 'no_lookahead_dataset.csv' in data_files:
            df = data_files['no_lookahead_dataset.csv']
            
            return_columns = ['return_1d', 'return_3d', 'return_5d', 'return_9d', 'return_10d']
            for col in return_columns:
                if col in df.columns:
                    # Test for normality (returns should be approximately normal)
                    if len(df[col].dropna()) > 8:  # Minimum for Shapiro-Wilk
                        try:
                            stat, p_value = stats.shapiro(df[col].dropna()[:5000])  # Limit sample size
                            if p_value < 0.001:  # Very non-normal
                                issues.append(f"Warning: {col} appears highly non-normal (p={p_value:.6f})")
                        except:
                            pass
                    
                    # Check for excessive skewness
                    skewness = stats.skew(df[col].dropna())
                    if abs(skewness) > 5:
                        issues.append(f"Warning: {col} has extreme skewness: {skewness:.2f}")
                    
                    # Check for excessive kurtosis
                    kurt = stats.kurtosis(df[col].dropna())
                    if kurt > 10:
                        issues.append(f"Warning: {col} has extreme kurtosis: {kurt:.2f}")
        
        # Check prediction accuracy distributions
        if 'no_lookahead_walk_forward_summary.csv' in data_files:
            df = data_files['no_lookahead_walk_forward_summary.csv']
            
            if 'directional_accuracy' in df.columns:
                accuracy_values = df['directional_accuracy'].dropna()
                
                # Test if accuracy is significantly different from 0.5
                if len(accuracy_values) > 1:
                    t_stat, p_value = stats.ttest_1samp(accuracy_values, 0.5)
                    
                    # Store statistical test results
                    self.validation_results['statistical_significance'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'mean_accuracy': accuracy_values.mean(),
                        'significantly_better_than_random': p_value < 0.05 and accuracy_values.mean() > 0.5
                    }
                    
                    if p_value >= 0.05:
                        issues.append(f"Warning: Accuracy not significantly different from random (p={p_value:.3f})")
        
        self.validation_results['statistical_properties'] = {
            'issues': issues,
            'status': 'WARNING' if issues else 'PASS'
        }
        
        if issues:
            logger.warning(f"Statistical properties validation found {len(issues)} issues")
            for issue in issues:
                self.warnings.append(issue)
    
    def validate_no_lookahead_bias(self, data_files):
        """Validate that no look-ahead bias exists.
        
        Args:
            data_files (dict): Loaded data files
        """
        logger.info("Validating absence of look-ahead bias")
        
        issues = []
        
        # Check for perfect correlations (sign of data leakage)
        prediction_files = [f for f in data_files.keys() if 'prediction' in f and f.endswith('.csv')]
        
        for file_name in prediction_files:
            df = data_files[file_name]
            
            # Look for predicted vs actual columns
            pred_cols = [col for col in df.columns if 'predicted' in col.lower()]
            actual_cols = [col for col in df.columns if 'actual' in col.lower()]
            
            for pred_col in pred_cols:
                for actual_col in actual_cols:
                    if pred_col in df.columns and actual_col in df.columns:
                        # Check for perfect correlation (only for numeric columns)
                        try:
                            valid_data = df[[pred_col, actual_col]].dropna()
                            # Ensure both columns are numeric
                            if (pd.api.types.is_numeric_dtype(valid_data[pred_col]) and 
                                pd.api.types.is_numeric_dtype(valid_data[actual_col]) and 
                                len(valid_data) > 1):
                                
                                correlation = valid_data[pred_col].corr(valid_data[actual_col])
                                
                                if correlation > 0.99:
                                    issues.append(f"CRITICAL: Perfect correlation between {pred_col} and {actual_col} in {file_name}")
                                elif correlation > 0.9:
                                    issues.append(f"Warning: Very high correlation ({correlation:.3f}) between {pred_col} and {actual_col} in {file_name}")
                        except (ValueError, TypeError):
                            # Skip non-numeric columns
                            continue
        
        # Check for unrealistic performance consistency
        if 'no_lookahead_walk_forward_summary.csv' in data_files:
            df = data_files['no_lookahead_walk_forward_summary.csv']
            
            if 'directional_accuracy' in df.columns:
                accuracy_std = df['directional_accuracy'].std()
                
                # Too little variation suggests overfitting or data leakage
                if accuracy_std < 0.02:
                    issues.append("CRITICAL: Accuracy variation too low (possible overfitting)")
                
                # Check for impossible accuracy patterns
                max_accuracy = df['directional_accuracy'].max()
                if max_accuracy > 0.85:
                    issues.append(f"CRITICAL: Unrealistically high accuracy: {max_accuracy:.3f}")
        
        self.validation_results['lookahead_bias'] = {
            'issues': issues,
            'status': 'CRITICAL' if any('CRITICAL' in issue for issue in issues) else 'WARNING' if issues else 'PASS'
        }
        
        if issues:
            logger.warning(f"Look-ahead bias validation found {len(issues)} issues")
            for issue in issues:
                if 'CRITICAL' in issue:
                    self.critical_issues.append(issue)
                else:
                    self.warnings.append(issue)
    
    def detect_suspicious_patterns(self, data_files):
        """Detect suspicious patterns that might indicate bugs.
        
        Args:
            data_files (dict): Loaded data files
        """
        logger.info("Detecting suspicious patterns")
        
        issues = []
        
        # Check for repeated values
        for file_name, df in data_files.items():
            if isinstance(df, pd.DataFrame):
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if col in df.columns:
                        # Check for too many repeated values
                        value_counts = df[col].value_counts()
                        most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
                        
                        if most_common_count > len(df) * 0.5:  # More than 50% same value
                            issues.append(f"Warning: {col} in {file_name} has {most_common_count} repeated values")
                        
                        # Check for suspicious round numbers
                        if col in ['directional_accuracy', 'prediction_confidence']:
                            round_values = df[col].apply(lambda x: x in [0.0, 0.5, 1.0] if pd.notna(x) else False).sum()
                            if round_values > len(df) * 0.3:
                                issues.append(f"Warning: {col} in {file_name} has many round values")
        
        # Check for missing data patterns
        for file_name, df in data_files.items():
            if isinstance(df, pd.DataFrame):
                missing_data = df.isnull().sum()
                total_missing = missing_data.sum()
                
                if total_missing > len(df) * len(df.columns) * 0.1:  # More than 10% missing
                    issues.append(f"Warning: {file_name} has high missing data rate")
                
                # Check for columns that are entirely missing
                completely_missing = missing_data[missing_data == len(df)]
                if len(completely_missing) > 0:
                    issues.append(f"Warning: {file_name} has completely empty columns: {list(completely_missing.index)}")
        
        self.validation_results['suspicious_patterns'] = {
            'issues': issues,
            'status': 'WARNING' if issues else 'PASS'
        }
        
        if issues:
            logger.warning(f"Suspicious pattern detection found {len(issues)} issues")
            for issue in issues:
                self.warnings.append(issue)
    
    def generate_validation_report(self):
        """Generate comprehensive validation report.
        
        Returns:
            dict: Validation report
        """
        logger.info("Generating validation report")
        
        # Determine overall status
        overall_status = 'PASS'
        if self.critical_issues:
            overall_status = 'CRITICAL'
        elif self.warnings:
            overall_status = 'WARNING'
        
        # Count issues by category
        issue_summary = {}
        for category, results in self.validation_results.items():
            if 'issues' in results:
                issue_summary[category] = {
                    'total_issues': len(results['issues']),
                    'status': results['status'],
                    'critical_issues': len([i for i in results['issues'] if 'CRITICAL' in i]),
                    'warnings': len([i for i in results['issues'] if 'Warning' in i])
                }
        
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'total_critical_issues': len(self.critical_issues),
            'total_warnings': len(self.warnings),
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'detailed_results': self.validation_results,
            'issue_summary': issue_summary,
            'recommendations': self.generate_recommendations()
        }
        
        return validation_report
    
    def generate_recommendations(self):
        """Generate recommendations based on validation results.
        
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        if self.critical_issues:
            recommendations.append("URGENT: Address critical issues before using results")
            
            # Specific recommendations for common issues
            for issue in self.critical_issues:
                if 'identical predicted and actual returns' in issue:
                    recommendations.append("Fix data leakage: Predicted and actual returns should never be identical")
                elif 'Perfect correlation' in issue:
                    recommendations.append("Investigate data leakage: Perfect correlations indicate look-ahead bias")
                elif 'Unrealistic' in issue and 'accuracy' in issue:
                    recommendations.append("Review model: Accuracy above 85% is unrealistic for financial predictions")
        
        if self.warnings:
            recommendations.append("Review warnings to improve result quality")
        
        if not self.critical_issues and not self.warnings:
            recommendations.append("Results appear valid and ready for use")
        
        return recommendations
    
    def save_validation_results(self, validation_report):
        """Save validation results to files.
        
        Args:
            validation_report (dict): Validation report to save
        """
        # Save JSON report
        with open(self.test_data_dir / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for category, results in validation_report['detailed_results'].items():
            if 'issues' in results:
                summary_data.append({
                    'category': category,
                    'status': results['status'],
                    'total_issues': len(results['issues']),
                    'critical_issues': len([i for i in results['issues'] if 'CRITICAL' in i]),
                    'warnings': len([i for i in results['issues'] if 'Warning' in i])
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.test_data_dir / "validation_summary.csv", index=False)
        
        logger.info("Validation results saved")

def main():
    """Main function for running validation."""
    validator = ResultsValidator()
    
    print("🔍 Starting comprehensive results validation...")
    
    # Run validation
    report = validator.validate_all_results()
    
    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Critical Issues: {report['total_critical_issues']}")
    print(f"Warnings: {report['total_warnings']}")
    
    if report['critical_issues']:
        print(f"\n🚨 Critical Issues:")
        for issue in report['critical_issues']:
            print(f"  - {issue}")
    
    if report['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in report['warnings'][:5]:  # Show first 5
            print(f"  - {warning}")
        if len(report['warnings']) > 5:
            print(f"  ... and {len(report['warnings']) - 5} more warnings")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    print(f"\n📁 Detailed results saved to: test-data/validation_report.json")

if __name__ == "__main__":
    main()

