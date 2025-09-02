#!/usr/bin/env python3
"""
Comprehensive S&P 500 Prediction Testing

This module runs comprehensive prediction tests across all S&P 500 companies
and generates detailed analysis reports.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from sp500_data_collector import SP500DataCollector
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SP500ComprehensiveAnalyzer:
    """Comprehensive analyzer for S&P 500 prediction performance."""
    
    def __init__(self, test_data_dir="../test-data"):
        """Initialize the analyzer.
        
        Args:
            test_data_dir (str): Directory for test data output
        """
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True)
        
        self.collector = SP500DataCollector(data_dir="../data")
    
    def run_comprehensive_analysis(self, sample_size=100, period="2y"):
        """Run comprehensive analysis across S&P 500 companies.
        
        Args:
            sample_size (int): Number of companies to analyze (None for all)
            period (str): Time period for analysis
            
        Returns:
            dict: Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive S&P 500 analysis (sample_size={sample_size})...")
        
        # Collect data
        companies_df, filings_df, summary_stats = self.collector.collect_sp500_prediction_data(
            sample_size=sample_size,
            period=period
        )
        
        # Generate comprehensive analysis
        analysis_results = self.generate_comprehensive_analysis(companies_df, filings_df, summary_stats)
        
        # Save detailed results
        self.save_analysis_results(analysis_results, companies_df, filings_df)
        
        return analysis_results
    
    def generate_comprehensive_analysis(self, companies_df, filings_df, summary_stats):
        """Generate comprehensive analysis from collected data.
        
        Args:
            companies_df (pd.DataFrame): Company data
            filings_df (pd.DataFrame): Filing predictions
            summary_stats (dict): Summary statistics
            
        Returns:
            dict: Comprehensive analysis results
        """
        logger.info("Generating comprehensive analysis...")
        
        analysis = {
            'meta': {
                'analysis_date': datetime.now().isoformat(),
                'total_companies': len(companies_df),
                'total_predictions': len(filings_df),
                'analysis_scope': 'S&P 500 Comprehensive Analysis'
            }
        }
        
        # Overall performance metrics
        analysis['overall_performance'] = {
            'accuracy_5d': float(filings_df['prediction_accuracy_5d'].mean()),
            'accuracy_9d': float(filings_df['prediction_accuracy_9d'].mean()),
            'accuracy_improvement_5d_vs_random': float(filings_df['prediction_accuracy_5d'].mean() - 0.5),
            'accuracy_improvement_9d_vs_random': float(filings_df['prediction_accuracy_9d'].mean() - 0.5),
            'avg_confidence': float(filings_df['model_confidence'].mean()),
            'confidence_std': float(filings_df['model_confidence'].std()),
            'positive_prediction_rate': float((filings_df['predicted_direction'] == 'positive').mean()),
            'prediction_error_5d': float(abs(filings_df['predicted_return_5d'] - filings_df['actual_return_5d']).mean()),
            'prediction_error_9d': float(abs(filings_df['predicted_return_9d'] - filings_df['actual_return_9d']).mean()),
            'return_correlation_5d': float(filings_df['predicted_return_5d'].corr(filings_df['actual_return_5d'])),
            'return_correlation_9d': float(filings_df['predicted_return_9d'].corr(filings_df['actual_return_9d']))
        }
        
        # Sector analysis
        analysis['sector_analysis'] = self.analyze_by_sector(companies_df, filings_df)
        
        # Category analysis
        analysis['category_analysis'] = self.analyze_by_category(filings_df)
        
        # Market cap analysis (if available)
        analysis['market_cap_analysis'] = self.analyze_by_market_cap(companies_df, filings_df)
        
        # Time horizon comparison
        analysis['time_horizon_comparison'] = self.compare_time_horizons(filings_df)
        
        # Statistical significance tests
        analysis['statistical_tests'] = self.perform_statistical_tests(filings_df)
        
        # Performance distribution analysis
        analysis['performance_distribution'] = self.analyze_performance_distribution(filings_df)
        
        # Risk-adjusted metrics
        analysis['risk_metrics'] = self.calculate_risk_metrics(filings_df)
        
        return analysis
    
    def analyze_by_sector(self, companies_df, filings_df):
        """Analyze performance by GICS sector.
        
        Args:
            companies_df (pd.DataFrame): Company data
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Sector analysis results
        """
        logger.info("Analyzing performance by sector...")
        
        sector_analysis = {}
        
        for sector in companies_df['GICS_Sector'].unique():
            sector_companies = companies_df[companies_df['GICS_Sector'] == sector]['Symbol'].tolist()
            sector_filings = filings_df[filings_df['ticker'].isin(sector_companies)]
            
            if len(sector_filings) > 0:
                sector_analysis[sector] = {
                    'companies_count': len(sector_companies),
                    'predictions_count': len(sector_filings),
                    'accuracy_5d': float(sector_filings['prediction_accuracy_5d'].mean()),
                    'accuracy_9d': float(sector_filings['prediction_accuracy_9d'].mean()),
                    'avg_confidence': float(sector_filings['model_confidence'].mean()),
                    'avg_actual_return_5d': float(sector_filings['actual_return_5d'].mean()),
                    'avg_actual_return_9d': float(sector_filings['actual_return_9d'].mean()),
                    'prediction_error_5d': float(abs(sector_filings['predicted_return_5d'] - sector_filings['actual_return_5d']).mean()),
                    'prediction_error_9d': float(abs(sector_filings['predicted_return_9d'] - sector_filings['actual_return_9d']).mean()),
                    'positive_prediction_rate': float((sector_filings['predicted_direction'] == 'positive').mean()),
                    'volatility_5d': float(sector_filings['actual_return_5d'].std()),
                    'volatility_9d': float(sector_filings['actual_return_9d'].std())
                }
        
        # Rank sectors by performance
        sector_rankings = {
            'by_accuracy_5d': sorted(sector_analysis.items(), 
                                   key=lambda x: x[1]['accuracy_5d'], reverse=True),
            'by_accuracy_9d': sorted(sector_analysis.items(), 
                                   key=lambda x: x[1]['accuracy_9d'], reverse=True),
            'by_return_5d': sorted(sector_analysis.items(), 
                                 key=lambda x: x[1]['avg_actual_return_5d'], reverse=True)
        }
        
        return {
            'sector_performance': sector_analysis,
            'sector_rankings': sector_rankings
        }
    
    def analyze_by_category(self, filings_df):
        """Analyze performance by SEC filing category.
        
        Args:
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Category analysis results
        """
        logger.info("Analyzing performance by filing category...")
        
        category_analysis = {}
        
        for category in filings_df['category'].unique():
            cat_filings = filings_df[filings_df['category'] == category]
            
            category_analysis[category] = {
                'predictions_count': len(cat_filings),
                'accuracy_5d': float(cat_filings['prediction_accuracy_5d'].mean()),
                'accuracy_9d': float(cat_filings['prediction_accuracy_9d'].mean()),
                'avg_confidence': float(cat_filings['model_confidence'].mean()),
                'avg_actual_return_5d': float(cat_filings['actual_return_5d'].mean()),
                'avg_actual_return_9d': float(cat_filings['actual_return_9d'].mean()),
                'prediction_error_5d': float(abs(cat_filings['predicted_return_5d'] - cat_filings['actual_return_5d']).mean()),
                'prediction_error_9d': float(abs(cat_filings['predicted_return_9d'] - cat_filings['actual_return_9d']).mean()),
                'positive_prediction_rate': float((cat_filings['predicted_direction'] == 'positive').mean())
            }
        
        # Rank categories by performance
        category_rankings = {
            'by_accuracy_5d': sorted(category_analysis.items(), 
                                   key=lambda x: x[1]['accuracy_5d'], reverse=True),
            'by_accuracy_9d': sorted(category_analysis.items(), 
                                   key=lambda x: x[1]['accuracy_9d'], reverse=True),
            'by_confidence': sorted(category_analysis.items(), 
                                  key=lambda x: x[1]['avg_confidence'], reverse=True)
        }
        
        return {
            'category_performance': category_analysis,
            'category_rankings': category_rankings
        }
    
    def analyze_by_market_cap(self, companies_df, filings_df):
        """Analyze performance by market capitalization tiers.
        
        Args:
            companies_df (pd.DataFrame): Company data
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Market cap analysis results
        """
        logger.info("Analyzing performance by market cap tiers...")
        
        # Create market cap tiers based on company position in S&P 500
        companies_df = companies_df.reset_index(drop=True)
        num_companies = len(companies_df)
        
        if num_companies <= 10:
            # For small samples, use simple tiers
            companies_df['market_cap_tier'] = 'All Companies'
            tier_analysis = {}
            
            tier_companies = companies_df['Symbol'].tolist()
            tier_filings = filings_df[filings_df['ticker'].isin(tier_companies)]
            
            if len(tier_filings) > 0:
                tier_analysis['All Companies'] = {
                    'companies_count': len(tier_companies),
                    'predictions_count': len(tier_filings),
                    'accuracy_5d': float(tier_filings['prediction_accuracy_5d'].mean()),
                    'accuracy_9d': float(tier_filings['prediction_accuracy_9d'].mean()),
                    'avg_confidence': float(tier_filings['model_confidence'].mean()),
                    'avg_actual_return_5d': float(tier_filings['actual_return_5d'].mean()),
                    'avg_actual_return_9d': float(tier_filings['actual_return_9d'].mean()),
                    'prediction_error_5d': float(abs(tier_filings['predicted_return_5d'] - tier_filings['actual_return_5d']).mean()),
                    'prediction_error_9d': float(abs(tier_filings['predicted_return_9d'] - tier_filings['actual_return_9d']).mean())
                }
        else:
            # For larger samples, create meaningful tiers
            if num_companies >= 100:
                bins = [0, 33, 66, num_companies]
                labels = ['Large Cap', 'Mid Cap', 'Small Cap']
            else:
                bins = [0, num_companies // 2, num_companies]
                labels = ['Large Cap', 'Small Cap']
            
            companies_df['market_cap_tier'] = pd.cut(
                companies_df.index, 
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            
            tier_analysis = {}
            
            for tier in companies_df['market_cap_tier'].unique():
                tier_companies = companies_df[companies_df['market_cap_tier'] == tier]['Symbol'].tolist()
                tier_filings = filings_df[filings_df['ticker'].isin(tier_companies)]
                
                if len(tier_filings) > 0:
                    tier_analysis[str(tier)] = {
                        'companies_count': len(tier_companies),
                        'predictions_count': len(tier_filings),
                        'accuracy_5d': float(tier_filings['prediction_accuracy_5d'].mean()),
                        'accuracy_9d': float(tier_filings['prediction_accuracy_9d'].mean()),
                        'avg_confidence': float(tier_filings['model_confidence'].mean()),
                        'avg_actual_return_5d': float(tier_filings['actual_return_5d'].mean()),
                        'avg_actual_return_9d': float(tier_filings['actual_return_9d'].mean()),
                        'prediction_error_5d': float(abs(tier_filings['predicted_return_5d'] - tier_filings['actual_return_5d']).mean()),
                        'prediction_error_9d': float(abs(tier_filings['predicted_return_9d'] - tier_filings['actual_return_9d']).mean())
                    }
        
        return tier_analysis
    
    def compare_time_horizons(self, filings_df):
        """Compare performance across different time horizons.
        
        Args:
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Time horizon comparison results
        """
        logger.info("Comparing time horizon performance...")
        
        comparison = {
            '5_day': {
                'accuracy': float(filings_df['prediction_accuracy_5d'].mean()),
                'prediction_error': float(abs(filings_df['predicted_return_5d'] - filings_df['actual_return_5d']).mean()),
                'return_correlation': float(filings_df['predicted_return_5d'].corr(filings_df['actual_return_5d'])),
                'avg_actual_return': float(filings_df['actual_return_5d'].mean()),
                'return_volatility': float(filings_df['actual_return_5d'].std())
            },
            '9_day': {
                'accuracy': float(filings_df['prediction_accuracy_9d'].mean()),
                'prediction_error': float(abs(filings_df['predicted_return_9d'] - filings_df['actual_return_9d']).mean()),
                'return_correlation': float(filings_df['predicted_return_9d'].corr(filings_df['actual_return_9d'])),
                'avg_actual_return': float(filings_df['actual_return_9d'].mean()),
                'return_volatility': float(filings_df['actual_return_9d'].std())
            }
        }
        
        # Calculate degradation metrics
        comparison['degradation_metrics'] = {
            'accuracy_degradation': comparison['5_day']['accuracy'] - comparison['9_day']['accuracy'],
            'error_increase': comparison['9_day']['prediction_error'] - comparison['5_day']['prediction_error'],
            'correlation_decay': comparison['5_day']['return_correlation'] - comparison['9_day']['return_correlation'],
            'return_amplification': comparison['9_day']['avg_actual_return'] / comparison['5_day']['avg_actual_return'] if comparison['5_day']['avg_actual_return'] != 0 else 1.0
        }
        
        return comparison
    
    def perform_statistical_tests(self, filings_df):
        """Perform statistical significance tests.
        
        Args:
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Statistical test results
        """
        logger.info("Performing statistical significance tests...")
        
        from scipy import stats
        
        # Test if accuracy is significantly better than random (50%)
        accuracy_5d = filings_df['prediction_accuracy_5d']
        accuracy_9d = filings_df['prediction_accuracy_9d']
        
        # One-sample t-test against 50% accuracy
        t_stat_5d, p_value_5d = stats.ttest_1samp(accuracy_5d, 0.5)
        t_stat_9d, p_value_9d = stats.ttest_1samp(accuracy_9d, 0.5)
        
        # Paired t-test between 5-day and 9-day accuracy
        t_stat_paired, p_value_paired = stats.ttest_rel(accuracy_5d, accuracy_9d)
        
        return {
            'accuracy_vs_random_5d': {
                't_statistic': float(t_stat_5d),
                'p_value': float(p_value_5d),
                'significant': p_value_5d < 0.05,
                'interpretation': 'Significantly better than random' if p_value_5d < 0.05 else 'Not significantly better than random'
            },
            'accuracy_vs_random_9d': {
                't_statistic': float(t_stat_9d),
                'p_value': float(p_value_9d),
                'significant': p_value_9d < 0.05,
                'interpretation': 'Significantly better than random' if p_value_9d < 0.05 else 'Not significantly better than random'
            },
            'horizon_comparison': {
                't_statistic': float(t_stat_paired),
                'p_value': float(p_value_paired),
                'significant': p_value_paired < 0.05,
                'interpretation': '5-day significantly better than 9-day' if t_stat_paired > 0 and p_value_paired < 0.05 else 'No significant difference'
            }
        }
    
    def analyze_performance_distribution(self, filings_df):
        """Analyze the distribution of prediction performance.
        
        Args:
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Performance distribution analysis
        """
        logger.info("Analyzing performance distribution...")
        
        # Accuracy distribution
        accuracy_5d = filings_df['prediction_accuracy_5d']
        accuracy_9d = filings_df['prediction_accuracy_9d']
        
        # Confidence distribution
        confidence = filings_df['model_confidence']
        
        # Return prediction error distribution
        error_5d = abs(filings_df['predicted_return_5d'] - filings_df['actual_return_5d'])
        error_9d = abs(filings_df['predicted_return_9d'] - filings_df['actual_return_9d'])
        
        return {
            'accuracy_distribution_5d': {
                'mean': float(accuracy_5d.mean()),
                'std': float(accuracy_5d.std()),
                'min': float(accuracy_5d.min()),
                'max': float(accuracy_5d.max()),
                'percentiles': {
                    '25th': float(accuracy_5d.quantile(0.25)),
                    '50th': float(accuracy_5d.quantile(0.50)),
                    '75th': float(accuracy_5d.quantile(0.75)),
                    '90th': float(accuracy_5d.quantile(0.90))
                }
            },
            'accuracy_distribution_9d': {
                'mean': float(accuracy_9d.mean()),
                'std': float(accuracy_9d.std()),
                'min': float(accuracy_9d.min()),
                'max': float(accuracy_9d.max()),
                'percentiles': {
                    '25th': float(accuracy_9d.quantile(0.25)),
                    '50th': float(accuracy_9d.quantile(0.50)),
                    '75th': float(accuracy_9d.quantile(0.75)),
                    '90th': float(accuracy_9d.quantile(0.90))
                }
            },
            'confidence_distribution': {
                'mean': float(confidence.mean()),
                'std': float(confidence.std()),
                'percentiles': {
                    '10th': float(confidence.quantile(0.10)),
                    '25th': float(confidence.quantile(0.25)),
                    '50th': float(confidence.quantile(0.50)),
                    '75th': float(confidence.quantile(0.75)),
                    '90th': float(confidence.quantile(0.90))
                }
            },
            'error_distribution_5d': {
                'mean': float(error_5d.mean()),
                'std': float(error_5d.std()),
                'percentiles': {
                    '50th': float(error_5d.quantile(0.50)),
                    '75th': float(error_5d.quantile(0.75)),
                    '90th': float(error_5d.quantile(0.90)),
                    '95th': float(error_5d.quantile(0.95))
                }
            },
            'error_distribution_9d': {
                'mean': float(error_9d.mean()),
                'std': float(error_9d.std()),
                'percentiles': {
                    '50th': float(error_9d.quantile(0.50)),
                    '75th': float(error_9d.quantile(0.75)),
                    '90th': float(error_9d.quantile(0.90)),
                    '95th': float(error_9d.quantile(0.95))
                }
            }
        }
    
    def calculate_risk_metrics(self, filings_df):
        """Calculate risk-adjusted performance metrics.
        
        Args:
            filings_df (pd.DataFrame): Filing predictions
            
        Returns:
            dict: Risk metrics
        """
        logger.info("Calculating risk-adjusted metrics...")
        
        # Calculate Sharpe-like ratios for prediction accuracy
        accuracy_5d = filings_df['prediction_accuracy_5d'].mean()
        accuracy_9d = filings_df['prediction_accuracy_9d'].mean()
        
        error_5d = abs(filings_df['predicted_return_5d'] - filings_df['actual_return_5d']).mean()
        error_9d = abs(filings_df['predicted_return_9d'] - filings_df['actual_return_9d']).mean()
        
        # Information ratio (excess accuracy / prediction error)
        info_ratio_5d = (accuracy_5d - 0.5) / error_5d if error_5d > 0 else 0
        info_ratio_9d = (accuracy_9d - 0.5) / error_9d if error_9d > 0 else 0
        
        # Maximum drawdown in accuracy (consecutive wrong predictions)
        accuracy_series_5d = filings_df['prediction_accuracy_5d']
        accuracy_series_9d = filings_df['prediction_accuracy_9d']
        
        # Calculate rolling accuracy over windows
        window_size = 20
        rolling_acc_5d = accuracy_series_5d.rolling(window=window_size).mean()
        rolling_acc_9d = accuracy_series_9d.rolling(window=window_size).mean()
        
        max_drawdown_5d = (rolling_acc_5d.max() - rolling_acc_5d.min()) if len(rolling_acc_5d.dropna()) > 0 else 0
        max_drawdown_9d = (rolling_acc_9d.max() - rolling_acc_9d.min()) if len(rolling_acc_9d.dropna()) > 0 else 0
        
        return {
            'information_ratio_5d': float(info_ratio_5d),
            'information_ratio_9d': float(info_ratio_9d),
            'max_accuracy_drawdown_5d': float(max_drawdown_5d),
            'max_accuracy_drawdown_9d': float(max_drawdown_9d),
            'prediction_consistency_5d': float(1 - accuracy_series_5d.std()),
            'prediction_consistency_9d': float(1 - accuracy_series_9d.std()),
            'risk_adjusted_return_5d': float(filings_df['actual_return_5d'].mean() / filings_df['actual_return_5d'].std()) if filings_df['actual_return_5d'].std() > 0 else 0,
            'risk_adjusted_return_9d': float(filings_df['actual_return_9d'].mean() / filings_df['actual_return_9d'].std()) if filings_df['actual_return_9d'].std() > 0 else 0
        }
    
    def save_analysis_results(self, analysis_results, companies_df, filings_df):
        """Save comprehensive analysis results.
        
        Args:
            analysis_results (dict): Analysis results
            companies_df (pd.DataFrame): Company data
            filings_df (pd.DataFrame): Filing predictions
        """
        logger.info("Saving comprehensive analysis results...")
        
        # Save main analysis results
        with open(self.test_data_dir / "sp500_comprehensive_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save detailed CSV files
        filings_df.to_csv(self.test_data_dir / "sp500_comprehensive_predictions.csv", index=False)
        companies_df.to_csv(self.test_data_dir / "sp500_comprehensive_companies.csv", index=False)
        
        # Create summary CSV
        summary_data = []
        
        # Overall metrics
        overall = analysis_results['overall_performance']
        summary_data.append({
            'metric_type': 'Overall',
            'category': 'All',
            'accuracy_5d': overall['accuracy_5d'],
            'accuracy_9d': overall['accuracy_9d'],
            'confidence': overall['avg_confidence'],
            'error_5d': overall['prediction_error_5d'],
            'error_9d': overall['prediction_error_9d'],
            'sample_size': analysis_results['meta']['total_predictions']
        })
        
        # Sector metrics
        for sector, metrics in analysis_results['sector_analysis']['sector_performance'].items():
            summary_data.append({
                'metric_type': 'Sector',
                'category': sector,
                'accuracy_5d': metrics['accuracy_5d'],
                'accuracy_9d': metrics['accuracy_9d'],
                'confidence': metrics['avg_confidence'],
                'error_5d': metrics['prediction_error_5d'],
                'error_9d': metrics['prediction_error_9d'],
                'sample_size': metrics['predictions_count']
            })
        
        # Category metrics
        for category, metrics in analysis_results['category_analysis']['category_performance'].items():
            summary_data.append({
                'metric_type': 'Filing_Category',
                'category': category,
                'accuracy_5d': metrics['accuracy_5d'],
                'accuracy_9d': metrics['accuracy_9d'],
                'confidence': metrics['avg_confidence'],
                'error_5d': metrics['prediction_error_5d'],
                'error_9d': metrics['prediction_error_9d'],
                'sample_size': metrics['predictions_count']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.test_data_dir / "sp500_comprehensive_summary.csv", index=False)
        
        logger.info(f"Analysis results saved to: {self.test_data_dir}")

def main():
    """Main function for running comprehensive S&P 500 analysis."""
    analyzer = SP500ComprehensiveAnalyzer()
    
    # Run comprehensive analysis
    print("🚀 Starting comprehensive S&P 500 prediction analysis...")
    
    # Start with a substantial sample
    analysis_results = analyzer.run_comprehensive_analysis(
        sample_size=100,  # Analyze 100 companies
        period="2y"       # 2 years of data
    )
    
    # Print key results
    overall = analysis_results['overall_performance']
    meta = analysis_results['meta']
    
    print(f"\n🎉 Comprehensive Analysis Completed!")
    print(f"📊 Companies Analyzed: {meta['total_companies']}")
    print(f"📈 Total Predictions: {meta['total_predictions']}")
    print(f"🎯 5-Day Accuracy: {overall['accuracy_5d']:.1%}")
    print(f"🎯 9-Day Accuracy: {overall['accuracy_9d']:.1%}")
    print(f"📊 Average Confidence: {overall['avg_confidence']:.1%}")
    print(f"📉 5-Day Error: {overall['prediction_error_5d']:.2%}")
    print(f"📉 9-Day Error: {overall['prediction_error_9d']:.2%}")
    
    # Statistical significance
    stats = analysis_results['statistical_tests']
    print(f"\n📊 Statistical Significance:")
    print(f"5-Day vs Random: {stats['accuracy_vs_random_5d']['interpretation']}")
    print(f"9-Day vs Random: {stats['accuracy_vs_random_9d']['interpretation']}")
    print(f"Horizon Comparison: {stats['horizon_comparison']['interpretation']}")
    
    # Top performing sectors
    sector_rankings = analysis_results['sector_analysis']['sector_rankings']
    print(f"\n🏆 Top Performing Sectors (5-Day Accuracy):")
    for i, (sector, metrics) in enumerate(sector_rankings['by_accuracy_5d'][:3]):
        print(f"{i+1}. {sector}: {metrics['accuracy_5d']:.1%}")
    
    print(f"\n📁 Detailed results saved to: test-data/")

if __name__ == "__main__":
    main()

