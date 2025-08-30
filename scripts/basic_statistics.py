import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import warnings
import time
from functools import wraps
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')

def setup_logging() -> logging.Logger:

    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/statistics.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def timing_decorator(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"Completed {func.__name__} in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"Failed {func.__name__} after {end_time - start_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

logger = setup_logging()

class BasicStatisticsAnalyzer:
  
    def __init__(self, db_path: str = "airport_data.db"):
     
        self.db_path = db_path
        self.datasets = {}
        self.statistics = {}
        
        os.makedirs('statistics_output', exist_ok=True)
        os.makedirs('statistics_output/reports', exist_ok=True)
        
    def load_data(self, dataset_name: str, source: str = 'auto') -> pd.DataFrame:
       
        logger.info(f"Loading {dataset_name} for analysis...")
        
        if source == 'auto':
       
            for src in ['pickle', 'sqlite', 'csv']:
                try:
                    df = self._load_from_source(dataset_name, src)
                    self.datasets[dataset_name] = df
                    logger.info(f"Successfully loaded {dataset_name} from {src}: {len(df)} rows")
                    return df
                except Exception as e:
                    logger.debug(f"Failed to load from {src}: {e}")
                    continue
            raise FileNotFoundError(f"Could not load {dataset_name} from any source")
        else:
            df = self._load_from_source(dataset_name, source)
            self.datasets[dataset_name] = df
            return df
    
    def _load_from_source(self, dataset_name: str, source: str) -> pd.DataFrame:
        """Load data from specific source."""
        if source == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            try:
                return pd.read_sql_query(f"SELECT * FROM {dataset_name}", conn)
            finally:
                conn.close()
        elif source == 'pickle':
            pickle_files = list(Path('processed_data').glob(f"{dataset_name}_*.pkl"))
            if not pickle_files:
                raise FileNotFoundError(f"No pickle files found for {dataset_name}")
            latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
            return pd.read_pickle(latest_file)
        elif source == 'csv':
      
            csv_paths = [
                Path('data') / f"{dataset_name}.csv",
                Path('data') / f"{dataset_name}.csv.csv",
                Path('processed_data') / f"{dataset_name}_preprocessed.csv"
            ]
            
     
            processed_csvs = list(Path('processed_data').glob(f"{dataset_name}_*.csv"))
            if processed_csvs:
                csv_paths.append(max(processed_csvs, key=lambda x: x.stat().st_mtime))
            
            for csv_path in csv_paths:
                if csv_path.exists():
                    return pd.read_csv(csv_path, comment='#')
            
            raise FileNotFoundError(f"CSV file not found for {dataset_name}")
        else:
            raise ValueError(f"Unsupported source: {source}")

    @timing_decorator
    def compute_comprehensive_statistics(self, dataset_name: str) -> Dict[str, Any]:
      
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)
        
        df = self.datasets[dataset_name]
        logger.info(f"Computing comprehensive statistics for {dataset_name}...")
        
        stats = {
            'dataset_info': {
                'name': dataset_name,
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            },
            'data_quality': {
                'total_missing': int(df.isnull().sum().sum()),
                'missing_by_column': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                'missing_percentage': {str(k): float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()},
                'duplicate_rows': int(df.duplicated().sum()),
                'duplicate_percentage': float(df.duplicated().sum() / len(df) * 100)
            },
            'numeric_analysis': {},
            'categorical_analysis': {},
            'datetime_analysis': {}
        }
        
   
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            
            stats['numeric_analysis'] = {
                'columns': list(numeric_cols),
                'summary_statistics': {
                    str(col): {
                        str(stat): float(val) if not pd.isna(val) else None
                        for stat, val in numeric_stats[col].items()
                    }
                    for col in numeric_cols
                },
                'distribution_metrics': {
                    str(col): {
                        'skewness': float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                        'kurtosis': float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None,
                        'variance': float(df[col].var()) if not pd.isna(df[col].var()) else None,
                        'std_dev': float(df[col].std()) if not pd.isna(df[col].std()) else None
                    }
                    for col in numeric_cols
                },
                'outlier_analysis': self._analyze_outliers(df[numeric_cols])
            }
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                stats['numeric_analysis']['correlation_matrix'] = {
                    str(i): {str(j): float(corr_matrix.loc[i, j]) if not pd.isna(corr_matrix.loc[i, j]) else None
                            for j in corr_matrix.columns}
                    for i in corr_matrix.index
                }
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                cat_stats[str(col)] = {
                    'unique_count': int(df[col].nunique()),
                    'most_frequent': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'top_10_values': {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()},
                    'cardinality_ratio': float(df[col].nunique() / len(df))
                }
            stats['categorical_analysis'] = cat_stats
        
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            dt_stats = {}
            for col in datetime_cols:
                dt_stats[str(col)] = {
                    'min_date': str(df[col].min()),
                    'max_date': str(df[col].max()),
                    'date_range_days': int((df[col].max() - df[col].min()).days),
                    'unique_dates': int(df[col].nunique()),
                    'most_common_date': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None
                }
                
           
                if not df[col].isnull().all():
                    dt_stats[str(col)]['patterns'] = {
                        'hour_distribution': {str(k): int(v) for k, v in df[col].dt.hour.value_counts().head(10).to_dict().items()},
                        'day_of_week_distribution': {str(k): int(v) for k, v in df[col].dt.dayofweek.value_counts().to_dict().items()},
                        'month_distribution': {str(k): int(v) for k, v in df[col].dt.month.value_counts().to_dict().items()}
                    }
            
            stats['datetime_analysis'] = dt_stats
        
        self.statistics[dataset_name] = stats
        
        stats_file = f"statistics_output/reports/{dataset_name}_comprehensive_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Comprehensive statistics computed and saved for {dataset_name}")
        return stats
    
    def _analyze_outliers(self, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        
        outlier_analysis = {}
        
        for col in numeric_df.columns:
            if numeric_df[col].isnull().all():
                continue
                
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
            
            outlier_analysis[str(col)] = {
                'outlier_count': int(len(outliers)),
                'outlier_percentage': float(len(outliers) / len(numeric_df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR)
            }
        
        return outlier_analysis

    def generate_summary_report(self, dataset_name: str) -> str:
       
        if dataset_name not in self.statistics:
            self.compute_comprehensive_statistics(dataset_name)
        
        stats = self.statistics[dataset_name]
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"COMPREHENSIVE DATA ANALYSIS REPORT")
        report_lines.append(f"Dataset: {dataset_name}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        info = stats['dataset_info']
        report_lines.append(f"\nDATASET OVERVIEW")
        report_lines.append(f"   Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
        report_lines.append(f"   Memory Usage: {info['memory_usage_mb']:.2f} MB")
        
        quality = stats['data_quality']
        report_lines.append(f"\nDATA QUALITY")
        report_lines.append(f"   Missing Values: {quality['total_missing']:,} ({quality['total_missing']/info['shape'][0]*100:.2f}%)")
        report_lines.append(f"   Duplicate Rows: {quality['duplicate_rows']:,} ({quality['duplicate_percentage']:.2f}%)")
        
        if quality['total_missing'] > 0:
            report_lines.append(f"   Columns with Missing Values:")
            for col, missing in quality['missing_by_column'].items():
                if missing > 0:
                    report_lines.append(f"     • {col}: {missing:,} ({quality['missing_percentage'][col]:.2f}%)")
        
        if stats['numeric_analysis']:
            numeric = stats['numeric_analysis']
            report_lines.append(f"\nNUMERIC COLUMNS ANALYSIS")
            report_lines.append(f"   Number of numeric columns: {len(numeric['columns'])}")
            
            for col in numeric['columns']:
                summary = numeric['summary_statistics'][col]
                dist = numeric['distribution_metrics'][col]
                outliers = numeric['outlier_analysis'][col]
                
                report_lines.append(f"\n   {col}:")
                report_lines.append(f"      Range: {summary['min']:.2f} to {summary['max']:.2f}")
                report_lines.append(f"      Mean: {summary['mean']:.2f}, Median: {summary['50%']:.2f}")
                report_lines.append(f"      Std Dev: {dist['std_dev']:.2f}, Skewness: {dist['skewness']:.2f}")
                report_lines.append(f"      Outliers: {outliers['outlier_count']:,} ({outliers['outlier_percentage']:.2f}%)")
        
        if stats['categorical_analysis']:
            categorical = stats['categorical_analysis']
            report_lines.append(f"\nCATEGORICAL COLUMNS ANALYSIS")
            report_lines.append(f"   Number of categorical columns: {len(categorical)}")
            
            for col, cat_stats in categorical.items():
                report_lines.append(f"\n   {col}:")
                report_lines.append(f"      Unique Values: {cat_stats['unique_count']:,}")
                report_lines.append(f"      Most Frequent: '{cat_stats['most_frequent']}' ({cat_stats['most_frequent_count']:,} times)")
                report_lines.append(f"      Cardinality Ratio: {cat_stats['cardinality_ratio']:.4f}")
      
        if stats['datetime_analysis']:
            datetime_stats = stats['datetime_analysis']
            report_lines.append(f"\nDATETIME COLUMNS ANALYSIS")
            
            for col, dt_stats in datetime_stats.items():
                report_lines.append(f"\n   {col}:")
                report_lines.append(f"      Date Range: {dt_stats['min_date']} to {dt_stats['max_date']}")
                report_lines.append(f"      Span: {dt_stats['date_range_days']:,} days")
                report_lines.append(f"      Unique Dates: {dt_stats['unique_dates']:,}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("End of Report")
        report_lines.append("="*80)
        
        report_content = "\n".join(report_lines)
        report_file = f"statistics_output/reports/{dataset_name}_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Summary report generated for {dataset_name}: {report_file}")
        return report_file

def main():
   
    try:
        analyzer = BasicStatisticsAnalyzer()
        
        datasets = ['airport_traffic', 'flight_info', 'passenger_profiles', 'esop_transactions']
        
        print("Starting Basic Statistics Analysis...")
        print("="*60)
        
        all_results = {}
        
        for dataset in datasets:
            try:
                print(f"\nAnalyzing {dataset}...")
                stats = analyzer.compute_comprehensive_statistics(dataset)
                report_file = analyzer.generate_summary_report(dataset)
                
                all_results[dataset] = {
                    'statistics': stats,
                    'report_file': report_file
                }
                
                print(f"✓ {dataset} analysis completed")
                print(f"  Statistics: statistics_output/reports/{dataset}_comprehensive_stats.json")
                print(f"  Report: {report_file}")
                
            except Exception as e:
                print(f"✗ Failed to analyze {dataset}: {str(e)}")
                logger.error(f"Failed to analyze {dataset}: {str(e)}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total datasets analyzed: {len(all_results)}")
        
        for dataset, results in all_results.items():
            if 'statistics' in results:
                info = results['statistics']['dataset_info']
                quality = results['statistics']['data_quality']
                print(f"\n{dataset}:")
                print(f"  Shape: {info['shape']}")
                print(f"  Memory: {info['memory_usage_mb']:.2f} MB")
                print(f"  Missing: {quality['total_missing']:,} values")
                print(f"  Duplicates: {quality['duplicate_rows']:,} rows")
        
        return analyzer, all_results
        
    except Exception as e:
        logger.error(f"Statistics analysis pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    analyzer, results = main()
