import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
import time
from functools import wraps
from datetime import datetime
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_logging() -> logging.Logger:
 
    import os
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/eda.log'),
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
r
logger = setup_logging()

class EDAAnalyzer:
  
    
    def __init__(self, db_path: str = "airport_data.db"):
      
        self.db_path = db_path
        self.datasets = {}
        self.statistics = {}
        self.visualizations = {}
        
        import os
        os.makedirs('eda_output', exist_ok=True)
        os.makedirs('eda_output/plots', exist_ok=True)
        os.makedirs('eda_output/statistics', exist_ok=True)
        
    def load_data(self, dataset_name: str, source: str = 'auto') -> pd.DataFrame:
      
        logger.info(f"Loading {dataset_name} for EDA...")
        
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
            csv_path = Path('data') / f"{dataset_name}.csv"
            if not csv_path.exists():
                # Try alternative naming
                alt_paths = [
                    Path('data') / f"{dataset_name}.csv.csv",
                    Path('data') / f"{dataset_name}_data.csv"
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        csv_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"CSV file not found for {dataset_name}")
            return pd.read_csv(csv_path, comment='#')
        else:
            raise ValueError(f"Unsupported source: {source}")

    @timing_decorator
    def compute_descriptive_statistics(self, dataset_name: str) -> Dict[str, Any]:
  
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)
        
        df = self.datasets[dataset_name]
        logger.info(f"Computing descriptive statistics for {dataset_name}...")
        
        stats = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': df.dtypes.to_dict()
            },
            'missing_values': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'numeric_statistics': {},
            'categorical_statistics': {},
            'datetime_statistics': {}
        }
        

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            stats['numeric_statistics'] = {
                'summary': numeric_stats.to_dict(),
                'skewness': df[numeric_cols].skew().to_dict(),
                'kurtosis': df[numeric_cols].kurtosis().to_dict(),
                'correlation_matrix': df[numeric_cols].corr().to_dict()
            }
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_stats = {}
            for col in categorical_cols:
                cat_stats[col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
            stats['categorical_statistics'] = cat_stats
    
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            dt_stats = {}
            for col in datetime_cols:
                dt_stats[col] = {
                    'min_date': str(df[col].min()),
                    'max_date': str(df[col].max()),
                    'date_range_days': (df[col].max() - df[col].min()).days
                }
            stats['datetime_statistics'] = dt_stats
        

        self.statistics[dataset_name] = stats
        

        stats_file = f"eda_output/statistics/{dataset_name}_statistics.json"
        with open(stats_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_stats = self._convert_numpy_types(stats)
            json.dump(json_stats, f, indent=2, default=str)
        
        logger.info(f"Statistics computed and saved for {dataset_name}")
        return stats
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    @timing_decorator
    def create_distribution_plots(self, dataset_name: str, max_cols: int = 6) -> List[str]:
      
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)
        
        df = self.datasets[dataset_name]
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
        
        if len(numeric_cols) == 0:
            logger.warning(f"No numeric columns found in {dataset_name}")
            return []
        
        logger.info(f"Creating distribution plots for {dataset_name}...")
        
        saved_plots = []

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
     
                df[col].hist(bins=30, alpha=0.7, ax=axes[i], density=True)
                df[col].plot.kde(ax=axes[i], color='red', linewidth=2)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Density')
        
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_file = f"eda_output/plots/{dataset_name}_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        logger.info(f"Distribution plots saved for {dataset_name}")
        return saved_plots

    @timing_decorator
    def create_correlation_heatmap(self, dataset_name: str) -> str:
    
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)

        df = self.datasets[dataset_name]
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            logger.warning(f"Not enough numeric columns for correlation analysis in {dataset_name}")
            return None

        logger.info(f"Creating correlation heatmap for {dataset_name}...")

        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'Correlation Heatmap - {dataset_name}')
        plt.tight_layout()

        plot_file = f"eda_output/plots/{dataset_name}_correlation_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation heatmap saved for {dataset_name}")
        return plot_file

    @timing_decorator
    def create_categorical_plots(self, dataset_name: str, max_cols: int = 4) -> List[str]:
     
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)

        df = self.datasets[dataset_name]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:max_cols]

        if len(categorical_cols) == 0:
            logger.warning(f"No categorical columns found in {dataset_name}")
            return []

        logger.info(f"Creating categorical plots for {dataset_name}...")

        saved_plots = []

        for col in categorical_cols:
            plt.figure(figsize=(12, 6))

            top_categories = df[col].value_counts().head(15)

            ax = top_categories.plot(kind='bar', color='skyblue', alpha=0.8)
            plt.title(f'Distribution of {col} - {dataset_name}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')

            for i, v in enumerate(top_categories.values):
                ax.text(i, v + max(top_categories.values) * 0.01, str(v),
                       ha='center', va='bottom')

            plt.tight_layout()
            plot_file = f"eda_output/plots/{dataset_name}_{col}_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_file)

        logger.info(f"Categorical plots saved for {dataset_name}")
        return saved_plots

    @timing_decorator
    def create_time_series_plots(self, dataset_name: str) -> List[str]:
       
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)

        df = self.datasets[dataset_name]
        datetime_cols = df.select_dtypes(include=['datetime64']).columns

        if len(datetime_cols) == 0:
            logger.warning(f"No datetime columns found in {dataset_name}")
            return []

        logger.info(f"Creating time series plots for {dataset_name}...")

        saved_plots = []

        for col in datetime_cols:
       
            daily_counts = df.groupby(df[col].dt.date).size()

            plt.figure(figsize=(15, 6))
            daily_counts.plot(kind='line', marker='o', alpha=0.7)
            plt.title(f'Daily Activity Over Time - {col} ({dataset_name})')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_file = f"eda_output/plots/{dataset_name}_{col}_timeseries.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_file)

        logger.info(f"Time series plots saved for {dataset_name}")
        return saved_plots

    @timing_decorator
    def create_interactive_dashboard(self, dataset_name: str) -> str:
      
        if dataset_name not in self.datasets:
            self.load_data(dataset_name)

        df = self.datasets[dataset_name]
        logger.info(f"Creating interactive dashboard for {dataset_name}...")

      
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Numeric Distributions', 'Categorical Counts',
                          'Correlation Heatmap', 'Summary Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )

      
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        for i, col in enumerate(numeric_cols):
            fig.add_trace(
                go.Box(y=df[col], name=col, showlegend=False),
                row=1, col=1
            )

    
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            top_cats = df[col].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=top_cats.index, y=top_cats.values, name=col, showlegend=False),
                row=1, col=2
            )

       
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          colorscale='RdBu',
                          showscale=False),
                row=2, col=1
            )

        summary_data = []
        summary_data.append(['Dataset', dataset_name])
        summary_data.append(['Total Rows', f"{len(df):,}"])
        summary_data.append(['Total Columns', len(df.columns)])
        summary_data.append(['Missing Values', f"{df.isnull().sum().sum():,}"])
        summary_data.append(['Memory Usage (MB)', f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"])

        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*summary_data)))
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f'Interactive Dashboard - {dataset_name}',
            height=800,
            showlegend=False
        )

        dashboard_file = f"eda_output/{dataset_name}_dashboard.html"
        fig.write_html(dashboard_file)

        logger.info(f"Interactive dashboard saved for {dataset_name}")
        return dashboard_file

    def run_complete_eda(self, dataset_name: str) -> Dict[str, Any]:
       
        logger.info(f"Running complete EDA for {dataset_name}...")

        results = {
            'dataset_name': dataset_name,
            'statistics': None,
            'plots': {
                'distributions': [],
                'correlation': None,
                'categorical': [],
                'timeseries': [],
                'dashboard': None
            },
            'execution_time': None
        }

        start_time = time.time()

        try:
           
            self.load_data(dataset_name)

            results['statistics'] = self.compute_descriptive_statistics(dataset_name)

            results['plots']['distributions'] = self.create_distribution_plots(dataset_name)
            results['plots']['correlation'] = self.create_correlation_heatmap(dataset_name)
            results['plots']['categorical'] = self.create_categorical_plots(dataset_name)
            results['plots']['timeseries'] = self.create_time_series_plots(dataset_name)
            results['plots']['dashboard'] = self.create_interactive_dashboard(dataset_name)

            end_time = time.time()
            results['execution_time'] = end_time - start_time

            logger.info(f"Complete EDA finished for {dataset_name} in {results['execution_time']:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"EDA failed for {dataset_name}: {str(e)}")
            raise

def main():
  
    try:
        analyzer = EDAAnalyzer()

        datasets = ['airport_traffic', 'flight_info', 'passenger_profiles', 'esop_transactions']

        print("Starting Exploratory Data Analysis...")
        print("="*60)

        all_results = {}

        for dataset in datasets:
            try:
                print(f"\nAnalyzing {dataset}...")
                results = analyzer.run_complete_eda(dataset)
                all_results[dataset] = results

                print(f"✓ {dataset} analysis completed")
                print(f"  Statistics saved to: eda_output/statistics/{dataset}_statistics.json")
                print(f"  Dashboard saved to: {results['plots']['dashboard']}")

            except Exception as e:
                print(f"✗ Failed to analyze {dataset}: {str(e)}")
                logger.error(f"Failed to analyze {dataset}: {str(e)}")

        print("\n" + "="*60)
        print("EDA SUMMARY")
        print("="*60)
        print(f"Total datasets analyzed: {len(all_results)}")

        for dataset, results in all_results.items():
            if results['statistics']:
                stats = results['statistics']['basic_info']
                print(f"\n{dataset}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Memory: {stats['memory_usage_mb']:.2f} MB")
                print(f"  Analysis time: {results['execution_time']:.2f} seconds")

        return analyzer, all_results

    except Exception as e:
        logger.error(f"EDA pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    analyzer, results = main()
