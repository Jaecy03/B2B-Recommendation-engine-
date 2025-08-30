import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import re
import pickle
from datetime import datetime, timedelta
import time
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

def setup_logging() -> logging.Logger:
 
    import os
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocessing.log'),
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

class DataPreprocessor:
 
    def __init__(self, db_path: str = "airport_data.db"):
    
        self.db_path = db_path
        self.processed_data = {}
        self.preprocessing_stats = {}
        
    def load_data_from_source(self, dataset_name: str, source: str = 'auto') -> pd.DataFrame:
     
        logger.info(f"Loading {dataset_name} from {source} source...")
        
        if source == 'auto':
       
            try:
                return self._load_from_sqlite(dataset_name)
            except:
                return self._load_from_pickle(dataset_name)
        elif source == 'sqlite':
            return self._load_from_sqlite(dataset_name)
        elif source == 'pickle':
            return self._load_from_pickle(dataset_name)
        else:
            raise ValueError(f"Unsupported source type: {source}")
    
    def _load_from_sqlite(self, dataset_name: str) -> pd.DataFrame:
   
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {dataset_name}", conn)
            logger.info(f"Loaded {dataset_name} from SQLite: {len(df)} rows")
            return df
        finally:
            conn.close()
    
    def _load_from_pickle(self, dataset_name: str) -> pd.DataFrame:
        """Load data from pickle files."""
        pickle_files = list(Path('processed_data').glob(f"{dataset_name}_*.pkl"))
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found for {dataset_name}")
       
        latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_pickle(latest_file)
        logger.info(f"Loaded {dataset_name} from pickle: {len(df)} rows")
        return df

    @timing_decorator
    def clean_missing_values(self, df: pd.DataFrame, dataset_name: str, 
                           strategy: Dict[str, str] = None) -> pd.DataFrame:
    
        logger.info(f"Cleaning missing values for {dataset_name}...")
        
        df_cleaned = df.copy()
        initial_missing = df_cleaned.isnull().sum().sum()
        
        if initial_missing == 0:
            logger.info(f"No missing values found in {dataset_name}")
            return df_cleaned
        
        logger.info(f"Found {initial_missing} missing values in {dataset_name}")
    
        if strategy is None:
            strategy = self._get_default_missing_value_strategy(df_cleaned, dataset_name)
        
        for column, method in strategy.items():
            if column in df_cleaned.columns and df_cleaned[column].isnull().any():
                missing_count = df_cleaned[column].isnull().sum()
                
                if method == 'drop':
                    df_cleaned = df_cleaned.dropna(subset=[column])
                elif method == 'forward_fill':
                    df_cleaned[column] = df_cleaned[column].fillna(method='ffill')
                elif method == 'backward_fill':
                    df_cleaned[column] = df_cleaned[column].fillna(method='bfill')
                elif method == 'mean':
                    if df_cleaned[column].dtype in ['int64', 'float64']:
                        df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())
                elif method == 'median':
                    if df_cleaned[column].dtype in ['int64', 'float64']:
                        df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                elif method == 'mode':
                    mode_value = df_cleaned[column].mode()
                    if len(mode_value) > 0:
                        df_cleaned[column] = df_cleaned[column].fillna(mode_value[0])
                elif method == 'zero':
                    df_cleaned[column] = df_cleaned[column].fillna(0)
                elif method == 'unknown':
                    df_cleaned[column] = df_cleaned[column].fillna('Unknown')
                
                logger.info(f"Handled {missing_count} missing values in {column} using {method}")
        
        final_missing = df_cleaned.isnull().sum().sum()
        logger.info(f"Missing values reduced from {initial_missing} to {final_missing}")
        
        return df_cleaned
    
    def _get_default_missing_value_strategy(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, str]:
      
        strategy = {}
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
             
                strategy[column] = 'median'
            elif df[column].dtype == 'object':
           
                if df[column].nunique() < len(df) * 0.5:
                    strategy[column] = 'mode'
                else:
                    strategy[column] = 'unknown'
            elif 'datetime' in str(df[column].dtype):
            
                strategy[column] = 'forward_fill'
            else:
   
                strategy[column] = 'mode'
        
        return strategy

    @timing_decorator
    def remove_duplicates(self, df: pd.DataFrame, dataset_name: str, 
                         subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
        
        logger.info(f"Removing duplicates from {dataset_name}...")
        
        initial_rows = len(df)
        duplicate_count = df.duplicated(subset=subset, keep=False).sum()
        
        if duplicate_count == 0:
            logger.info(f"No duplicates found in {dataset_name}")
            return df
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_cleaned)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"Removed {removed_rows} duplicate rows from {dataset_name} "
                   f"({initial_rows} -> {final_rows} rows)")
        
        return df_cleaned

    @timing_decorator
    def standardize_data_types(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
     
        logger.info(f"Standardizing data types for {dataset_name}...")
        
        df_standardized = df.copy()
      
        if dataset_name == 'airport_traffic':
            df_standardized = self._standardize_airport_traffic_types(df_standardized)
        elif dataset_name == 'flight_info':
            df_standardized = self._standardize_flight_info_types(df_standardized)
        elif dataset_name == 'passenger_profiles':
            df_standardized = self._standardize_passenger_profiles_types(df_standardized)
        elif dataset_name == 'esop_transactions':
            df_standardized = self._standardize_esop_transactions_types(df_standardized)
        
        logger.info(f"Data type standardization completed for {dataset_name}")
        return df_standardized

    def _standardize_airport_traffic_types(self, df: pd.DataFrame) -> pd.DataFrame:
       
        try:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'passenger_count' in df.columns:
                df['passenger_count'] = pd.to_numeric(df['passenger_count'], errors='coerce')
            if 'terminal_id' in df.columns:
                df['terminal_id'] = df['terminal_id'].astype('category')
            if 'passenger_type' in df.columns:
                df['passenger_type'] = df['passenger_type'].astype('category')
            if 'flight_type' in df.columns:
                df['flight_type'] = df['flight_type'].astype('category')
        except Exception as e:
            logger.warning(f"Error in airport traffic type conversion: {e}")
        return df

    def _standardize_flight_info_types(self, df: pd.DataFrame) -> pd.DataFrame:
   
        try:
            datetime_cols = ['scheduled_time', 'actual_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            numeric_cols = ['passenger_count', 'delay_minutes']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            categorical_cols = ['airline', 'flight_type', 'aircraft_type']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        except Exception as e:
            logger.warning(f"Error in flight info type conversion: {e}")
        return df

    def _standardize_passenger_profiles_types(self, df: pd.DataFrame) -> pd.DataFrame:
     
        try:
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
            if 'dwell_time' in df.columns:
                df['dwell_time'] = pd.to_numeric(df['dwell_time'], errors='coerce')
            if 'average_spend' in df.columns:
                df['average_spend'] = pd.to_numeric(df['average_spend'], errors='coerce')
            if 'num_categories_visited' in df.columns:
                df['num_categories_visited'] = pd.to_numeric(df['num_categories_visited'], errors='coerce')

            categorical_cols = ['gender', 'nationality', 'marital_status', 'passenger_segment',
                              'class_of_travel', 'income_bracket']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')

            if 'loyalty_program_member' in df.columns:
                df['loyalty_program_member'] = df['loyalty_program_member'].astype('bool')
            if 'made_purchase' in df.columns:
                df['made_purchase'] = df['made_purchase'].astype('bool')
        except Exception as e:
            logger.warning(f"Error in passenger profiles type conversion: {e}")
        return df

    def _standardize_esop_transactions_types(self, df: pd.DataFrame) -> pd.DataFrame:
   
        try:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if 'net_amount' in df.columns:
                df['net_amount'] = pd.to_numeric(df['net_amount'], errors='coerce')

            categorical_cols = ['store_name', 'store_category', 'location']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        except Exception as e:
            logger.warning(f"Error in ESOP transactions type conversion: {e}")
        return df

    @timing_decorator
    def detect_and_handle_outliers(self, df: pd.DataFrame, dataset_name: str,
                                 method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
       
        logger.info(f"Detecting and handling outliers in {dataset_name} using {method} method...")

        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns

        outlier_stats = {}

        for column in numeric_columns:
            if df_cleaned[column].isnull().all():
                continue

            initial_count = len(df_cleaned)

            if method == 'iqr':
                Q1 = df_cleaned[column].quantile(0.25)
                Q3 = df_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers_mask = (df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df_cleaned[column] - df_cleaned[column].mean()) / df_cleaned[column].std())
                outliers_mask = z_scores > threshold

            outlier_count = outliers_mask.sum()

            if outlier_count > 0:
         
                if method == 'iqr':
                    df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound
                    df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound
                elif method == 'zscore':
                 
                    mean_val = df_cleaned[column].mean()
                    std_val = df_cleaned[column].std()
                    lower_cap = mean_val - threshold * std_val
                    upper_cap = mean_val + threshold * std_val
                    df_cleaned.loc[df_cleaned[column] < lower_cap, column] = lower_cap
                    df_cleaned.loc[df_cleaned[column] > upper_cap, column] = upper_cap

                outlier_stats[column] = {
                    'outliers_detected': outlier_count,
                    'percentage': (outlier_count / initial_count) * 100
                }

                logger.info(f"Handled {outlier_count} outliers in {column} "
                           f"({outlier_stats[column]['percentage']:.2f}%)")

        if outlier_stats:
            logger.info(f"Outlier handling completed for {dataset_name}")
        else:
            logger.info(f"No outliers detected in {dataset_name}")

        return df_cleaned

    @timing_decorator
    def create_derived_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
     
        logger.info(f"Creating derived features for {dataset_name}...")

        df_enhanced = df.copy()

        if dataset_name == 'airport_traffic':
            df_enhanced = self._create_airport_traffic_features(df_enhanced)
        elif dataset_name == 'flight_info':
            df_enhanced = self._create_flight_info_features(df_enhanced)
        elif dataset_name == 'passenger_profiles':
            df_enhanced = self._create_passenger_profile_features(df_enhanced)
        elif dataset_name == 'esop_transactions':
            df_enhanced = self._create_esop_transaction_features(df_enhanced)

        new_features = len(df_enhanced.columns) - len(df.columns)
        logger.info(f"Created {new_features} derived features for {dataset_name}")

        return df_enhanced

    def _create_airport_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
   
        try:
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
                df['is_weekend'] = df['day_of_week'].isin([5, 6])
                df['time_period'] = pd.cut(df['hour'],
                                         bins=[0, 6, 12, 18, 24],
                                         labels=['Night', 'Morning', 'Afternoon', 'Evening'])

            if 'passenger_count' in df.columns:
                df['passenger_count_log'] = np.log1p(df['passenger_count'])
        except Exception as e:
            logger.warning(f"Error creating airport traffic features: {e}")
        return df

    def _create_flight_info_features(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            if 'scheduled_time' in df.columns and 'actual_time' in df.columns:
                df['actual_delay'] = (df['actual_time'] - df['scheduled_time']).dt.total_seconds() / 60
                df['is_delayed'] = df['actual_delay'] > 15  

            if 'scheduled_time' in df.columns:
                df['scheduled_hour'] = df['scheduled_time'].dt.hour
                df['scheduled_day_of_week'] = df['scheduled_time'].dt.dayofweek
                df['is_weekend_flight'] = df['scheduled_day_of_week'].isin([5, 6])

            if 'passenger_count' in df.columns:
                df['passenger_count_log'] = np.log1p(df['passenger_count'])
        except Exception as e:
            logger.warning(f"Error creating flight info features: {e}")
        return df

    def _create_passenger_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
 
        try:
            if 'age' in df.columns:
                df['age_group'] = pd.cut(df['age'],
                                       bins=[0, 25, 35, 50, 65, 100],
                                       labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly'])

            if 'average_spend' in df.columns:
                df['spend_category'] = pd.cut(df['average_spend'],
                                            bins=[0, 50, 200, 500, float('inf')],
                                            labels=['Low', 'Medium', 'High', 'Premium'])
                df['average_spend_log'] = np.log1p(df['average_spend'])

            if 'dwell_time' in df.columns:
                df['dwell_time_category'] = pd.cut(df['dwell_time'],
                                                 bins=[0, 30, 60, 120, float('inf')],
                                                 labels=['Short', 'Medium', 'Long', 'Extended'])
        except Exception as e:
            logger.warning(f"Error creating passenger profile features: {e}")
        return df

    def _create_esop_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            if 'date' in df.columns:
                df['hour'] = df['date'].dt.hour
                df['day_of_week'] = df['date'].dt.dayofweek
                df['month'] = df['date'].dt.month
                df['is_weekend'] = df['day_of_week'].isin([5, 6])

            if 'net_amount' in df.columns:
                df['amount_category'] = pd.cut(df['net_amount'],
                                             bins=[0, 100, 300, 500, float('inf')],
                                             labels=['Low', 'Medium', 'High', 'Premium'])
                df['net_amount_log'] = np.log1p(df['net_amount'])
        except Exception as e:
            logger.warning(f"Error creating ESOP transaction features: {e}")
        return df

    @timing_decorator
    def run_preprocessing_pipeline(self, dataset_name: str,
                                 custom_config: Dict[str, Any] = None) -> pd.DataFrame:
      
        logger.info(f"Starting preprocessing pipeline for {dataset_name}...")

        df = self.load_data_from_source(dataset_name)
        initial_shape = df.shape

        config = {
            'handle_missing': True,
            'remove_duplicates': True,
            'standardize_types': True,
            'handle_outliers': True,
            'create_features': True,
            'missing_value_strategy': None,
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5
        }

        if custom_config:
            config.update(custom_config)

        preprocessing_steps = []

        try:
 
            if config['handle_missing']:
                df = self.clean_missing_values(df, dataset_name, config['missing_value_strategy'])
                preprocessing_steps.append('missing_values_handled')

            if config['remove_duplicates']:
                df = self.remove_duplicates(df, dataset_name)
                preprocessing_steps.append('duplicates_removed')

            if config['standardize_types']:
                df = self.standardize_data_types(df, dataset_name)
                preprocessing_steps.append('types_standardized')

            if config['handle_outliers']:
                df = self.detect_and_handle_outliers(df, dataset_name,
                                                   config['outlier_method'],
                                                   config['outlier_threshold'])
                preprocessing_steps.append('outliers_handled')

            if config['create_features']:
                df = self.create_derived_features(df, dataset_name)
                preprocessing_steps.append('features_created')

            final_shape = df.shape

            self.preprocessing_stats[dataset_name] = {
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'steps_completed': preprocessing_steps,
                'rows_change': final_shape[0] - initial_shape[0],
                'columns_change': final_shape[1] - initial_shape[1]
            }

            self.processed_data[dataset_name] = df

            logger.info(f"Preprocessing completed for {dataset_name}: "
                       f"{initial_shape} -> {final_shape}")

            return df

        except Exception as e:
            logger.error(f"Preprocessing failed for {dataset_name}: {str(e)}")
            raise

    def save_processed_data(self, dataset_name: str, format_type: str = 'pickle') -> str:
   
        if dataset_name not in self.processed_data:
            raise ValueError(f"No processed data found for {dataset_name}")

        import os
        os.makedirs('processed_data', exist_ok=True)

        df = self.processed_data[dataset_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == 'pickle':
            file_path = f"processed_data/{dataset_name}_preprocessed_{timestamp}.pkl"
            df.to_pickle(file_path)
        elif format_type == 'csv':
            file_path = f"processed_data/{dataset_name}_preprocessed_{timestamp}.csv"
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"Saved preprocessed {dataset_name} to {file_path}")
        return file_path

    def get_preprocessing_summary(self) -> Dict[str, Any]:
      
        return {
            'datasets_processed': list(self.preprocessing_stats.keys()),
            'statistics': self.preprocessing_stats,
            'total_datasets': len(self.preprocessing_stats)
        }

def main():
   
    try:
        preprocessor = DataPreprocessor()

        datasets = ['airport_traffic', 'flight_info', 'passenger_profiles', 'esop_transactions']

        print("Starting data preprocessing pipeline...")
        print("="*60)

        for dataset in datasets:
            try:
                print(f"\nProcessing {dataset}...")
                df = preprocessor.run_preprocessing_pipeline(dataset)

                pickle_path = preprocessor.save_processed_data(dataset, 'pickle')
                csv_path = preprocessor.save_processed_data(dataset, 'csv')

                print(f"✓ {dataset} processed successfully")
                print(f"  Saved to: {pickle_path}")
                print(f"  CSV backup: {csv_path}")

            except Exception as e:
                print(f"✗ Failed to process {dataset}: {str(e)}")
                logger.error(f"Failed to process {dataset}: {str(e)}")

        summary = preprocessor.get_preprocessing_summary()
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total datasets processed: {summary['total_datasets']}")

        for dataset, stats in summary['statistics'].items():
            print(f"\n{dataset}:")
            print(f"  Shape change: {stats['initial_shape']} -> {stats['final_shape']}")
            print(f"  Steps: {', '.join(stats['steps_completed'])}")

        return preprocessor

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    preprocessor = main()
