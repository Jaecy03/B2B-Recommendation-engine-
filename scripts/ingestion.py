import pandas as pd
import sqlite3
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
import os
import time
from functools import wraps
import pickle
from datetime import datetime

def setup_logging() -> logging.Logger:
  
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ingestion.log'),
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

os.makedirs('logs', exist_ok=True)
logger = setup_logging()

class DataIngestionPipeline:
  
    def __init__(self, data_folder: str = "data", db_path: str = "airport_data.db"):
     
        self.data_folder = Path(data_folder)
        self.db_path = db_path
        self.dataframes = {}

        self.files = {
            "airport_traffic": "airport_traffic.csv",
            "esop_transactions": "esop_transactions.csv",
            "flight_info": "flight_info.csv.csv", 
            "passenger_profiles": "passenger_profiles.csv"
        }

        self.large_dataset_threshold = 50000 

    @timing_decorator
    def load_csv_to_df(self, file_name: str, dataset_name: str) -> pd.DataFrame:
        
        file_path = self.data_folder / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading {dataset_name} from {file_path}...")

        try:
            
            df = pd.read_csv(
                file_path,
                encoding="utf-8",
                sep=",",
                comment='#', 
                on_bad_lines='skip',
                low_memory=False
            )

            if df.empty:
                raise pd.errors.EmptyDataError(f"No data found in {file_name}")

            logger.info(f"Successfully loaded {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")


            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            logger.info(f"Memory usage for {dataset_name}: {memory_usage:.2f} MB")

            return df

        except Exception as e:
            logger.error(f"Error loading {file_name}: {str(e)}")
            raise

    @timing_decorator
    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, any]:
      
        logger.info(f"Validating data quality for {dataset_name}...")

        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }

        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            logger.warning(f"{dataset_name}: Found {total_missing} missing values")

        if quality_metrics['duplicate_rows'] > 0:
            logger.warning(f"{dataset_name}: Found {quality_metrics['duplicate_rows']} duplicate rows")

        logger.info(f"Data quality validation completed for {dataset_name}")
        return quality_metrics

    @timing_decorator
    def create_sqlite_db(self, dataframes: Dict[str, pd.DataFrame]) -> None:
    
        logger.info(f"Creating SQLite database at {self.db_path}...")

        conn = sqlite3.connect(self.db_path)
        try:
            for name, df in dataframes.items():
               
                if len(df) >= self.large_dataset_threshold:
                    logger.info(f"Storing {name} in SQLite (large dataset: {len(df)} rows)")
                    df.to_sql(name, conn, if_exists='replace', index=False, chunksize=10000)
                    logger.info(f"Successfully stored {name} with {len(df)} rows")
                else:
                    logger.info(f"Skipping {name} for SQLite (small dataset: {len(df)} rows - keeping in memory)")

            self._create_database_indexes(conn)

        except Exception as e:
            logger.error(f"Error creating SQLite database: {str(e)}")
            raise
        finally:
            conn.close()

        logger.info("SQLite database creation completed")

    def _create_database_indexes(self, conn: sqlite3.Connection) -> None:
      
        cursor = conn.cursor()
        index_commands = {
            "airport_traffic": [
                ("flight_id", "idx_traffic_flight_id"),
                ("airport", "idx_traffic_airport"),
                ("timestamp", "idx_traffic_timestamp")
            ],
            "passenger_profiles": [
                ("passenger_id", "idx_passenger_id"),
                ("passenger_segment", "idx_passenger_segment"),
                ("class_of_travel", "idx_class_travel")
            ],
            "flight_info": [
                ("flight_id", "idx_flight_id"),
                ("airline", "idx_airline"),
                ("origin", "idx_origin"),
                ("destination", "idx_destination")
            ]
        }

        for table, indexes in index_commands.items():
            try:
          
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    continue

                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]

                for col, idx_name in indexes:
                    if col in columns:
                        cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col})")
                        logger.info(f"Created index {idx_name} on {table}({col})")
                    else:
                        logger.warning(f"Column '{col}' not found in table '{table}', skipping index")

            except Exception as e:
                logger.error(f"Error creating indexes for {table}: {e}")

        conn.commit()
        logger.info("Database indexing completed")

    @timing_decorator
    def save_processed_data(self, df: pd.DataFrame, dataset_name: str,
                          format_type: str = 'pickle') -> str:
        
        os.makedirs('processed_data', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == 'pickle':
            file_path = f"processed_data/{dataset_name}_processed_{timestamp}.pkl"
            df.to_pickle(file_path)
        elif format_type == 'csv':
            file_path = f"processed_data/{dataset_name}_processed_{timestamp}.csv"
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"Saved processed {dataset_name} to {file_path}")
        return file_path

    @timing_decorator
    def run_ingestion_pipeline(self) -> Dict[str, any]:
       
        logger.info("Starting data ingestion pipeline...")

        pipeline_results = {
            'loaded_datasets': {},
            'quality_metrics': {},
            'storage_decisions': {},
            'execution_time': None
        }

        start_time = time.time()

        try:
     
            for dataset_name, file_name in self.files.items():
                try:
                    df = self.load_csv_to_df(file_name, dataset_name)
                    self.dataframes[dataset_name] = df

                    quality_metrics = self.validate_data_quality(df, dataset_name)
                    pipeline_results['quality_metrics'][dataset_name] = quality_metrics

                    is_large = len(df) >= self.large_dataset_threshold
                    storage_strategy = 'SQLite' if is_large else 'In-Memory'
                    pipeline_results['storage_decisions'][dataset_name] = {
                        'rows': len(df),
                        'strategy': storage_strategy,
                        'size_mb': df.memory_usage(deep=True).sum() / 1024**2
                    }

                    pipeline_results['loaded_datasets'][dataset_name] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'file_name': file_name
                    }

                except Exception as e:
                    logger.error(f"Failed to load {dataset_name}: {str(e)}")
                    pipeline_results['loaded_datasets'][dataset_name] = {'error': str(e)}

            large_datasets = {name: df for name, df in self.dataframes.items()
                            if len(df) >= self.large_dataset_threshold}

            if large_datasets:
                self.create_sqlite_db(large_datasets)

            end_time = time.time()
            pipeline_results['execution_time'] = end_time - start_time

            logger.info(f"Data ingestion pipeline completed successfully in {pipeline_results['execution_time']:.2f} seconds")

            self._print_pipeline_summary(pipeline_results)

            return pipeline_results

        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise

    def _print_pipeline_summary(self, results: Dict[str, any]) -> None:
      
        print("\n" + "="*60)
        print("DATA INGESTION PIPELINE SUMMARY")
        print("="*60)

        print(f"Total execution time: {results['execution_time']:.2f} seconds")
        print(f"Datasets processed: {len(results['loaded_datasets'])}")

        print("\nDataset Overview:")
        for dataset, info in results['loaded_datasets'].items():
            if 'error' not in info:
                storage = results['storage_decisions'][dataset]['strategy']
                size_mb = results['storage_decisions'][dataset]['size_mb']
                print(f"  • {dataset}: {info['rows']:,} rows, {info['columns']} columns, "
                      f"{size_mb:.1f}MB ({storage})")
            else:
                print(f"  • {dataset}: ERROR - {info['error']}")

        print("\nStorage Strategy:")
        for dataset, decision in results['storage_decisions'].items():
            print(f"  • {dataset}: {decision['strategy']} ({decision['rows']:,} rows)")

        print("="*60)

def main():
 
    try:
        pipeline = DataIngestionPipeline()
        results = pipeline.run_ingestion_pipeline()

        return pipeline, results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    pipeline, results = main()

