# AeroGuide
AeroGuide is a b2b recommendation engine for businesses at the airport. 

## Project Structure

```
airport/
├── data/                           # Raw CSV datasets
│   ├── airport_traffic.csv         # Airport traffic data (1M rows)
│   ├── flight_info.csv.csv         # Flight information (1M rows)
│   ├── passenger_profiles.csv      # Passenger profiles (1M rows)
│   └── esop_transactions.csv       # ESOP transactions (10K rows)
├── scripts/                        # Processing scripts
│   ├── ingestion.py                # Data ingestion pipeline
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── basic_statistics.py         # Statistical analysis
│   └── eda_statistics.py           # Advanced EDA (with plotting)
├── processed_data/                 # Processed datasets (pickle & CSV)
├── statistics_output/              # Analysis outputs
│   └── reports/                    # Generated reports
├── logs/                          # Pipeline logs
├── airport_data.db                # SQLite database (large datasets)
└── README.md                      # This file
```


### 1. Data Ingestion
```bash
cd airport
python scripts/ingestion.py
```

**What it does:**
- Loads all 4 CSV datasets with validation
- Stores large datasets (1M+ rows) in SQLite for efficient processing
- Keeps small datasets (ESOP) in memory
- Creates optimized indexes for query performance
- Generates comprehensive ingestion logs

**Output:**
- `airport_data.db` - SQLite database with indexed tables
- Ingestion logs and performance metrics. Find airport_data.db by downloading this zip file: https://drive.google.com/file/d/1Y0IhyYPKlDQ3Egz-y-chlB1oF1JTIT95/view?usp=sharing

### 2. Data Preprocessing
```bash
python scripts/preprocessing.py
```

**What it does:**
- Handles missing values with intelligent strategies
- Removes duplicates while preserving data integrity
- Standardizes data types (datetime, categorical, numeric)
- Detects and handles outliers using IQR method
- Creates derived features (time-based, categorical bins, log transforms)
- Saves processed data in both pickle and CSV formats

**Output:**
- `processed_data/` - Clean datasets ready for analysis
- Preprocessing logs with detailed statistics. Find all the processed data here, by downloading the zip file: https://drive.google.com/file/d/1Uo_j5CVM8dCpYL2FtSx6zjrofsHvtp_X/view?usp=sharing

### 3. Statistical Analysis
```bash
python scripts/basic_statistics.py
```

**What it does:**
- Computes comprehensive descriptive statistics
- Analyzes data quality (missing values, duplicates)
- Performs outlier analysis using IQR method
- Generates correlation matrices for numeric variables
- Creates human-readable summary reports
- Analyzes temporal patterns in datetime columns

**Output:**
- `statistics_output/reports/` - JSON statistics and text reports
- Detailed analysis for each dataset

## Dataset Overview

| Dataset | Rows | Columns | Size | Storage Strategy |
|---------|------|---------|------|------------------|
| Airport Traffic | 1,000,000 | 15 | 274MB | SQLite |
| Flight Info | 1,000,000 | 16 | 232MB | SQLite |
| Passenger Profiles | 1,000,000 | 18 | 202MB | SQLite |
| ESOP Transactions | 10,000 | 6 | 3.4MB | In-Memory |

Find all 4 original datasets here: https://drive.google.com/file/d/1465Kl0HaMTjuubeNJGSMzqH3bYNeSJA4/view?usp=sharing

## 🔧 Key Features

### Data Ingestion Pipeline
- **Robust Error Handling**: Graceful handling of malformed data
- **Performance Optimization**: Chunked loading for large datasets
- **Smart Storage**: SQLite for large data, memory for small data
- **Comprehensive Logging**: Detailed execution logs with timing
- **Data Validation**: Quality checks during ingestion

### Preprocessing Pipeline
- **Missing Value Handling**: Multiple strategies (mean, median, mode, forward-fill)
- **Outlier Detection**: IQR-based outlier capping (preserves data)
- **Feature Engineering**: Time-based features, categorical binning, log transforms
- **Type Standardization**: Automatic datetime parsing, categorical encoding
- **Duplicate Removal**: Intelligent duplicate detection and removal

### Statistical Analysis
- **Descriptive Statistics**: Mean, median, std dev, skewness, kurtosis
- **Data Quality Metrics**: Missing value analysis, duplicate detection
- **Correlation Analysis**: Pearson correlation matrices
- **Outlier Analysis**: IQR-based outlier detection and quantification
- **Temporal Analysis**: Time-based pattern detection

## Generated Features

### Airport Traffic Dataset
- `hour`, `day_of_week`, `month` - Temporal components
- `is_weekend` - Weekend indicator
- `time_period` - Time of day categories (Morning, Afternoon, Evening, Night)
- `passenger_count_log` - Log-transformed passenger count

### Flight Info Dataset
- `actual_delay` - Calculated delay in minutes
- `is_delayed` - Delay indicator (>15 minutes)
- `scheduled_hour`, `scheduled_day_of_week` - Temporal components
- `is_weekend_flight` - Weekend flight indicator
- `passenger_count_log` - Log-transformed passenger count

### Passenger Profiles Dataset
- `age_group` - Age categories (Young, Adult, Middle-aged, Senior, Elderly)
- `spend_category` - Spending categories (Low, Medium, High, Premium)
- `dwell_time_category` - Dwell time categories
- `average_spend_log` - Log-transformed spending

### ESOP Transactions Dataset
- `hour`, `day_of_week`, `month` - Temporal components
- `is_weekend` - Weekend indicator
- `amount_category` - Transaction amount categories
- `net_amount_log` - Log-transformed amount

## Design Decisions & Tradeoffs

### Storage Strategy
- **Large Datasets (1M+ rows)**: SQLite storage for memory efficiency and incremental processing
- **Small Datasets (<50K rows)**: In-memory processing for speed
- **Rationale**: Balances memory usage with processing speed

### Outlier Handling
- **Method**: IQR-based capping instead of removal
- **Rationale**: Preserves data volume while reducing extreme value impact
- **Threshold**: 1.5 × IQR (standard statistical practice)

### Missing Value Strategy
- **Numeric**: Median (robust to outliers)
- **Categorical**: Mode or "Unknown" based on cardinality
- **Datetime**: Forward fill (temporal continuity)

### Feature Engineering
- **Time-based Features**: Extract hour, day, month for temporal analysis
- **Log Transforms**: Reduce skewness in count/amount variables
- **Categorical Binning**: Create meaningful groups for continuous variables

## Sample Analysis Results

### Airport Traffic Dataset
- **Shape**: 1,000,000 rows × 15 columns
- **Missing Values**: 41,532 (4.15%) - only in derived time_period feature
- **Key Insights**: 
  - Passenger count ranges 1-159, mean 49.13
  - No outliers after preprocessing
  - Balanced temporal distribution

### Flight Info Dataset
- **Shape**: 1,000,000 rows × 16 columns
- **Missing Values**: 0 (0.00%)
- **Key Insights**:
  - 13.10% outliers in passenger count (handled)
  - 2.72% outliers in delay minutes (handled)
  - Strong correlation between scheduled and actual times

### Passenger Profiles Dataset
- **Shape**: 1,000,000 rows × 18 columns
- **Missing Values**: 655,779 (65.58%) - primarily in purchase_categories
- **Key Insights**:
  - 12.43% outliers in average spend (handled)
  - High cardinality in categorical variables
  - Clear spending patterns by passenger segment


## Logs and Monitoring

All pipeline operations are logged with:
- Execution timing for performance monitoring
- Data quality metrics for each step
- Error handling with detailed stack traces
- Memory usage tracking for optimization

Log files are stored in the `logs/` directory:
- `ingestion.log` - Data loading operations
- `preprocessing.log` - Data cleaning operations  
- `statistics.log` - Analysis operations

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **sqlite3**: Database operations
- **matplotlib/seaborn**: Visualization (optional)
- **plotly**: Interactive visualizations (optional)


