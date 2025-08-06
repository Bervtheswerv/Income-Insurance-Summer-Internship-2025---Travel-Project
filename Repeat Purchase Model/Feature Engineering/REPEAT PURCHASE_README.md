# README - Repeat Purchase Pipeline

For each customer's first purchase within a period, the script creates features by looking backwards (historical) and forwards (future) within a 1-year window. This pipeline analyses customer repeat purchase patterns across two defined periods.

- **Period 1**: June 2022 - May 2023  
- **Period 2**: June 2023 - May 2024

It handles temporal relationships by creating lookback features (pre-purchase behaviour) and lookahead features (post-purchase outcomes). Finally, it consolidates these into period-neutral features that will be used for modelling. The end goal is producing a customer-level dataset where each row represents a customer's first purchase in either period, enriched with their historical behaviour patterns and subsequent purchase outcomes. It also includes the raw features from the original input dataset.

## 1. Usage

```python
from optimised_pipeline import memory_efficient_repeat_purchase_pipeline

# Process your data (file_path names can be changed)
results = memory_efficient_repeat_purchase_pipeline('travelpolicies_data.csv')
results.to_csv('repeat_purchase_features.csv', index=False)
```

## 2. Feature Output

The pipeline creates these exact features for model training:

### A) Target Variable
- `repeat_purchase_final` - Boolean flag (1 if customer purchased again within 1 year, 0 if otherwise)

### B) Categorical Features
**Pre-period (Historical):**
- `Region_[ASEAN/ASIA/WORLDWIDE]_pre_final` - Geographic regions
- `plantype_fgi_[Individual/Group/Family]_pre_final` - Plan types
- `Coverage_final_[Classic/Deluxe/Preferred]_pre_final` - Coverage levels

**Post-period (Future):**
- `Region_[ASEAN/ASIA/WORLDWIDE]_post_final` - Geographic regions  
- `plantype_fgi_[Individual/Group/Family]_post_final` - Plan types
- `Coverage_final_[Classic/Deluxe/Preferred]_post_final` - Coverage levels

### C) Numerical Features (Averaged)
**Pre-period:**
- `Claim_count_avg_pre_final` - Average claims per policy
- `Discounts_avg_pre_final` - Average discount amount
- `Advance_purchase_avg_pre_final` - Days purchased in advance
- `trip_duration_avg_pre_final` - Average trip length
- `Insured_Premium_avg_pre_final` - Average premium amount
- `Tenure_Years_avg_pre_final` - Customer tenure
- `claim_close_days_avg_pre_final` - Days to close claims

**Post-period:**
- `Claim_count_avg_post_final` - Average claims per policy
- `Discounts_avg_post_final` - Average discount amount
- `Advance_purchase_avg_post_final` - Days purchased in advance
- `trip_duration_avg_post_final` - Average trip length
- `Insured_Premium_avg_post_final` - Average premium amount
- `Tenure_Years_avg_post_final` - Customer tenure
- `claim_close_days_avg_post_final` - Days to close claims

### D) Behavioural Features
- `policies_pre_final` - Number of policies before first purchase
- `policies_post_final` - Number of policies after first purchase
- `family_flag_pre_final` - Boolean flag for family coverage (pre-period)
- `family_flag_post_final` - Boolean flag for family coverage (post-period)

### E) Identifier Columns
- `InsuredNric` - Customer identifier
- `analysis_period` - Whether the first purchase was made in Period1 or Period2

## 3. Configuration

To modify the pipeline, edit these constants:

```python
# Analysis periods
PERIOD1_START = pd.to_datetime('2022-06-01')
PERIOD1_END = pd.to_datetime('2023-05-30')
PERIOD2_START = pd.to_datetime('2023-06-01') 
PERIOD2_END = pd.to_datetime('2024-05-30')

# Feature windows
LOOKBACK_DAYS = 366
LOOKAHEAD_DAYS = 366

# Feature columns
CATEGORICAL_COLUMNS = {'Region': 'Region', 'plantype_fgi': 'plantype_fgi', 'Coverage_final': 'Coverage_final'}
NUMERICAL_COLUMNS = ['Claim_count', 'Discounts', 'Advance_purchase', 'trip_duration', 'Insured_Premium', 'Tenure_Years', 'claim_close_days']

# Main runner = file input/output path
if __name__ == '__main__':
    input_file = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Input_Data/Actual/travelpolicies_2013_2025may.csv'
    output_file = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Output_Data/repeat_purchase_optimised.csv'
```

## 4. Key Modification Points

1. **Date Periods**: Change `PERIOD1_START`, `PERIOD2_START`, etc. constants
2. **Feature Columns**: Modify `CATEGORICAL_COLUMNS` and `NUMERICAL_COLUMNS` dictionaries
3. **Feature Creation**: Update `create_period_features_direct()` function
4. **Analysis Window**: Adjust `LOOKBACK_DAYS` and `LOOKAHEAD_DAYS` constants

---

# READMETOO - Technical Documentation

## 1. Script Architecture & Function Flow

```
├── Input CSV Data
├── repeat_purchase_pipeline() → Main script runner, data flow from here
├── calculate_first_purchase_dates()  → df with first purchase dates
├── calculate_repeat_purchase_flags() → df with repeat purchase indicators
├── create_period_features_direct() → [Period 1 & 2] customer features + raw data
└── Final Combined Dataset (Raw + Engineered Features)
```

## 2. Core Functions

### `repeat_purchase_pipeline()`
**Purpose**: Main orchestrator that controls the entire feature engineering process  
**Inputs**: CSV file path, customer ID column, purchase date column  
**Returns**: Final dataset with raw data + engineered features  
**Role**: Coordinates all processing steps, loads data, manages the two-period analysis, and combines results  
**Key Logic**: Processes periods separately to minimize memory usage, preserves original data structure  

### `calculate_first_purchase_dates()`
**Purpose**: Identifies each customer's first purchase within defined time periods  
**Inputs**: Customer transaction data with dates  
**Returns**: Original data + two new columns with first purchase dates per period  
**Role**: Foundation step that establishes customer analysis windows  
**Key Logic**: Groups transactions by customer, finds minimum date within Period 1 (Jun 2022-May 2023) and Period 2 (Jun 2023-May 2024)  

### `calculate_repeat_purchase_flags()`
**Purpose**: Determines if customers made repeat purchases within 366 days  
**Inputs**: Data with first purchase dates  
**Returns**: Data + binary flags indicating repeat purchase behavior  
**Role**: Creates the target variable for predictive modeling  
**Key Logic**: Compares each customer's next purchase date against their first purchase, flags as repeat if within time window  

### `create_period_features_direct()`
**Purpose**: Generates behavioral features by analyzing customer activity before and after first purchase  
**Inputs**: Full dataset, period number (1 or 2)  
**Returns**: Customer records with original data + engineered features  
**Role**: Core feature engineering engine that creates model-ready variables  

**Key Logic**: 
- **Lookback Analysis**: Examines 366 days before first purchase (historical behavior)
- **Lookahead Analysis**: Examines 366 days after first purchase (future behavior)
- **Feature Creation**: Transforms raw transactions into aggregated patterns

## 3. Feature Engineering Process

### Data Transformation Steps
1. **Customer Identification**: Extract customers with first purchases in each period
2. **Time Window Creation**: Define lookback (historical) and lookahead (future) periods
3. **Data Filtering**: Separate transactions within analysis windows
4. **Feature Aggregation**: Convert individual transactions into customer-level patterns

### Feature Types Generated

**Categorical Features** (One-hot encoding: converts text categories into boolean columns)
- **Pre-period**: `Region_Asia_pre_final`, `plantype_fgi_Basic_pre_final` (historical preferences)
- **Post-period**: `Region_Asia_post_final`, `plantype_fgi_Basic_post_final` (future patterns)

**Numerical Features** (Averaging: calculates mean values across multiple transactions)
- **Pre-period**: `Claim_count_avg_pre_final`, `trip_duration_avg_pre_final` (historical averages)
- **Post-period**: `Claim_count_avg_post_final`, `trip_duration_avg_post_final` (future averages)

**Behavioral Features** (Counting and flagging: identifies specific customer behaviors)
- **Policy Counts**: `policies_pre_final`, `policies_post_final` (transaction frequency)
- **Family Indicators**: `family_flag_pre_final`, `family_flag_post_final` (coverage type patterns)

### Memory Efficiency Strategy
- **Direct Feature Creation**: Creates final features immediately without storing intermediate calculations
- **Period Separation**: Processes Period 1 and Period 2 independently to reduce peak memory usage
- **Selective Processing**: Only analyzes customers relevant to each period rather than entire dataset