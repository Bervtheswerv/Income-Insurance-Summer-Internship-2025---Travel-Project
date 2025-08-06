"""Repeat Purchase Pipeline
This script processes travel insurance policy data to create features for a repeat purchase model.
It calculates first purchase dates, repeat purchase flags, and various features for two defined periods.
It is designed to be memory-efficient, processing data in chunks and avoiding large intermediate storage.
It includes detailed logging and error handling to ensure robustness.
The final output is a DataFrame with engineered features ready for modeling.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PERIOD1_START = pd.to_datetime('2022-06-01')
PERIOD1_END = pd.to_datetime('2023-05-30')
PERIOD2_START = pd.to_datetime('2023-06-01')
PERIOD2_END = pd.to_datetime('2024-05-30')

LOOKBACK_DAYS = 366
LOOKAHEAD_DAYS = 366

CATEGORICAL_COLUMNS = {'Region': 'Region', 'plantype_fgi': 'plantype_fgi', 'Coverage_final': 'Coverage_final'}
NUMERICAL_COLUMNS = ['Claim_count', 'Discounts', 'Advance_purchase', 'trip_duration', 'Insured_Premium', 'Tenure_Years', 'claim_close_days']

#______________________________________________________________________________________________________________________________________
def clean_column_name(name):
    """Clean column names for consistent formatting"""
    return str(name).replace(' ', '_').replace('/', '_').replace('-', '_').replace('(', '').replace(')', '').replace('.', '_')

#______________________________________________________________________________________________________________________________________
def calculate_first_purchase_dates(df, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """Calculate first purchase date within each defined period for each customer"""
    print("Calculating first purchase dates...")
    
    # Create masks and calculate first purchases in one go
    period1_data = df[(df[purchase_date_col] >= PERIOD1_START) & (df[purchase_date_col] <= PERIOD1_END)]
    period2_data = df[(df[purchase_date_col] >= PERIOD2_START) & (df[purchase_date_col] <= PERIOD2_END)]
    
    first_p1 = period1_data.groupby(customer_nric)[purchase_date_col].min().rename('first_purchase_date_period1')
    first_p2 = period2_data.groupby(customer_nric)[purchase_date_col].min().rename('first_purchase_date_period2')
    
    # Merge both at once
    df = df.merge(first_p1.to_frame().reset_index(), on=customer_nric, how='left')
    df = df.merge(first_p2.to_frame().reset_index(), on=customer_nric, how='left')
    
    return df

#______________________________________________________________________________________________________________________________________
def calculate_repeat_purchase_flags(df, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """Calculate repeat purchase flags for both periods"""
    print("Calculating repeat purchase flags...")
    
    df = df.sort_values([customer_nric, purchase_date_col]).reset_index(drop=True)
    
    # Calculate next purchase date once
    next_purchase = df.groupby(customer_nric)[purchase_date_col].shift(-1)
    
    # Calculate time differences for both periods
    time_diff_p1 = (next_purchase - df['first_purchase_date_period1']).dt.days
    time_diff_p2 = (next_purchase - df['first_purchase_date_period2']).dt.days
    
    # Create period masks
    p1_mask = (df[purchase_date_col] >= PERIOD1_START) & (df[purchase_date_col] <= PERIOD1_END)
    p2_mask = (df[purchase_date_col] >= PERIOD2_START) & (df[purchase_date_col] <= PERIOD2_END)
    
    # Create flags
    df['repeat_purchase_period1'] = ((p1_mask & (time_diff_p1 > 0) & (time_diff_p1 <= LOOKAHEAD_DAYS)).fillna(False).astype(int))
    df['repeat_purchase_period2'] = ((p2_mask & (time_diff_p2 > 0) & (time_diff_p2 <= LOOKAHEAD_DAYS)).fillna(False).astype(int))
    
    return df

#______________________________________________________________________________________________________________________________________
def create_period_features(df, period_num, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """Create final features directly for a specific period (no intermediate storage)"""
    print(f"Processing Period {period_num} features...")
    
    first_purchase_col = f'first_purchase_date_period{period_num}'
    repeat_col = f'repeat_purchase_period{period_num}'
    
    # Get customers with first purchase in this period (matching original logic)
    period_mask = (df[first_purchase_col].notna() & (df[purchase_date_col] == df[first_purchase_col]))
    
    if not period_mask.any():
        return pd.DataFrame()
    
    # Get the base customer data (original raw features)
    base_customer_data = df[period_mask].copy()
    base_customer_data['analysis_period'] = f'Period{period_num}'
    base_customer_data = base_customer_data.rename(columns={repeat_col: 'repeat_purchase_final'})
    
    print(f"Base customer records for Period {period_num}: {len(base_customer_data)}")
    
    # Get unique customers for feature engineering
    customers = base_customer_data[customer_nric].unique()
    period_customers = df[first_purchase_col].notna()
    
    # Create a series with first purchase dates aligned to df index
    first_purchase_series = df[customer_nric].map(df[period_customers].groupby(customer_nric)[first_purchase_col].first())
    
    # Calculate date boundaries vectorised
    lookback_start = first_purchase_series - pd.Timedelta(days=LOOKBACK_DAYS)
    lookback_end = first_purchase_series
    lookahead_start = first_purchase_series
    lookahead_end = first_purchase_series + pd.Timedelta(days=LOOKAHEAD_DAYS)
    
    # Create efficient masks
    lookback_mask = ((df[purchase_date_col] >= lookback_start) & (df[purchase_date_col] < lookback_end) & period_customers)
    lookahead_mask = ((df[purchase_date_col] > lookahead_start) & (df[purchase_date_col] <= lookahead_end) & period_customers)
    
    # Extract data subsets
    pre_data = df[lookback_mask]
    post_data = df[lookahead_mask]
    
    # Initialize result with customer list for features
    feature_result = pd.DataFrame({customer_nric: customers})
    
    # Process categorical features - PRE PERIOD
    for col_name, col_data in CATEGORICAL_COLUMNS.items():
        if col_data in pre_data.columns and len(pre_data) > 0:
            pivot_pre = pd.pivot_table(pre_data, index=customer_nric, columns=col_data, aggfunc='size', fill_value=0)
            pivot_pre.columns = [f'{clean_column_name(col) if col_name in ["plantype_fgi", "Coverage_final"] else col}_pre_final' for col in pivot_pre.columns]
            feature_result = feature_result.merge(pivot_pre.reset_index(), on=customer_nric, how='left')
    
    # Process categorical features - POST PERIOD 
    for col_name, col_data in CATEGORICAL_COLUMNS.items():
        if col_data in post_data.columns and len(post_data) > 0:
            pivot_post = pd.pivot_table(post_data, index=customer_nric, columns=col_data, aggfunc='size', fill_value=0)
            pivot_post.columns = [f'{clean_column_name(col) if col_name in ["plantype_fgi", "Coverage_final"] else col}_post_final' for col in pivot_post.columns]
            feature_result = feature_result.merge(pivot_post.reset_index(), on=customer_nric, how='left')
    
    # Process numerical features - PRE PERIOD
    available_num_cols = [col for col in NUMERICAL_COLUMNS if col in pre_data.columns]
    if available_num_cols and len(pre_data) > 0:
        num_pre = pre_data.groupby(customer_nric)[available_num_cols].mean()
        num_pre.columns = [f'{col}_avg_pre_final' for col in num_pre.columns]
        
        # Add policy count
        policy_count_pre = pre_data.groupby(customer_nric).size().to_frame('policies_pre_final')
        num_pre = num_pre.join(policy_count_pre, how='outer')
        
        feature_result = feature_result.merge(num_pre.reset_index(), on=customer_nric, how='left')
    
    # Process numerical features - POST PERIOD
    available_num_cols = [col for col in NUMERICAL_COLUMNS if col in post_data.columns]
    if available_num_cols and len(post_data) > 0:
        num_post = post_data.groupby(customer_nric)[available_num_cols].mean()
        num_post.columns = [f'{col}_avg_post_final' for col in num_post.columns]
        
        # Add policy count
        policy_count_post = post_data.groupby(customer_nric).size().to_frame('policies_post_final')
        num_post = num_post.join(policy_count_post, how='outer')
        
        feature_result = feature_result.merge(num_post.reset_index(), on=customer_nric, how='left')
    
    # Fill missing values in engineered features
    feature_cols = [col for col in feature_result.columns if col != customer_nric]
    feature_result[feature_cols] = feature_result[feature_cols].fillna(0)
    
    # Create family flags
    family_pre_cols = [col for col in feature_result.columns if 'Family_pre_final' in col]
    family_post_cols = [col for col in feature_result.columns if 'Family_post_final' in col]
    
    feature_result['family_flag_pre_final'] = (feature_result[family_pre_cols].sum(axis=1) > 0).astype(int) if family_pre_cols else 0
    feature_result['family_flag_post_final'] = (feature_result[family_post_cols].sum(axis=1) > 0).astype(int) if family_post_cols else 0
    
    # Merge engineered features with original raw data
    final_result = base_customer_data.merge(feature_result, on=customer_nric, how='left')
    
    # Fill any remaining missing values in engineered features
    engineered_cols = [col for col in feature_result.columns if col != customer_nric]
    final_result[engineered_cols] = final_result[engineered_cols].fillna(0)
    
    print(f"Period {period_num} completed: {final_result.shape}")
    return final_result

#______________________________________________________________________________________________________________________________________
def repeat_purchase_pipeline(file_path, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """
    Main pipeline - creates final features directly without intermediate storage
    """
    print("="*60)
    print("REPEAT PURCHASE PIPELINE")
    print("="*60)
    
    start_time = datetime.now()
    
    # Load and prepare data
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Convert date column
    df[purchase_date_col] = pd.to_datetime(df[purchase_date_col])
    
    # Calculate first purchase dates and repeat flags
    df = calculate_first_purchase_dates(df, customer_nric, purchase_date_col)
    df = calculate_repeat_purchase_flags(df, customer_nric, purchase_date_col)
    
    # Process each period separately (memory efficient)
    period1_result = create_period_features(df, 1, customer_nric, purchase_date_col)
    period2_result = create_period_features(df, 2, customer_nric, purchase_date_col)
    
    # Combine results
    final_result = pd.concat([period1_result, period2_result], ignore_index=True)
    
    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\nPipeline completed in {total_time:.2f} seconds")
    print(f"Final dataset shape: {final_result.shape}")
    print(f"Customers processed: {final_result[customer_nric].nunique():,}")
    
    feature_counts = {
        'Pre-period features': len([col for col in final_result.columns if '_pre_final' in col]),
        'Post-period features': len([col for col in final_result.columns if '_post_final' in col]),
        'Family flags': len([col for col in final_result.columns if 'family_flag' in col])
    }
    
    for feature_type, count in feature_counts.items():
        print(f"{feature_type}: {count}")
    
    return final_result

#______________________________________________________________________________________________________________________________________
# Main execution block
# This allows the script to be run as a standalone module or imported without executing the pipeline
if __name__ == '__main__':
    input_file = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Input_Data/Actual/travelpolicies_2013_2025may.csv'
    output_file = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Output_Data/repeat_purchase_optimised.csv'
    
    try:
        # Run pipeline
        result = repeat_purchase_pipeline(input_file)
        
        # Save results
        result.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise