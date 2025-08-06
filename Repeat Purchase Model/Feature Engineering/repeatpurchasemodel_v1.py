# Imports and data loading
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
file_path = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Input_Data/Actual/travelpolicies_2013_2025may.csv'
df = pd.read_csv(file_path)

# Clean column names function
def clean_column_name(name):
    """Clean column names for consistent formatting"""
    return str(name).replace(' ', '_').replace('/', '_').replace('-', '_').replace('(', '').replace(')', '').replace('.', '_')

# First purchase period function
def first_purchase_period(df_work, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """
    Find first purchase date within each defined period for each customer
    Period 1: 1/6/2022 to 30/5/2023
    Period 2: 1/6/2023 to 30/5/2024
    """
    
    # Define period boundaries
    period1_start = pd.to_datetime('2022-06-01')
    period1_end = pd.to_datetime('2023-05-30')
    period2_start = pd.to_datetime('2023-06-01')
    period2_end = pd.to_datetime('2024-05-30')
    
    # Filter data for each period
    period1_mask = (df_work[purchase_date_col] >= period1_start) & (df_work[purchase_date_col] <= period1_end)
    period2_mask = (df_work[purchase_date_col] >= period2_start) & (df_work[purchase_date_col] <= period2_end)
    
    # Get first purchase date in each period by customer
    period1_first = (df_work[period1_mask].groupby(customer_nric)[purchase_date_col].min().reset_index().rename(columns={purchase_date_col: 'first_purchase_date_period1'}))
    period2_first = (df_work[period2_mask].groupby(customer_nric)[purchase_date_col].min().reset_index().rename(columns={purchase_date_col: 'first_purchase_date_period2'}))
    
    # Merge both periods together and then finally into the original df
    first_purchases = period1_first.merge(period2_first, on=customer_nric, how='outer')
    df_work = df_work.merge(first_purchases, on=customer_nric, how='left')
    
    return df_work

# Repeat purchase function
def repeat_purchase(df_work, customer_nric='InsuredNric', purchase_date_col='PropDate', first_purchase_date_period1='first_purchase_date_period1', first_purchase_date_period2='first_purchase_date_period2'):
    """
    Calculate repeat purchase flags for the two defined periods
    """
    
    # Sort data at customer level
    df_work = df_work.sort_values([customer_nric, purchase_date_col]).reset_index(drop=True)
    
    # Define period boundaries
    period1_start = pd.to_datetime('2022-06-01')
    period1_end = pd.to_datetime('2023-05-30')
    period2_start = pd.to_datetime('2023-06-01')
    period2_end = pd.to_datetime('2024-05-30')
    
    # Conditions for within valid period and if within 366 days
    period1_mask = (df_work[purchase_date_col] >= period1_start) & (df_work[purchase_date_col] <= period1_end)
    period2_mask = (df_work[purchase_date_col] >= period2_start) & (df_work[purchase_date_col] <= period2_end)
    
    time_diff_period1 = (df_work.groupby(customer_nric)[purchase_date_col].shift(-1) - df_work[first_purchase_date_period1]).dt.days
    time_diff_period2 = (df_work.groupby(customer_nric)[purchase_date_col].shift(-1) - df_work[first_purchase_date_period2]).dt.days
    
    # Create flag labels
    df_work['repeat_purchase_period1'] = ((period1_mask & (time_diff_period1 > 0) & (time_diff_period1 <= 366)).fillna(False).astype(int))
    df_work['repeat_purchase_period2'] = ((period2_mask & (time_diff_period2 > 0) & (time_diff_period2 <= 366)).fillna(False).astype(int))
    
    return df_work

# Add categorical features function
def add_categorical_features(df_work, data_subset, customer_nric, feature_prefix, categorical_cols):
    """
    Helper function to add categorical features using pivot_table
    Returns: Original dataframe with new categorical features added
    """
    
    for col_name, col_data in categorical_cols.items():
        if len(data_subset) > 0 and col_data in data_subset.columns:
            # Create pivot table for categorical features
            pivot_features = pd.pivot_table(data_subset, index=customer_nric, columns=col_data, aggfunc='size', fill_value=0)
            
            # Clean column names and add prefix
            if col_name in ['plantype_fgi', 'Coverage_final']:
                pivot_features.columns = [f'{clean_column_name(col)}_{feature_prefix}' for col in pivot_features.columns]
            else:
                pivot_features.columns = [f'{col}_{feature_prefix}' for col in pivot_features.columns]
            
            # Reset index and merge back to main dataframe
            pivot_features = pivot_features.reset_index()
            df_work = df_work.merge(pivot_features, on=customer_nric, how='left')
            
            # Fill NaN values with 0 for new feature columns
            new_cols = [col for col in pivot_features.columns if col != customer_nric]
            df_work[new_cols] = df_work[new_cols].fillna(0)
    
    return df_work

# Add numerical features function
def add_numerical_features(df_work, data_subset, customer_nric, feature_prefix, numerical_columns):
    """
    Helper function to add numerical features using aggregation
    Returns: Original dataframe with new numerical features added
    """
    
    if len(data_subset) > 0:
        # Calculate numerical aggregations
        numerical_features = data_subset.groupby(customer_nric)[numerical_columns].mean()
        numerical_features.columns = [f'{col}_avg_{feature_prefix}' for col in numerical_features.columns]
        
        # Add policy count with specific naming for pre/post periods
        if 'period1_lookback' in feature_prefix:
            policy_count_col = 'policies_pre_period1'
        elif 'period1_lookahead' in feature_prefix:
            policy_count_col = 'policies_post_period1'
        elif 'period2_lookback' in feature_prefix:
            policy_count_col = 'policies_pre_period2'
        elif 'period2_lookahead' in feature_prefix:
            policy_count_col = 'policies_post_period2'
        else:
            policy_count_col = f'policies_{feature_prefix}'
        
        policy_count = data_subset.groupby(customer_nric).size().to_frame(policy_count_col)
        
        # Combine numerical features and policy count
        combined_features = pd.concat([numerical_features, policy_count], axis=1).reset_index()
        df_work = df_work.merge(combined_features, on=customer_nric, how='left')
        
        # Fill NaN values with 0 for new feature columns
        new_cols = [col for col in combined_features.columns if col != customer_nric]
        df_work[new_cols] = df_work[new_cols].fillna(0)
    
    return df_work

# Add policy count function
def add_policy_count(df_work, data_subset, customer_nric, policy_count_col):
    """
    Helper function to add only policy count (for lookahead periods)
    Returns: Original dataframe with policy count feature added
    """
    
    if len(data_subset) > 0:
        # Add policy count only
        policy_count = data_subset.groupby(customer_nric).size().to_frame(policy_count_col).reset_index()
        df_work = df_work.merge(policy_count, on=customer_nric, how='left')
        
        # Fill NaN values with 0 for new feature column
        df_work[policy_count_col] = df_work[policy_count_col].fillna(0)
    
    return df_work

# Extract Period 1 features function
def extract_period1_features(df_work, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """
    Extract Period 1 lookback and lookahead features and append to dataframe
    Returns: Original dataframe with Period 1 features added
    """
    
    # Filter to customers with Period 1 purchases
    period1_customers = df_work['first_purchase_date_period1'].notna()
    
    if not period1_customers.any():
        return df_work
    
    # Calculate Period 1 date boundaries
    p1_lookback_start = df_work['first_purchase_date_period1'] - pd.Timedelta(days=366)
    p1_lookback_end = df_work['first_purchase_date_period1']
    p1_lookahead_start = df_work['first_purchase_date_period1']
    p1_lookahead_end = df_work['first_purchase_date_period1'] + pd.Timedelta(days=366)
    
    # Create Period 1 masks
    p1_lookback_mask = ((df_work[purchase_date_col] >= p1_lookback_start) & (df_work[purchase_date_col] < p1_lookback_end) & period1_customers)
    p1_lookahead_mask = ((df_work[purchase_date_col] > p1_lookahead_start) & (df_work[purchase_date_col] <= p1_lookahead_end) & period1_customers)
    
    # Extract Period 1 data subsets
    p1_lookback_data = df_work[p1_lookback_mask]
    p1_lookahead_data = df_work[p1_lookahead_mask]
    
    # Define categorical columns
    categorical_cols = {'Region': 'Region', 'plantype_fgi': 'plantype_fgi', 'Coverage_final': 'Coverage_final'}
    
    # Define numerical columns
    numerical_columns = ['Claim_count', 'Discounts', 'Advance_purchase', 'trip_duration', 'Insured_Premium', 'Tenure_Years', 'claim_close_days']
    
    # Add Period 1 lookback (pre) features
    df_work = add_categorical_features(df_work, p1_lookback_data, customer_nric, 'period1_lookback', categorical_cols)
    df_work = add_numerical_features(df_work, p1_lookback_data, customer_nric, 'period1_lookback', numerical_columns)

    # Add Period 1 lookahead (post) features
    df_work = add_categorical_features(df_work, p1_lookahead_data, customer_nric, 'period1_lookahead', categorical_cols)
    df_work = add_numerical_features(df_work, p1_lookahead_data, customer_nric, 'period1_lookahead', numerical_columns)
    
    return df_work

# Extract Period 2 features function
def extract_period2_features(df_work, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """
    Extract Period 2 lookback and lookahead features and append to dataframe
    Returns: Original dataframe with Period 2 features added
    """
    
    # Filter to customers with Period 2 purchases
    period2_customers = df_work['first_purchase_date_period2'].notna()
    
    if not period2_customers.any():
        return df_work
    
    # Calculate Period 2 date boundaries
    p2_lookback_start = df_work['first_purchase_date_period2'] - pd.Timedelta(days=366)
    p2_lookback_end = df_work['first_purchase_date_period2']
    p2_lookahead_start = df_work['first_purchase_date_period2']
    p2_lookahead_end = df_work['first_purchase_date_period2'] + pd.Timedelta(days=366)
    
    # Create Period 2 masks vectorized
    p2_lookback_mask = ((df_work[purchase_date_col] >= p2_lookback_start) & (df_work[purchase_date_col] < p2_lookback_end) & period2_customers)
    p2_lookahead_mask = ((df_work[purchase_date_col] > p2_lookahead_start) & (df_work[purchase_date_col] <= p2_lookahead_end) & period2_customers)
    
    # Extract Period 2 data subsets
    p2_lookback_data = df_work[p2_lookback_mask]
    p2_lookahead_data = df_work[p2_lookahead_mask]
    
    # Define categorical columns
    categorical_cols = {'Region': 'Region', 'plantype_fgi': 'plantype_fgi', 'Coverage_final': 'Coverage_final'}
    
    # Define numerical columns
    numerical_columns = ['Claim_count', 'Discounts', 'Advance_purchase','trip_duration', 'Insured_Premium', 'Tenure_Years', 'claim_close_days']
    
    # Add Period 2 lookback (pre) features
    df_work = add_categorical_features(df_work, p2_lookback_data, customer_nric, 'period2_lookback', categorical_cols)
    df_work = add_numerical_features(df_work, p2_lookback_data, customer_nric, 'period2_lookback', numerical_columns)

    # Add Period 2 lookahead (post) features
    df_work = add_categorical_features(df_work, p2_lookahead_data, customer_nric, 'period2_lookahead', categorical_cols)
    df_work = add_numerical_features(df_work, p2_lookahead_data, customer_nric, 'period2_lookahead', numerical_columns)
    
    return df_work

# Family flag function
def family_flag(df_work):
    """
    Create family flags based on family-related features
    Returns: Original dataframe with family flag features added
    """
    
    # Find family-related columns and create flags
    family_cols_mapping = {
        'family_flag_period1_pre': [col for col in df_work.columns if 'Family_period1_lookback' in col],
        'family_flag_period1_post': [col for col in df_work.columns if 'Family_period1_lookahead' in col],
        'family_flag_period2_pre': [col for col in df_work.columns if 'Family_period2_lookback' in col],
        'family_flag_period2_post': [col for col in df_work.columns if 'Family_period2_lookahead' in col]
    }
    
    # Create family flags
    for flag_name, cols in family_cols_mapping.items():
        df_work[flag_name] = (df_work[cols].sum(axis=1) > 0).astype(int) if cols else 0
    
    return df_work

# Combined period feature engineering function
def combined_period_feature_engineering(df, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """
    Main function to run complete period-based feature engineering with runtime tracking
    Process:
    1. Prepare data and add first purchase dates
    2. Add repeat purchase flags
    3. Extract Period 1 features (lookback/lookahead)
    4. Extract Period 2 features (lookback/lookahead)
    5. Add family flags
    6. Track performance metrics
    
    Returns: Original dataframe with all new features appended
    """
    
    print("="*60)
    print("PERIOD-BASED FEATURE ENGINEERING")
    print("="*60)
    
    # Track overall runtime
    overall_start = datetime.now()
    
    # ========================================================================
    # Step 1: Prepare data and add first purchase dates
    # ========================================================================
    
    print("Step 1: Adding first purchase dates...")
    step1_start = datetime.now()
    df_work = df.copy()
    df_work[purchase_date_col] = pd.to_datetime(df_work[purchase_date_col])
    df_work = first_purchase_period(df_work, customer_nric, purchase_date_col)
    step1_time = (datetime.now() - step1_start).total_seconds()
    print(f"Step 1 completed in {step1_time:.2f} seconds")
    
    # ========================================================================
    # Step 2: Add repeat purchase flags
    # ========================================================================
    
    print("Step 2: Adding repeat purchase flags...")
    step2_start = datetime.now()
    df_work = repeat_purchase(df_work, customer_nric, purchase_date_col)
    step2_time = (datetime.now() - step2_start).total_seconds()
    print(f"Step 2 completed in {step2_time:.2f} seconds")
    
    # ========================================================================
    # Step 3: Extract Period 1 features
    # ========================================================================
    
    print("Step 3: Extracting Period 1 features...")
    step3_start = datetime.now()
    df_work = extract_period1_features(df_work, customer_nric, purchase_date_col)
    step3_time = (datetime.now() - step3_start).total_seconds()
    print(f"Step 3 completed in {step3_time:.2f} seconds")
    
    # ========================================================================
    # Step 4: Extract Period 2 features
    # ========================================================================
    
    print("Step 4: Extracting Period 2 features...")
    step4_start = datetime.now()
    df_work = extract_period2_features(df_work, customer_nric, purchase_date_col)
    step4_time = (datetime.now() - step4_start).total_seconds()
    print(f"Step 4 completed in {step4_time:.2f} seconds")
    
    # ========================================================================
    # Step 5: Add family flags
    # ========================================================================
    
    print("Step 5: Adding family flags...")
    step5_start = datetime.now()
    df_work = family_flag(df_work)
    step5_time = (datetime.now() - step5_start).total_seconds()
    print(f"Step 5 completed in {step5_time:.2f} seconds")
    
    # ========================================================================
    # Step 6: Final performance summary
    # ========================================================================
    
    overall_time = (datetime.now() - overall_start).total_seconds()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETED")
    print("="*60)
    print(f"Total runtime: {overall_time:.2f} seconds")
    print(f"Original shape: {df.shape}")
    print(f"Final shape: {df_work.shape}")
    
    # Feature summary
    period1_features = [col for col in df_work.columns if 'period1' in col]
    period2_features = [col for col in df_work.columns if 'period2' in col]
    repeat_purchase_features = [col for col in df_work.columns if 'repeat_purchase' in col]
    family_features = [col for col in df_work.columns if 'family_flag' in col]
    first_purchase_features = [col for col in df_work.columns if 'first_purchase_date' in col]
    
    print(f"First purchase date features: {len(first_purchase_features)}")
    print(f"Period 1 features: {len(period1_features)}")
    print(f"Period 2 features: {len(period2_features)}")
    print(f"Repeat purchase features: {len(repeat_purchase_features)}")
    print(f"Family flag features: {len(family_features)}")
    print(f"Total new features added: {len(first_purchase_features) + len(period1_features) + len(period2_features) + len(repeat_purchase_features) + len(family_features)}")
    
    return df_work

def create_final_features(customer_level_df):
    """
    Create final features from period1/period2 features with pre/post naming
    """
    print("Creating final features...")
    
    cols = customer_level_df.columns.tolist()
    created = 0
    
    # Get all period features (exclude first_purchase_date columns)
    period_features = [col for col in cols if ('period1' in col or 'period2' in col) and 'first_purchase_date' not in col]
    
    # Extract unique base names and their types
    base_features = set()
    for col in period_features:
        # Remove period and timing suffixes to get base name
        base = col.replace('_period1_lookback', '').replace('_period2_lookback', '').replace('_period1_lookahead', '').replace('_period2_lookahead', '').replace('_period1', '').replace('_period2', '')
        base_features.add(base)
    
    # Create final features for each base
    for base in base_features:
        # lookback (pre) features
        p1_pre = f"{base}_period1_lookback"
        p2_pre = f"{base}_period2_lookback" 
        final_pre = f"{base}_pre_final"
        
        if p1_pre in cols or p2_pre in cols:
            customer_level_df[final_pre] = 0  # default
            
            if p1_pre in cols:
                mask = customer_level_df['analysis_period'] == 'Period1'
                customer_level_df.loc[mask, final_pre] = customer_level_df.loc[mask, p1_pre]
            
            if p2_pre in cols:
                mask = customer_level_df['analysis_period'] == 'Period2'
                customer_level_df.loc[mask, final_pre] = customer_level_df.loc[mask, p2_pre]
            
            created += 1
        
        # lookahead (post) features  
        p1_post = f"{base}_period1_lookahead"
        p2_post = f"{base}_period2_lookahead"
        final_post = f"{base}_post_final"
        
        if p1_post in cols or p2_post in cols:
            customer_level_df[final_post] = 0  # default
            
            if p1_post in cols:
                mask = customer_level_df['analysis_period'] == 'Period1'
                customer_level_df.loc[mask, final_post] = customer_level_df.loc[mask, p1_post]
            
            if p2_post in cols:
                mask = customer_level_df['analysis_period'] == 'Period2'
                customer_level_df.loc[mask, final_post] = customer_level_df.loc[mask, p2_post]
            
            created += 1
    
    # Handle special naming for policies (they don't follow the standard pattern)
    special_cases = [
        (['policies_pre_period1', 'policies_pre_period2'], 'policies_pre_final'),
        (['policies_post_period1', 'policies_post_period2'], 'policies_post_final'),
        (['repeat_purchase_period1', 'repeat_purchase_period2'], 'repeat_purchase_final')
    ]

    for period_cols, final_col in special_cases:
        p1_col, p2_col = period_cols
        
        if p1_col in cols or p2_col in cols:
            customer_level_df[final_col] = 0
            
            if p1_col in cols:
                mask = customer_level_df['analysis_period'] == 'Period1'
                customer_level_df.loc[mask, final_col] = customer_level_df.loc[mask, p1_col]
            
            if p2_col in cols:
                mask = customer_level_df['analysis_period'] == 'Period2'
                customer_level_df.loc[mask, final_col] = customer_level_df.loc[mask, p2_col]
            
            created += 1
    
    print(f"Created {created} final features")
    return customer_level_df

def final_dataset(df_processed, customer_nric='InsuredNric', purchase_date_col='PropDate'):
    """
    Create final customer dataset - combine Period 1 and Period 2 data
    """
    print("Creating final dataset...")
    
    # Get Period 1 customers
    period1_mask = (df_processed['first_purchase_date_period1'].notna() & (df_processed[purchase_date_col] == df_processed['first_purchase_date_period1']))
    period1_data = df_processed[period1_mask].copy()
    period1_data['analysis_period'] = 'Period1'
    
    # Get Period 2 customers
    period2_mask = (df_processed['first_purchase_date_period2'].notna() & (df_processed[purchase_date_col] == df_processed['first_purchase_date_period2']))
    period2_data = df_processed[period2_mask].copy() 
    period2_data['analysis_period'] = 'Period2'
    
    # Combine periods
    customer_level_df = pd.concat([period1_data, period2_data], ignore_index=True)
    
    print(f"Dataset created: {len(customer_level_df):,} records")
    
    # Create final features
    customer_level_df = create_final_features(customer_level_df)
    
    return customer_level_df