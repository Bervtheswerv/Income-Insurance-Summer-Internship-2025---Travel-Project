import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# ROUND 1: BASIC FEATURE ENGINEERING
# ============================================================================

def benefit_counter(df):
    """
    Create binary counter columns for RG benefit amounts.
    Each RG_count column will be 1 if corresponding RG_capped > 0, else 0.
    """
    rg_columns = ['RG5_capped', 'RG6_capped', 'RG7_capped', 'RG8_capped', 'RG9_capped', 'RG10_capped']
    
    for rg_col in rg_columns:
        # Extract the RG number (e.g., 'RG5_capped' -> 'RG5_count')
        count_col = rg_col.replace('_capped', '_count')
        
        # Create binary indicator: 1 if amount > 0, else 0
        df[count_col] = (df[rg_col] > 0).astype(int)
    
    return df

def date_features(df):
    """
    Extract date-based features from EffDate and ExpDate columns.
    """
    # Convert date columns to datetime
    df['EffDate'] = pd.to_datetime(df['EffDate'])
    df['ExpDate'] = pd.to_datetime(df['ExpDate'])
    
    # Extract travel month from EffDate
    df['Travel_month'] = df['EffDate'].dt.strftime('%m')
    
    # Extract expiry year from ExpDate
    df['Expiry_year'] = df['ExpDate'].dt.year
    
    return df

def trip_duration_bins(df):
    """
    Group trip_duration into bins of 5 days, with >30 grouped together.
    Bins: 0-5, 6-10, 11-15, 16-20, 21-25, 26-30, >30
    """
    bins = [0, 6, 11, 16, 21, 26, 30, float('inf')]
    labels = ['0-5 days', '6-10 days', '11-15 days', '16-20 days', '21-25 days', '26-30 days', '>30 days']
    df['trip_duration_binned'] = pd.cut(df['trip_duration'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    return df

def age_bins(df):
    """
    Group Insured_Age into bins of 10 years, with >90 grouped together.
    Bins: 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90+
    """
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    df['Insured_Age_binned'] = pd.cut(df['Insured_age'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    return df

def advance_purchase_bins(df):
    """
    Group advance_purchase into weekly bins (every 7 days).
    Bins: 0-7, 8-14, 15-21, 22-28, 29+ (since max is 30, group >30 together)
    """
    # Define bin edges for weekly groupings
    bins = [0, 8, 15, 22, 29, float('inf')]
    labels = ['0-7 days', '8-14 days', '15-21 days', '22-28 days', '29+ days']
    df['Advance_purchase_binned'] = pd.cut(df['Advance_purchase'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    return df

def cumulative_policies_bins(df):
    """
    Group Cumulative_Policies by 10, with >60 grouped together.
    Bins: 1-10, 11-20, 21-30, 31-40, 41-50, 51-60, 60+
    """
    bins = [0, 11, 21, 31, 41, 51, 61, float('inf')]
    labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+']
    df['Cumulative_Policies_binned'] = pd.cut(df['Cumulative_Policies'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    return df

def avg_claims_amount_bins(df):
    """
    Group avg_past_3_trips_claims_amount into bins of 100, with >500 grouped together.
    Bins: 0, 1-100, 101-200, 201-300, 301-400, 401-500, >500
    """
    if 'avg_past_3_trips_claims_amount' not in df.columns:
        df['avg_past_3_trips_claims_amount'] = 0.0
    
    bins = [0, 1, 101, 201, 301, 401, 501, float('inf')]
    labels = ['0', '1-100', '101-200', '201-300', '301-400', '401-500', '>500']
    df['avg_past_3_trips_claims_amount_binned'] = pd.cut(df['avg_past_3_trips_claims_amount'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    return df

def feature_engineering(df):
    """
    Apply all Round 1 transformations.
    """
    print("\n" + "="*60)
    print("ROUND 1: BASIC FEATURE ENGINEERING")
    print("="*60)
    
    df = benefit_counter(df)
    df = date_features(df)
    df = trip_duration_bins(df)
    df = age_bins(df)
    df = advance_purchase_bins(df)
    df = cumulative_policies_bins(df)
    df = avg_claims_amount_bins(df)
    
    print(f"Round 1 completed. Dataset shape: {df.shape}")
    return df

# ============================================================================
# ROUND 2: TRANSPOSE AND AGGREGATION
# ============================================================================

def validate_totals(df_original, df_transformed):
    """
    Checks if original row total = transposed column total
    """
    original_amount = (df_original['RG5_capped'] + df_original['RG6_capped'] +
                      df_original['RG7_capped'] + df_original['RG8_capped'] +
                      df_original['RG9_capped'] + df_original['RG10_capped']).sum()
    
    original_count = (df_original['RG5_count'] + df_original['RG6_count'] +
                     df_original['RG7_count'] + df_original['RG8_count'] +
                     df_original['RG9_count'] + df_original['RG10_count']).sum()
    
    transformed_amount = df_transformed['benefit_amount'].sum()
    transformed_count = df_transformed['benefit_count'].sum()
    
    print(f"Benefit amount: original row total = {original_amount}, transposed column total = {transformed_amount}")
    print(f"Benefit count: original row total = {original_count}, transposed column total = {transformed_count}")

def datatypes(df):
    """
    Establish policy attributes as categorical groupby features
    """
    policy_attributes = ['Expiry_year', 'Coverage_final', 'Travel_month', 'trip_duration_binned',
                        'Insured_Age_binned', 'Advance_purchase_binned', 'Cumulative_Policies_binned',
                        'avg_past_3_trips_claims_amount_binned', 'Region', 'plantype_fgi', 'benefit_type']
    
    for col in policy_attributes:
        df[col] = df[col].astype('category')
    
    # Numeric features
    df['benefit_amount'] = pd.to_numeric(df['benefit_amount'], downcast='float')
    df['benefit_count'] = pd.to_numeric(df['benefit_count'], downcast='integer')
    
    return df

def transpose_benefits(df):
    """Transpose benefits amount data"""
    policy_attributes = ['Expiry_year', 'Coverage_final', 'Travel_month', 'trip_duration_binned',
                        'Insured_Age_binned', 'Advance_purchase_binned', 'Cumulative_Policies_binned',
                        'avg_past_3_trips_claims_amount_binned', 'Region', 'plantype_fgi']
    
    # Establish RG columns
    rg_amount_columns = ['RG5_capped', 'RG6_capped', 'RG7_capped', 'RG8_capped', 'RG9_capped', 'RG10_capped']
    rg_count_columns = ['RG5_count', 'RG6_count', 'RG7_count', 'RG8_count', 'RG9_count', 'RG10_count']
    
    # STEP 1: Melt AMOUNTS
    df_amounts_long = pd.melt(df[policy_attributes + rg_amount_columns],
                             id_vars=policy_attributes,
                             value_vars=rg_amount_columns,
                             var_name='benefit_type',
                             value_name='benefit_amount')
    
    # Extract benefit type
    df_amounts_long['benefit_type'] = df_amounts_long['benefit_type'].str[:3]  # RG5, RG6, etc.
    
    # STEP 2: Melt COUNTS
    df_counts_long = pd.melt(df[policy_attributes + rg_count_columns],
                            id_vars=policy_attributes,
                            value_vars=rg_count_columns,
                            var_name='benefit_type',
                            value_name='benefit_count')
    
    # Extract benefit type
    df_counts_long['benefit_type'] = df_counts_long['benefit_type'].str[:3]  # RG5, RG6, etc.
    
    # STEP 3: Merge together
    df_combined = pd.merge(df_amounts_long, df_counts_long, on=policy_attributes + ['benefit_type'])
    
    # STEP 4: Optimize datatypes
    df_combined = datatypes(df_combined)
    
    # STEP 5: Aggregate data
    df_aggregated = (df_combined.groupby(policy_attributes + ['benefit_type'], observed=True).agg({'benefit_amount': 'sum', 'benefit_count': 'sum'}).reset_index())
    
    return df_aggregated

def generate_summary(df_transformed):
    """Generate summary statistics"""
    print(f"\nDataset: {len(df_transformed):,} records")
    print(f"Benefit types: {df_transformed['benefit_type'].nunique()}")
    print(f"Total amount: ${df_transformed['benefit_amount'].sum():,.2f}")
    print(f"Total count: {df_transformed['benefit_count'].sum():,}")
    
    # Top benefit types by amount
    benefit_totals = (df_transformed.groupby('benefit_type', observed=True)
                     ['benefit_amount'].sum().sort_values(ascending=False))
    print(f"\nTop benefit types by amount:")
    for bg, amt in benefit_totals.head(3).items():
        print(f"  {bg}: ${amt:,.2f}")

def transpose_and_aggregate(df):
    """
    Apply Round 2: Transpose and Aggregation
    """
    print("\n" + "="*60)
    print("ROUND 2: TRANSPOSE & AGGREGATION")
    print("="*60)
    
    # Store original for validation
    df_original = df.copy()
    
    # Transform
    df_transformed = transpose_benefits(df)
    
    # Validate
    validate_totals(df_original, df_transformed)
    
    # Summary
    generate_summary(df_transformed)
    
    print(f"Round 2 completed. Dataset shape: {df_transformed.shape}")
    return df_transformed

# ============================================================================
# CONSOLIDATED MAIN FUNCTION
# ============================================================================

def main(input_file_path, output_file_path):
    """
    Consolidated main function that runs both rounds of feature engineering.
    """
    print("="*80)
    print("CONSOLIDATED PROFITABILITY MODEL FEATURE ENGINEERING")
    print("="*80)
    
    start_time = datetime.now()
    
    # Load data
    print(f"\nLoading data from: {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # ROUND 1: Basic feature engineering
    df_round1 = feature_engineering(df)
    
    # ROUND 2: Transpose and aggregation
    df_final = transpose_and_aggregate(df_round1)
    
    # Save final result
    print(f"\nSaving final dataset to: {output_file_path}")
    df_final.to_csv(output_file_path, index=False)
    
    # Final summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("CONSOLIDATED FEATURE ENGINEERING COMPLETED")
    print("="*80)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Original shape: {df.shape}")
    print(f"Final shape: {df_final.shape}")
    print(f"Output saved to: {output_file_path}")
    
    return df_final

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Define file paths
    input_path = '/home/blue_user/workspace/jupyter_workspace/Profitability_Model/Input_Data/travelpolicies_wpred.csv'
    output_path = '/home/blue_user/workspace/jupyter_workspace/Profitability_Model/Output_Data/travelpolicies_consolidated_final.csv'
    
    # Run consolidated feature engineering
    df_final = main(input_path, output_path)
    
    # Quick preview
    print(f"\nPreview of final dataset:")
    preview_cols = ['Expiry_year', 'Coverage_final', 'benefit_type', 'benefit_amount', 'benefit_count']
    print(df_final[preview_cols].head(10))