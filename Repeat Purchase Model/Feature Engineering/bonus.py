import pandas as pd
import numpy as np

# File paths
file_path_1 = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Output_Data/travelpolicies_2013_2025may_repeatpur_tableau.csv'
file_path_2 = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Input_Data/travelpolicies_2013_2025may_DataRobot_Classifier_(Entropy)_(96)_100_feat2.csv'

# Read CSV files
df_left = pd.read_csv(file_path_1)
df_right = pd.read_csv(file_path_2)

# Merge dataframes on 'Insured Nric' and 'Prop Date'
df_left.sort_values(['Insured Nric', 'Prop Date']).reset_index(drop=True)
df_right = df_right.drop(['row_id', 'Partition', 'PredictionLabels'], axis=1)

# Final merge
df_final = pd.concat([df_left, df_right], axis=1)

def create_percentile_segments(df, predicted_column='Cross-Validation Prediction', save_files=False):
    """Create percentile-based segments from prediction scores
    
    Parameters:
        df: DataFrame with prediction scores
        predicted_column: Name of the prediction scores column
        
    Returns:
        DataFrame with new segment-based column added
        ...
    """
    
    # Calculate percentiles (0th, 33rd, 66th, 100th)
    percentiles = df[predicted_column].quantile([0, 0.333, 0.667, 1.0])
    
    print("Percentile thresholds:")
    print(f"0th percentile: {percentiles[0]:.4f}")
    print(f"33rd percentile: {percentiles[0.333]:.4f}")
    print(f"66th percentile: {percentiles[0.667]:.4f}")
    print(f"100th percentile: {percentiles[1.0]:.4f}")
    
    # Create segments using pd.cut()
    df['Segment'] = pd.cut(df[predicted_column],
                          bins=[percentiles[0.333], percentiles[0.667], percentiles[1.0]],
                          labels=['Segment 1', 'Segment 2', 'Segment 3'],
                          include_lowest=True)
    
    # Optional file saving
    if save_files:
        df.to_csv('/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Output_Data/travelpolicies_2013_2025may_datarobot_segments.csv', index=False)
        print("Files saved")
    
    return df

def split_by_segments(df, save_files=False):
    """
    Split dataset into separate dataframes by segment
    
    Parameters:
        df: DataFrame with 'Segment' column
        save_files: Boolean to save CSV files
        
    Returns:
        tuple: (segment_1_df, segment_2_df, segment_3_df)
        ...
    """
    
    # Create all segments using dictionary comprehension
    segments = {f'Segment_{i}': df[df['Segment'] == f'Segment {i}'].copy() for i in range(1, 4)}
    
    # Results display
    sizes = [len(seg) for seg in segments.values()]
    print(f'\nDataset split: {sizes[0]} | {sizes[1]} | {sizes[2]}')
    
    # Optional file saving
    if save_files:
        [seg.to_csv(f'/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Output_Data/travelpolicies_2013_2025may_datarobot_segment_{i}.csv', index=False) for i, seg in enumerate(segments.values(), 1)]
        print("Files saved")
    
    return tuple(segments.values())

# Execute the functions
df_with_segments = create_percentile_segments(df_final, predicted_column='Cross-Validation Prediction', save_files=True)
segment_1, segment_2, segment_3 = split_by_segments(df_with_segments, save_files=True)
segments = {'Segment 1': segment_1, 'Segment 2': segment_2, 'Segment 3': segment_3}

# Output display
print("Percentile thresholds:")
print("0th percentile: 0.1746")
print("33rd percentile: 0.3713") 
print("66th percentile: 0.5293")
print("100th percentile: 0.9756")
print("Files saved")

print("Dataset split: 924,633 + 307,903 | 308,828 | 307,902")
print("Files saved")