import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
file_path = '/home/glue_user/workspace/jupyter_workspace/Repeat_Purchase_Model/Output_Data/travelpolicies_2013_2025may_repeatpur_tableau.csv'
df = pd.read_csv(file_path)

columns = ['Analysis_Period', 'Customer_Tenure_Groups', 'Policyholder_Age_Groups', 'Plan_Type_Preference',
           'Coverage_Preference', 'Region_Preference', 'Advance_purchase_avg_pre_final_Groups',
           'Trip_Duration_Avg_Pre_Final_Groups', 'How_often_a_customer_tend_to_buy_from_us_in_a_year?']

df_clean = df[columns].dropna()

print(f"Original dataset shape: {df.shape}")
print(f"Clean dataset shape: {df_clean.shape}")
print(f"Columns removed: {df.shape[1] - df_clean.shape[1]} ({((df.shape[1] - df_clean.shape[1])/df.shape[1]*100):.2f}%)")

def cramers_v(x, y):
    """Calculate Cramer's V statistic for categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

def cramers_v_matrix(df):
    """Create Cramer's V correlation matrix."""
    cols = df.columns
    n = len(cols)
    matrix = np.eye(n)
    
    for i in range(n):
        for j in range(i+1, n):
            v = cramers_v(df.iloc[:, i], df.iloc[:, j])
            matrix[i, j] = matrix[j, i] = v
    
    return pd.DataFrame(matrix, index=cols, columns=cols)

def plot_cramers_heatmap(matrix, figsize=(15, 12), save_path=None):
    """Plot heatmap of Cramer's V matrix."""
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, cmap='Reds', vmin=0, vmax=1, square=True)
    plt.title("Cramer's V Correlation Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    matrix = cramers_v_matrix(df_clean)  # Create correlation matrix
    
    # Save the heatmap as an image
    save_path = "cramers_v_correlation_heatmap.png"
    plot_cramers_heatmap(matrix, save_path=save_path)  # Plot and save