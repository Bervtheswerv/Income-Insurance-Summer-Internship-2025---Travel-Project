# **READMETOO - Repeat Purchase Correlation Matrix Analysis**

## 1. Overview

This script performs exploratory data analysis on categorical features derived from the repeat purchase pipeline output. It creates a Cramer's V correlation matrix to identify associations between customer behavioral variables, supporting feature selection decisions and customer segmentation insights for business visualisation in Tableau.

**Script Purpose**: Generate correlation heatmap for categorical variables to understand feature relationships and guide EDA processes.

**Business Context**: Part of the exploratory data analysis workflow to identify strong associations (≥0.3 threshold) between customer characteristics and purchasing behaviors.

**Output**: Visual correlation heatmap saved as PNG image for business reporting and feature selection documentation.

## 2. Data Source & Preprocessing

#### Input Data
- **File**: `travelpolicies_2013_2025may_repeatpur_tableau.csv`
- **Source**: Output from repeat purchase pipeline, transformed for Tableau visualisation
- **Preprocessing**: Numerical features have been binned into categorical groups via Tableau calculated fields

#### Feature Transformation Pipeline
```
Raw Pipeline Output → Tableau Calculated Fields → Categorical Binning → Correlation Analysis
```
## 3. Analysis of existing variables

The script analyses 9 categorical variables representing different aspects of customer behavior:

#### **A) Temporal & Structural**
- `Analysis_Period` - Whether customer's first purchase was in Period1 or Period2
- `Customer_Tenure_Groups` - Binned customer tenure segments
- `Policyholder_Age_Groups` - Age-based customer segments

#### **B) Preference Patterns**
- `Plan_Type_Preference` - Customer preference for Individual/Group/Family plans
- `Coverage_Preference` - Preference for Classic/Deluxe/Preferred coverage levels
- `Region_Preference` - Geographic preference patterns (ASEAN/ASIA/WORLDWIDE)

#### **C) Behavioral Patterns**
- `Advance_purchase_avg_pre_final_Groups` - Binned advance booking behavior
- `Trip_Duration_Avg_Pre_Final_Groups` - Binned average trip length patterns

#### **D) Target Variable**
- `How_often_a_customer_tend_to_buy_from_us_in_a_year?` - Purchase frequency derived from `Policies_Per_Year`, binned into frequency segments

## 4. Statistical Method: Cramer's V
Cramer's V measures association strength between categorical variables, ranging from 0 (no association) to 1 (perfect association).

#### Mathematical Formula
```
V = √(χ²/(n × min(r-1, c-1)))
```
Where:
- χ² = Chi-square statistic from contingency table
- n = Sample size
- r, c = Number of rows and columns in contingency table

#### Thresholds
- **Strong Association**: ≥0.3
- **Moderate Association**: 0.1-0.3
- **Weak Association**: <0.1

## 5. Core Functions

#### `cramers_v(x, y)`
**Purpose**: Calculate Cramer's V statistic between two categorical variables<p>
**Inputs**: Two pandas Series with categorical data<p>
**Returns**: Float value (0-1) representing association strength<p>
**Key Logic**: 
- Creates contingency table using `pd.crosstab()`
- Performs chi-square test via `chi2_contingency()`
- Normalizes chi-square by sample size and degrees of freedom<p>

#### `cramers_v_matrix(df)`
**Purpose**: Generate full correlation matrix for all variable pairs<p>
**Inputs**: DataFrame with categorical columns<p>
**Returns**: Symmetric correlation matrix as pandas DataFrame<p>
**Key Logic**:
- Creates identity matrix (1s on diagonal)
- Calculates pairwise Cramer's V for upper triangle
- Mirrors values to lower triangle for symmetry
- Preserves original column names as index/columns<p>

#### `plot_cramers_heatmap(matrix, figsize, save_path)`
**Purpose**: Visualize correlation matrix as annotated heatmap<p>
**Inputs**: Correlation matrix, figure dimensions, optional save path<p>
**Returns**: Displays plot and optionally saves PNG file<p>
**Visualisation Features**: Red color scheme (intensity = association strength)<p>

## 6. Output & Usage

#### Primary Output
- **Output File**: `cramers_v_correlation_heatmap.png`

#### Business Applications
1. **Feature Selection**: Identify highly correlated features (≥0.3) for model simplification
2. **Customer Segmentation**: Discover which characteristics cluster together
3. **Business Insights**: Understand relationships between customer preferences and behaviors

#### Customisation Options
- **Figure Size**: Modify `figsize=(15, 12)` parameter
- **Color Scheme**: Change `cmap='Reds'` to other matplotlib colormaps
- **Save Location**: Update `save_path` variable for different output directory
- **Variables**: Modify `columns` list to analyse different feature sets