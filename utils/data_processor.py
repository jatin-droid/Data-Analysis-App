"""
Data Processing Utilities
Helper functions for data manipulation and cleaning
"""

import pandas as pd

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all applicable columns to numeric type
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with numeric conversions applied
    """
    df_numeric = df.copy()
    
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    return df_numeric


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of missing values in the dataframe
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': missing_pct.values.round(2)
    })
    
    # Filter to only columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    missing_df = missing_df.sort_values('Missing Count', ascending=False)
    
    return missing_df


def fill_missing_values(df: pd.DataFrame, column: str, method: str = 'mean') -> pd.DataFrame:
    """
    Fill missing values in a column using specified method
    
    Args:
        df: Input dataframe
        column: Column name to fill
        method: Fill method ('mean', 'median', 'mode', 'forward', 'backward')
    
    Returns:
        Dataframe with filled values
    """
    df_filled = df.copy()
    
    if method == 'mean':
        df_filled[column].fillna(df_filled[column].mean(), inplace=True)
    elif method == 'median':
        df_filled[column].fillna(df_filled[column].median(), inplace=True)
    elif method == 'mode':
        mode_val = df_filled[column].mode()
        if not mode_val.empty:
            df_filled[column].fillna(mode_val[0], inplace=True)
    elif method == 'forward':
        df_filled[column].fillna(method='ffill', inplace=True)
    elif method == 'backward':
        df_filled[column].fillna(method='bfill', inplace=True)
    
    return df_filled


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from a numeric column
    
    Args:
        df: Input dataframe
        column: Column name
        method: Outlier detection method ('iqr' or 'zscore')
    
    Returns:
        Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df_clean[column].dropna())
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3)
        df_clean = df_clean[filtered_entries]
    
    return df_clean


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive information about all columns
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with column information
    """
    info_data = []
    
    for col in df.columns:
        col_info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique(),
            'Memory Usage': df[col].memory_usage(deep=True)
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max()
            })
        
        info_data.append(col_info)
    
    return pd.DataFrame(info_data)


def detect_data_types(df: pd.DataFrame) -> dict:
    """
    Detect and categorize column data types
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with categorized columns
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'boolean': boolean_cols
    }