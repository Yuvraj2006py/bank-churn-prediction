"""
Utility functions for Exploratory Data Analysis (EDA).

This module provides reusable functions for data exploration, visualization,
and statistical analysis to ensure consistency across the project.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
# Use non-interactive backend for testing (when running in test environment)
# Check if we're in a test environment or if backend needs to be set
try:
    current_backend = matplotlib.get_backend()
    # Force Agg backend for testing or if TkAgg is problematic
    if ('PYTEST_CURRENT_TEST' in os.environ or 
        'pytest' in os.environ.get('_', '').lower() or
        current_backend == 'TkAgg' or 
        'tk' in current_backend.lower()):
        matplotlib.use('Agg', force=True)
except Exception:
    # If backend can't be changed, try to use Agg anyway
    try:
        matplotlib.use('Agg')
    except Exception:
        pass  # Backend already set or cannot be changed
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the churn dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
        
    Raises:
    -------
    FileNotFoundError
        If the file path is invalid
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}")


def get_data_overview(df: pd.DataFrame) -> dict:
    """
    Get comprehensive overview of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing overview statistics
        
    Raises:
    -------
    ValueError
        If dataframe is empty
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    overview = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict() if len(df) > 0 else {},
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return overview


def get_statistical_summary(df: pd.DataFrame, target_col: str = 'Exited') -> pd.DataFrame:
    """
    Get statistical summary grouped by target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
        
    Returns:
    --------
    pd.DataFrame
        Statistical summary grouped by target
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    summary = df.groupby(target_col)[numeric_cols].describe()
    return summary


def check_class_imbalance(df: pd.DataFrame, target_col: str = 'Exited') -> dict:
    """
    Check for class imbalance in the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
        
    Returns:
    --------
    dict
        Dictionary with class distribution and imbalance ratio
    """
    class_counts = df[target_col].value_counts().to_dict()
    class_proportions = df[target_col].value_counts(normalize=True).to_dict()
    
    # Handle case where all classes are the same
    if len(class_counts) == 1:
        imbalance_ratio = 1.0  # Perfectly balanced (all same class)
    else:
        imbalance_ratio = min(class_counts.values()) / max(class_counts.values())
    
    return {
        'counts': class_counts,
        'proportions': class_proportions,
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': imbalance_ratio < 0.5
    }


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Exited', 
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot distribution of target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    sns.countplot(data=df, x=target_col, ax=axes[0], hue=target_col, palette='viridis', legend=False)
    axes[0].set_title(f'Distribution of {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')
    
    # Add count labels
    for container in axes[0].containers:
        axes[0].bar_label(container)
    
    # Pie chart
    counts = df[target_col].value_counts()
    axes[1].pie(counts.values, labels=[f'{label} ({val})' for label, val in 
                                       zip(counts.index, counts.values)],
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
    axes[1].set_title(f'Proportion of {target_col}')
    
    plt.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, target_col: str = 'Exited',
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot distributions of numeric features by target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    # Remove RowNumber and CustomerId if present
    numeric_cols = [col for col in numeric_cols if col not in ['RowNumber', 'CustomerId']]
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        df.boxplot(column=col, by=target_col, ax=ax, grid=False)
        ax.set_title(f'{col} by {target_col}')
        ax.set_xlabel(target_col)
        ax.set_ylabel(col)
        ax.get_figure().suptitle('')  # Remove default title
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_categorical_analysis(df: pd.DataFrame, categorical_cols: List[str],
                              target_col: str = 'Exited',
                              figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot categorical features analysis with churn rates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    target_col : str
        Name of the target column
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    n_cols = len(categorical_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        
        # Calculate churn rates
        churn_rates = df.groupby(col)[target_col].agg(['mean', 'count'])
        churn_rates.columns = ['Churn Rate', 'Count']
        
        # Create bar plot
        x_pos = np.arange(len(churn_rates.index))
        bars = ax.bar(x_pos, churn_rates['Churn Rate'], 
                     color=sns.color_palette('viridis', len(churn_rates)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(churn_rates.index, rotation=45, ha='right')
        ax.set_ylabel('Churn Rate')
        ax.set_title(f'Churn Rate by {col}')
        ax.set_ylim([0, 1])
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, churn_rates['Count'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}\n(n={count})',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, target_col: str = 'Exited',
                           figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot correlation matrix for numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove RowNumber and CustomerId if present
    numeric_cols = [col for col in numeric_cols if col not in ['RowNumber', 'CustomerId']]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    return fig


def get_feature_importance_correlation(df: pd.DataFrame, target_col: str = 'Exited') -> pd.DataFrame:
    """
    Get correlation of features with target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and their correlation with target
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    # Remove RowNumber and CustomerId if present
    numeric_cols = [col for col in numeric_cols if col not in ['RowNumber', 'CustomerId']]
    
    correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    result_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    
    return result_df


def validate_data_quality(df: pd.DataFrame, target_col: str = 'Exited') -> dict:
    """
    Validate data quality and return issues found.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    issues = {
        'missing_values': {},
        'duplicates': df.duplicated().sum(),
        'invalid_target_values': None,
        'outliers': {},
        'data_types_issues': []
    }
    
    # Check missing values
    missing = df.isnull().sum()
    issues['missing_values'] = missing[missing > 0].to_dict()
    
    # Check target variable
    if target_col in df.columns:
        unique_values = df[target_col].unique()
        if not all(val in [0, 1] for val in unique_values):
            issues['invalid_target_values'] = f"Target contains unexpected values: {unique_values}"
    
    # Check for outliers in numeric columns (using IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in [target_col, 'RowNumber', 'CustomerId']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                issues['outliers'][col] = outliers
    
    return issues

