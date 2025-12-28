"""
Feature Engineering Module for Bank Churn Prediction.

This module creates derived features and interactions to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


def create_age_groups(df: pd.DataFrame, age_col: str = 'Age') -> pd.DataFrame:
    """
    Create age groups from age column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    age_col : str
        Name of age column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with age groups added
    """
    df = df.copy()
    
    # Create age groups
    bins = [0, 30, 40, 50, 60, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60+']
    
    df['AgeGroup'] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True)
    
    return df


def create_balance_groups(df: pd.DataFrame, balance_col: str = 'Balance') -> pd.DataFrame:
    """
    Create balance groups from balance column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    balance_col : str
        Name of balance column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with balance groups added
    """
    df = df.copy()
    
    # Create balance groups
    bins = [-1, 0, 50000, 100000, 150000, float('inf')]
    labels = ['Zero', 'Low', 'Medium', 'High', 'VeryHigh']
    
    df['BalanceGroup'] = pd.cut(df[balance_col], bins=bins, labels=labels, include_lowest=True)
    
    # Also create binary feature for zero balance
    df['HasBalance'] = (df[balance_col] > 0).astype(int)
    
    return df


def create_tenure_groups(df: pd.DataFrame, tenure_col: str = 'Tenure') -> pd.DataFrame:
    """
    Create tenure groups from tenure column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    tenure_col : str
        Name of tenure column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with tenure groups added
    """
    df = df.copy()
    
    # Create tenure groups
    bins = [-1, 2, 5, 8, 11]
    labels = ['New', 'Short', 'Medium', 'Long']
    
    df['TenureGroup'] = pd.cut(df[tenure_col], bins=bins, labels=labels, include_lowest=True)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between important variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with interaction features added
    """
    df = df.copy()
    
    # Age × Balance interaction
    if 'Age' in df.columns and 'Balance' in df.columns:
        df['Age_Balance'] = df['Age'] * df['Balance']
    
    # Age × IsActiveMember interaction
    if 'Age' in df.columns and 'IsActiveMember' in df.columns:
        df['Age_Active'] = df['Age'] * df['IsActiveMember']
    
    # Balance × NumOfProducts interaction
    if 'Balance' in df.columns and 'NumOfProducts' in df.columns:
        df['Balance_Products'] = df['Balance'] * df['NumOfProducts']
    
    # CreditScore × Age interaction
    if 'CreditScore' in df.columns and 'Age' in df.columns:
        df['CreditScore_Age'] = df['CreditScore'] * df['Age']
    
    # EstimatedSalary / Balance ratio (if balance > 0)
    if 'EstimatedSalary' in df.columns and 'Balance' in df.columns:
        df['Salary_Balance_Ratio'] = np.where(
            df['Balance'] > 0,
            df['EstimatedSalary'] / (df['Balance'] + 1),  # Add 1 to avoid division by zero
            0
        )
    
    # Age / Tenure ratio (customer age relative to tenure)
    if 'Age' in df.columns and 'Tenure' in df.columns:
        df['Age_Tenure_Ratio'] = np.where(
            df['Tenure'] > 0,
            df['Age'] / (df['Tenure'] + 1),
            0
        )
    
    return df


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features that combine multiple variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with aggregated features added
    """
    df = df.copy()
    
    # Total engagement score (combination of active membership and products)
    if 'IsActiveMember' in df.columns and 'NumOfProducts' in df.columns:
        df['EngagementScore'] = df['IsActiveMember'] * df['NumOfProducts']
    
    # Customer value score (balance + salary normalized)
    if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
        # Normalize both to 0-1 scale for combination
        balance_norm = (df['Balance'] - df['Balance'].min()) / (df['Balance'].max() - df['Balance'].min() + 1)
        salary_norm = (df['EstimatedSalary'] - df['EstimatedSalary'].min()) / (df['EstimatedSalary'].max() - df['EstimatedSalary'].min() + 1)
        df['CustomerValueScore'] = (balance_norm + salary_norm) / 2
    
    # Risk score (inverse of credit score, normalized)
    if 'CreditScore' in df.columns:
        df['RiskScore'] = 1 - ((df['CreditScore'] - df['CreditScore'].min()) / 
                               (df['CreditScore'].max() - df['CreditScore'].min() + 1))
    
    return df


def create_polynomial_features(df: pd.DataFrame, 
                              columns: Optional[List[str]] = None,
                              degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to create polynomial features for. If None, uses key numeric columns.
    degree : int
        Degree of polynomial features
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with polynomial features added
    """
    df = df.copy()
    
    if columns is None:
        columns = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    
    # Only use columns that exist
    columns = [col for col in columns if col in df.columns]
    
    if degree == 2:
        for col in columns:
            df[f'{col}_squared'] = df[col] ** 2
    
    return df


def engineer_all_features(df: pd.DataFrame,
                         include_age_groups: bool = True,
                         include_balance_groups: bool = True,
                         include_tenure_groups: bool = True,
                         include_interactions: bool = True,
                         include_aggregated: bool = True,
                         include_polynomial: bool = False) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    include_age_groups : bool
        Whether to create age groups
    include_balance_groups : bool
        Whether to create balance groups
    include_tenure_groups : bool
        Whether to create tenure groups
    include_interactions : bool
        Whether to create interaction features
    include_aggregated : bool
        Whether to create aggregated features
    include_polynomial : bool
        Whether to create polynomial features
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all engineered features
    """
    df_engineered = df.copy()
    
    if include_age_groups:
        df_engineered = create_age_groups(df_engineered)
    
    if include_balance_groups:
        df_engineered = create_balance_groups(df_engineered)
    
    if include_tenure_groups:
        df_engineered = create_tenure_groups(df_engineered)
    
    if include_interactions:
        df_engineered = create_interaction_features(df_engineered)
    
    if include_aggregated:
        df_engineered = create_aggregated_features(df_engineered)
    
    if include_polynomial:
        df_engineered = create_polynomial_features(df_engineered)
    
    return df_engineered


def get_feature_importance_summary(df: pd.DataFrame, 
                                   target_col: str = 'Exited',
                                   top_n: int = 10) -> pd.DataFrame:
    """
    Get summary of feature importance based on correlation with target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance summary
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Remove ID columns
    exclude_cols = ['RowNumber', 'CustomerId']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    result_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values,
        'AbsCorrelation': correlations.values
    }).head(top_n)
    
    return result_df

