"""
Comprehensive test suite for EDA utility functions.

This module tests all utility functions to ensure data quality,
correctness of analysis, and proper error handling.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    load_data, get_data_overview, get_statistical_summary,
    check_class_imbalance, plot_target_distribution,
    plot_numeric_distributions, plot_categorical_analysis,
    plot_correlation_matrix, get_feature_importance_correlation,
    validate_data_quality
)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_success(self):
        """Test successful data loading."""
        df = load_data('data/Churn Modeling.csv')
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv')
    
    def test_load_data_columns(self):
        """Test that all expected columns are present."""
        df = load_data('data/Churn Modeling.csv')
        expected_cols = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore',
                        'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                        'EstimatedSalary', 'Exited']
        assert all(col in df.columns for col in expected_cols)


class TestDataOverview:
    """Test data overview functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_get_data_overview_structure(self, sample_df):
        """Test that overview returns correct structure."""
        overview = get_data_overview(sample_df)
        
        assert isinstance(overview, dict)
        assert 'shape' in overview
        assert 'columns' in overview
        assert 'dtypes' in overview
        assert 'memory_usage_mb' in overview
        assert 'null_counts' in overview
        assert 'null_percentages' in overview
        assert 'duplicate_rows' in overview
        assert 'numeric_columns' in overview
        assert 'categorical_columns' in overview
    
    def test_get_data_overview_values(self, sample_df):
        """Test that overview values are correct."""
        overview = get_data_overview(sample_df)
        
        assert overview['shape'] == sample_df.shape
        assert overview['columns'] == sample_df.columns.tolist()
        assert overview['duplicate_rows'] == sample_df.duplicated().sum()
        assert isinstance(overview['memory_usage_mb'], (int, float))
        assert overview['memory_usage_mb'] > 0
    
    def test_get_data_overview_no_missing(self, sample_df):
        """Test that dataset has no missing values."""
        overview = get_data_overview(sample_df)
        null_counts = overview['null_counts']
        
        # Verify no missing values
        assert all(count == 0 for count in null_counts.values()), \
            f"Found missing values: {null_counts}"


class TestStatisticalSummary:
    """Test statistical summary functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_get_statistical_summary_structure(self, sample_df):
        """Test statistical summary structure."""
        summary = get_statistical_summary(sample_df, target_col='Exited')
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        assert len(summary.columns) > 0
    
    def test_get_statistical_summary_target_groups(self, sample_df):
        """Test that summary is grouped by target variable."""
        summary = get_statistical_summary(sample_df, target_col='Exited')
        
        # Should have rows for each target class (0 and 1)
        assert 0 in summary.index.get_level_values(0)
        assert 1 in summary.index.get_level_values(0)


class TestClassImbalance:
    """Test class imbalance checking."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_check_class_imbalance_structure(self, sample_df):
        """Test class imbalance output structure."""
        result = check_class_imbalance(sample_df, target_col='Exited')
        
        assert isinstance(result, dict)
        assert 'counts' in result
        assert 'proportions' in result
        assert 'imbalance_ratio' in result
        assert 'is_imbalanced' in result
    
    def test_check_class_imbalance_values(self, sample_df):
        """Test class imbalance values are correct."""
        result = check_class_imbalance(sample_df, target_col='Exited')
        
        # Check counts sum to total
        total = sum(result['counts'].values())
        assert total == len(sample_df)
        
        # Check proportions sum to 1
        total_prop = sum(result['proportions'].values())
        assert abs(total_prop - 1.0) < 0.001
        
        # Check imbalance ratio is between 0 and 1
        assert 0 <= result['imbalance_ratio'] <= 1
    
    def test_check_class_imbalance_detection(self, sample_df):
        """Test that imbalance is correctly detected."""
        result = check_class_imbalance(sample_df, target_col='Exited')
        
        # Calculate expected ratio
        counts = result['counts']
        expected_ratio = min(counts.values()) / max(counts.values())
        
        assert abs(result['imbalance_ratio'] - expected_ratio) < 0.001


class TestDataQuality:
    """Test data quality validation."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_validate_data_quality_structure(self, sample_df):
        """Test validation output structure."""
        issues = validate_data_quality(sample_df, target_col='Exited')
        
        assert isinstance(issues, dict)
        assert 'missing_values' in issues
        assert 'duplicates' in issues
        assert 'invalid_target_values' in issues
        assert 'outliers' in issues
        assert 'data_types_issues' in issues
    
    def test_validate_data_quality_no_missing(self, sample_df):
        """Test that no missing values are found."""
        issues = validate_data_quality(sample_df, target_col='Exited')
        
        assert len(issues['missing_values']) == 0, \
            f"Found missing values: {issues['missing_values']}"
    
    def test_validate_data_quality_no_duplicates(self, sample_df):
        """Test that no duplicates are found."""
        issues = validate_data_quality(sample_df, target_col='Exited')
        
        assert issues['duplicates'] == 0, \
            f"Found {issues['duplicates']} duplicate rows"
    
    def test_validate_data_quality_target_valid(self, sample_df):
        """Test that target variable has valid values."""
        issues = validate_data_quality(sample_df, target_col='Exited')
        
        assert issues['invalid_target_values'] is None, \
            f"Invalid target values: {issues['invalid_target_values']}"
    
    def test_validate_data_quality_target_values(self, sample_df):
        """Test that target only contains 0 and 1."""
        unique_values = sample_df['Exited'].unique()
        assert all(val in [0, 1] for val in unique_values), \
            f"Target contains unexpected values: {unique_values}"


class TestFeatureCorrelation:
    """Test feature correlation analysis."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_get_feature_importance_correlation_structure(self, sample_df):
        """Test correlation output structure."""
        corr_df = get_feature_importance_correlation(sample_df, target_col='Exited')
        
        assert isinstance(corr_df, pd.DataFrame)
        assert 'Feature' in corr_df.columns
        assert 'Correlation' in corr_df.columns
        assert len(corr_df) > 0
    
    def test_get_feature_importance_correlation_values(self, sample_df):
        """Test correlation values are valid."""
        corr_df = get_feature_importance_correlation(sample_df, target_col='Exited')
        
        # All correlations should be between 0 and 1 (absolute values)
        assert all(0 <= corr <= 1 for corr in corr_df['Correlation']), \
            "Correlation values should be between 0 and 1"
        
        # Should be sorted in descending order
        assert corr_df['Correlation'].is_monotonic_decreasing, \
            "Correlations should be sorted in descending order"
    
    def test_get_feature_importance_correlation_no_target(self, sample_df):
        """Test that target column is not in feature list."""
        corr_df = get_feature_importance_correlation(sample_df, target_col='Exited')
        
        assert 'Exited' not in corr_df['Feature'].values, \
            "Target column should not be in feature list"


class TestVisualizations:
    """Test visualization functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_plot_target_distribution(self, sample_df):
        """Test target distribution plotting."""
        fig = plot_target_distribution(sample_df, target_col='Exited')
        
        assert fig is not None
        assert len(fig.axes) == 2  # Should have 2 subplots
    
    def test_plot_numeric_distributions(self, sample_df):
        """Test numeric distributions plotting."""
        fig = plot_numeric_distributions(sample_df, target_col='Exited')
        
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_categorical_analysis(self, sample_df):
        """Test categorical analysis plotting."""
        categorical_cols = ['Geography', 'Gender']
        fig = plot_categorical_analysis(sample_df, categorical_cols, target_col='Exited')
        
        assert fig is not None
        assert len(fig.axes) == len(categorical_cols)
    
    def test_plot_correlation_matrix(self, sample_df):
        """Test correlation matrix plotting."""
        fig = plot_correlation_matrix(sample_df, target_col='Exited')
        
        assert fig is not None
        # Correlation matrix includes main axes and colorbar
        assert len(fig.axes) >= 1


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return load_data('data/Churn Modeling.csv')
    
    def test_data_shape_consistency(self, sample_df):
        """Test that data shape is consistent."""
        assert sample_df.shape[0] > 0, "Dataset should have rows"
        assert sample_df.shape[1] > 0, "Dataset should have columns"
        assert sample_df.shape[0] == len(sample_df), "Row count mismatch"
    
    def test_target_variable_presence(self, sample_df):
        """Test that target variable exists."""
        assert 'Exited' in sample_df.columns, "Target variable 'Exited' not found"
    
    def test_target_variable_binary(self, sample_df):
        """Test that target variable is binary."""
        unique_values = sorted(sample_df['Exited'].unique())
        assert unique_values == [0, 1], \
            f"Target should be binary [0, 1], got {unique_values}"
    
    def test_numeric_features_valid(self, sample_df):
        """Test that numeric features contain valid values."""
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['RowNumber', 'CustomerId', 'Exited']:
                assert not sample_df[col].isnull().any(), \
                    f"Column {col} contains null values"
                # Check for infinite values
                assert np.isfinite(sample_df[col]).all(), \
                    f"Column {col} contains infinite values"
    
    def test_categorical_features_valid(self, sample_df):
        """Test that categorical features contain valid values."""
        categorical_cols = ['Geography', 'Gender']
        
        for col in categorical_cols:
            assert col in sample_df.columns, f"Column {col} not found"
            assert not sample_df[col].isnull().any(), \
                f"Column {col} contains null values"
    
    def test_binary_features_valid(self, sample_df):
        """Test that binary features contain valid values."""
        binary_cols = ['HasCrCard', 'IsActiveMember']
        
        for col in binary_cols:
            unique_values = sorted(sample_df[col].unique())
            assert unique_values == [0, 1], \
                f"Column {col} should be binary [0, 1], got {unique_values}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            get_data_overview(empty_df)
    
    def test_missing_target_column(self):
        """Test handling when target column is missing."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pytest.raises(KeyError):
            check_class_imbalance(df, target_col='Exited')
    
    def test_single_class_target(self):
        """Test handling of single class in target."""
        df = pd.DataFrame({'Exited': [0, 0, 0, 0, 0]})
        
        result = check_class_imbalance(df, target_col='Exited')
        # When all classes are the same, ratio is 1.0 (perfectly balanced)
        assert result['imbalance_ratio'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

