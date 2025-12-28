"""
Integration tests for complete preprocessing pipeline.
Tests feature engineering + data preprocessing together.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_data
from src.feature_engineering import engineer_all_features
from src.data_preprocessing import preprocess_data


class TestCompletePipeline:
    """Test complete preprocessing pipeline with feature engineering."""
    
    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        return load_data('data/Churn Modeling.csv')
    
    def test_feature_engineering_then_preprocessing(self, real_data):
        """Test feature engineering followed by preprocessing."""
        # Step 1: Feature engineering
        df_engineered = engineer_all_features(real_data)
        
        # Verify features are created
        assert 'AgeGroup' in df_engineered.columns
        assert 'BalanceGroup' in df_engineered.columns
        assert 'HasBalance' in df_engineered.columns
        
        # Step 2: Preprocessing
        result = preprocess_data(df_engineered, random_state=42)
        
        # Verify preprocessing worked
        assert 'X_train' in result
        assert 'X_val' in result
        assert 'X_test' in result
        assert len(result['X_train']) > 0
        assert len(result['X_val']) > 0
        assert len(result['X_test']) > 0
        
        # Verify no missing values
        assert result['X_train'].isnull().sum().sum() == 0
        assert result['X_val'].isnull().sum().sum() == 0
        assert result['X_test'].isnull().sum().sum() == 0
    
    def test_pipeline_preserves_data_integrity(self, real_data):
        """Test that pipeline preserves data integrity."""
        # Feature engineering
        df_engineered = engineer_all_features(real_data)
        
        # Preprocessing
        result = preprocess_data(df_engineered, random_state=42)
        
        # Check that all samples are accounted for
        total_samples = len(real_data)
        train_samples = len(result['X_train'])
        val_samples = len(result['X_val'])
        test_samples = len(result['X_test'])
        
        assert train_samples + val_samples + test_samples == total_samples
        
        # Check target distributions
        assert set(result['y_train'].unique()).issubset({0, 1})
        assert set(result['y_val'].unique()).issubset({0, 1})
        assert set(result['y_test'].unique()).issubset({0, 1})
    
    def test_pipeline_feature_consistency(self, real_data):
        """Test that features are consistent across splits."""
        df_engineered = engineer_all_features(real_data)
        result = preprocess_data(df_engineered, random_state=42)
        
        # Check that all splits have same features
        train_features = set(result['X_train'].columns)
        val_features = set(result['X_val'].columns)
        test_features = set(result['X_test'].columns)
        
        assert train_features == val_features == test_features
    
    def test_pipeline_with_smote(self, real_data):
        """Test pipeline with SMOTE enabled."""
        df_engineered = engineer_all_features(real_data)
        result = preprocess_data(df_engineered, use_smote=True, random_state=42)
        
        # Check that resampled data has more samples
        assert len(result['X_train_resampled']) >= len(result['X_train'])
        assert len(result['y_train_resampled']) >= len(result['y_train'])
        
        # Check that resampled features match
        assert set(result['X_train_resampled'].columns) == set(result['X_train'].columns)
    
    def test_pipeline_without_smote(self, real_data):
        """Test pipeline without SMOTE."""
        df_engineered = engineer_all_features(real_data)
        result = preprocess_data(df_engineered, use_smote=False, random_state=42)
        
        # Check that resampled equals original
        assert len(result['X_train_resampled']) == len(result['X_train'])
        assert len(result['y_train_resampled']) == len(result['y_train'])


class TestPipelineEdgeCases:
    """Test edge cases in the pipeline."""
    
    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        return load_data('data/Churn Modeling.csv')
    
    def test_pipeline_handles_all_features(self, real_data):
        """Test that pipeline handles all engineered features correctly."""
        # Feature engineering with all options
        df_engineered = engineer_all_features(
            real_data,
            include_age_groups=True,
            include_balance_groups=True,
            include_tenure_groups=True,
            include_interactions=True,
            include_aggregated=True,
            include_polynomial=False  # Skip polynomial to avoid too many features
        )
        
        # Preprocessing should handle all features
        result = preprocess_data(df_engineered, random_state=42)
        
        # Verify preprocessing succeeded
        assert len(result['X_train'].columns) > 0
        assert result['X_train'].isnull().sum().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

