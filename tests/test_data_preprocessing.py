"""
Comprehensive test suite for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_preprocessing import DataPreprocessor, preprocess_data
from src.utils import load_data


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'RowNumber': range(1, n_samples + 1),
            'CustomerId': range(100000, 100000 + n_samples),
            'Surname': [f'Name{i}' for i in range(n_samples)],
            'CreditScore': np.random.randint(300, 850, n_samples),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 10, n_samples),
            'Balance': np.random.uniform(0, 200000, n_samples),
            'NumOfProducts': np.random.randint(1, 4, n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples),
            'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.target_col == 'Exited'
        assert preprocessor.test_size == 0.2
        assert preprocessor.random_state == 42
        assert preprocessor.use_smote is True
    
    def test_initialization_custom_params(self):
        """Test DataPreprocessor with custom parameters."""
        preprocessor = DataPreprocessor(
            categorical_cols=['Geography', 'Gender'],
            numeric_cols=['Age', 'Balance'],
            target_col='Exited',
            test_size=0.3,
            val_size=0.2,
            random_state=123,
            use_smote=False
        )
        assert preprocessor.categorical_cols == ['Geography', 'Gender']
        assert preprocessor.numeric_cols == ['Age', 'Balance']
        assert preprocessor.test_size == 0.3
        assert preprocessor.random_state == 123
        assert preprocessor.use_smote is False
    
    def test_detect_columns(self, sample_data):
        """Test column detection."""
        preprocessor = DataPreprocessor()
        cat_cols, num_cols = preprocessor._detect_columns(sample_data)
        
        assert 'Geography' in cat_cols
        assert 'Gender' in cat_cols
        assert 'Age' in num_cols
        assert 'Balance' in num_cols
        assert 'Exited' not in num_cols
        assert 'RowNumber' not in num_cols
        assert 'CustomerId' not in num_cols
    
    def test_remove_id_columns(self, sample_data):
        """Test ID column removal."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor._remove_id_columns(sample_data)
        
        assert 'RowNumber' not in df_clean.columns
        assert 'CustomerId' not in df_clean.columns
        assert 'Surname' not in df_clean.columns
        assert 'Exited' in df_clean.columns
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor()
        df_transformed = preprocessor.fit_transform(sample_data)
        
        # Check that ID columns are removed
        assert 'RowNumber' not in df_transformed.columns
        assert 'CustomerId' not in df_transformed.columns
        
        # Check that target is present
        assert 'Exited' in df_transformed.columns
        
        # Check that categorical columns are one-hot encoded
        assert 'Geography_Spain' in df_transformed.columns or 'Geography_Germany' in df_transformed.columns
        
        # Check that numeric columns are present
        assert 'Age' in df_transformed.columns
        assert 'Balance' in df_transformed.columns
        
        # Check preprocessor is fitted
        assert preprocessor.preprocessor is not None
        assert len(preprocessor.feature_names) > 0
    
    def test_fit_transform_preserves_rows(self, sample_data):
        """Test that fit_transform preserves number of rows."""
        preprocessor = DataPreprocessor()
        df_transformed = preprocessor.fit_transform(sample_data)
        
        assert len(df_transformed) == len(sample_data)
    
    def test_transform_after_fit(self, sample_data):
        """Test transform method after fitting."""
        preprocessor = DataPreprocessor()
        
        # Fit on first half
        df_train = sample_data.iloc[:50]
        preprocessor.fit_transform(df_train)
        
        # Transform second half
        df_test = sample_data.iloc[50:]
        df_transformed = preprocessor.transform(df_test)
        
        # Check shape
        assert len(df_transformed) == len(df_test)
        assert 'Exited' in df_transformed.columns
        
        # Check feature names match
        expected_features = set(preprocessor.feature_names)
        actual_features = set(df_transformed.columns) - {preprocessor.target_col}
        assert expected_features == actual_features
    
    def test_transform_before_fit_raises_error(self, sample_data):
        """Test that transform before fit raises error."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_data)
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        preprocessor = DataPreprocessor(test_size=0.2, val_size=0.2, random_state=42)
        df_transformed = preprocessor.fit_transform(sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_transformed)
        
        # Check shapes
        total_samples = len(sample_data)
        assert len(X_train) + len(X_val) + len(X_test) == total_samples
        
        # Check proportions (approximately)
        assert abs(len(X_test) / total_samples - 0.2) < 0.05
        assert abs(len(X_val) / (total_samples - len(X_test)) - 0.2) < 0.05
        
        # Check that features and targets match
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        
        # Check no overlap
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0
    
    def test_handle_imbalance_with_smote(self, sample_data):
        """Test SMOTE imbalance handling."""
        preprocessor = DataPreprocessor(use_smote=True, random_state=42)
        df_transformed = preprocessor.fit_transform(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_transformed)
        
        # Get original class distribution
        original_dist = y_train.value_counts()
        
        # Apply SMOTE
        X_resampled, y_resampled = preprocessor.handle_imbalance(X_train, y_train)
        
        # Check that resampled data has more samples
        assert len(X_resampled) >= len(X_train)
        assert len(y_resampled) >= len(y_train)
        
        # Check that classes are more balanced
        resampled_dist = y_resampled.value_counts()
        imbalance_ratio_original = min(original_dist) / max(original_dist)
        imbalance_ratio_resampled = min(resampled_dist) / max(resampled_dist)
        
        # Resampled should be more balanced (or at least not worse)
        assert imbalance_ratio_resampled >= imbalance_ratio_original - 0.1
    
    def test_handle_imbalance_without_smote(self, sample_data):
        """Test without SMOTE."""
        preprocessor = DataPreprocessor(use_smote=False)
        df_transformed = preprocessor.fit_transform(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_transformed)
        
        X_resampled, y_resampled = preprocessor.handle_imbalance(X_train, y_train)
        
        # Should return original data
        assert len(X_resampled) == len(X_train)
        assert len(y_resampled) == len(y_train)
        assert X_resampled.equals(X_train)
        assert y_resampled.equals(y_train)
    
    def test_save_and_load_preprocessor(self, sample_data, tmp_path):
        """Test saving and loading preprocessor."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_data)
        
        # Save
        filepath = tmp_path / "preprocessor.pkl"
        preprocessor.save_preprocessor(str(filepath))
        
        # Load into new preprocessor
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_preprocessor(str(filepath))
        
        # Check that attributes are loaded
        assert new_preprocessor.categorical_cols == preprocessor.categorical_cols
        assert new_preprocessor.numeric_cols == preprocessor.numeric_cols
        assert new_preprocessor.target_col == preprocessor.target_col
        assert new_preprocessor.feature_names == preprocessor.feature_names
        
        # Test that loaded preprocessor can transform
        df_test = sample_data.iloc[:10]
        df_transformed_original = preprocessor.transform(df_test)
        df_transformed_loaded = new_preprocessor.transform(df_test)
        
        # Check that transformations match
        pd.testing.assert_frame_equal(
            df_transformed_original.drop(columns=['Exited']),
            df_transformed_loaded.drop(columns=['Exited']),
            check_exact=False,
            rtol=1e-5
        )


class TestPreprocessDataFunction:
    """Test preprocess_data convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'RowNumber': range(1, n_samples + 1),
            'CustomerId': range(100000, 100000 + n_samples),
            'Surname': [f'Name{i}' for i in range(n_samples)],
            'CreditScore': np.random.randint(300, 850, n_samples),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(0, 10, n_samples),
            'Balance': np.random.uniform(0, 200000, n_samples),
            'NumOfProducts': np.random.randint(1, 4, n_samples),
            'HasCrCard': np.random.choice([0, 1], n_samples),
            'IsActiveMember': np.random.choice([0, 1], n_samples),
            'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        return pd.DataFrame(data)
    
    def test_preprocess_data_complete_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        result = preprocess_data(sample_data, random_state=42)
        
        # Check all required keys
        required_keys = [
            'preprocessor', 'X_train', 'X_val', 'X_test',
            'y_train', 'y_val', 'y_test',
            'X_train_resampled', 'y_train_resampled'
        ]
        for key in required_keys:
            assert key in result
        
        # Check shapes
        assert len(result['X_train']) == len(result['y_train'])
        assert len(result['X_val']) == len(result['y_val'])
        assert len(result['X_test']) == len(result['y_test'])
        assert len(result['X_train_resampled']) == len(result['y_train_resampled'])
        
        # Check that resampled has more or equal samples
        assert len(result['X_train_resampled']) >= len(result['X_train'])
    
    def test_preprocess_data_without_smote(self, sample_data):
        """Test preprocessing without SMOTE."""
        result = preprocess_data(sample_data, use_smote=False, random_state=42)
        
        # Without SMOTE, resampled should equal original
        assert len(result['X_train_resampled']) == len(result['X_train'])
        assert len(result['y_train_resampled']) == len(result['y_train'])
    
    def test_preprocess_data_custom_splits(self, sample_data):
        """Test preprocessing with custom split sizes."""
        result = preprocess_data(
            sample_data,
            test_size=0.3,
            val_size=0.15,
            random_state=42
        )
        
        total_samples = len(sample_data)
        
        # Check approximate proportions (more lenient for small datasets)
        assert abs(len(result['X_test']) / total_samples - 0.3) < 0.1
        remaining = total_samples - len(result['X_test'])
        assert abs(len(result['X_val']) / remaining - 0.15) < 0.1


class TestPreprocessingWithRealData:
    """Test preprocessing with real dataset."""
    
    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        return load_data('data/Churn Modeling.csv')
    
    def test_preprocess_real_data(self, real_data):
        """Test preprocessing on real dataset."""
        result = preprocess_data(real_data, random_state=42)
        
        # Check that all splits are created
        assert len(result['X_train']) > 0
        assert len(result['X_val']) > 0
        assert len(result['X_test']) > 0
        
        # Check that features are transformed
        assert len(result['X_train'].columns) > 0
        
        # Check that target is binary
        assert set(result['y_train'].unique()).issubset({0, 1})
        assert set(result['y_val'].unique()).issubset({0, 1})
        assert set(result['y_test'].unique()).issubset({0, 1})
    
    def test_preprocess_real_data_no_missing_values(self, real_data):
        """Test that preprocessing handles missing values correctly."""
        result = preprocess_data(real_data, random_state=42)
        
        # Check no missing values in transformed data
        assert result['X_train'].isnull().sum().sum() == 0
        assert result['X_val'].isnull().sum().sum() == 0
        assert result['X_test'].isnull().sum().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

