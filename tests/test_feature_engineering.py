"""
Comprehensive test suite for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.feature_engineering import (
    create_age_groups, create_balance_groups, create_tenure_groups,
    create_interaction_features, create_aggregated_features,
    create_polynomial_features, engineer_all_features,
    get_feature_importance_summary
)
from src.utils import load_data


class TestAgeGroups:
    """Test age group creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'Age': [25, 35, 45, 55, 65, 75]
        })
    
    def test_create_age_groups(self, sample_data):
        """Test age group creation."""
        df_result = create_age_groups(sample_data)
        
        assert 'AgeGroup' in df_result.columns
        assert len(df_result) == len(sample_data)
        
        # Check age groups are created
        assert df_result['AgeGroup'].dtype.name == 'category'
    
    def test_age_groups_categories(self, sample_data):
        """Test age group categories."""
        df_result = create_age_groups(sample_data)
        
        categories = df_result['AgeGroup'].cat.categories.tolist()
        expected = ['<30', '30-40', '40-50', '50-60', '60+']
        
        assert categories == expected
    
    def test_age_groups_correct_assignment(self):
        """Test that ages are assigned to correct groups."""
        df = pd.DataFrame({'Age': [25, 35, 45, 55, 65]})
        df_result = create_age_groups(df)
        
        assert df_result.loc[0, 'AgeGroup'] == '<30'
        assert df_result.loc[1, 'AgeGroup'] == '30-40'
        assert df_result.loc[2, 'AgeGroup'] == '40-50'
        assert df_result.loc[3, 'AgeGroup'] == '50-60'
        assert df_result.loc[4, 'AgeGroup'] == '60+'


class TestBalanceGroups:
    """Test balance group creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'Balance': [0, 25000, 75000, 125000, 175000]
        })
    
    def test_create_balance_groups(self, sample_data):
        """Test balance group creation."""
        df_result = create_balance_groups(sample_data)
        
        assert 'BalanceGroup' in df_result.columns
        assert 'HasBalance' in df_result.columns
        assert len(df_result) == len(sample_data)
    
    def test_has_balance_feature(self, sample_data):
        """Test HasBalance binary feature."""
        df_result = create_balance_groups(sample_data)
        
        assert df_result.loc[0, 'HasBalance'] == 0  # Zero balance
        assert df_result.loc[1, 'HasBalance'] == 1  # Non-zero balance
    
    def test_balance_groups_categories(self):
        """Test balance group categories."""
        df = pd.DataFrame({'Balance': [0, 25000, 75000, 125000, 175000]})
        df_result = create_balance_groups(df)
        
        categories = df_result['BalanceGroup'].cat.categories.tolist()
        expected = ['Zero', 'Low', 'Medium', 'High', 'VeryHigh']
        
        assert categories == expected


class TestTenureGroups:
    """Test tenure group creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'Tenure': [0, 3, 6, 9, 10]
        })
    
    def test_create_tenure_groups(self, sample_data):
        """Test tenure group creation."""
        df_result = create_tenure_groups(sample_data)
        
        assert 'TenureGroup' in df_result.columns
        assert len(df_result) == len(sample_data)
    
    def test_tenure_groups_categories(self):
        """Test tenure group categories."""
        df = pd.DataFrame({'Tenure': [0, 3, 6, 9]})
        df_result = create_tenure_groups(df)
        
        categories = df_result['TenureGroup'].cat.categories.tolist()
        expected = ['New', 'Short', 'Medium', 'Long']
        
        assert categories == expected


class TestInteractionFeatures:
    """Test interaction feature creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'Age': [30, 40, 50],
            'Balance': [10000, 50000, 100000],
            'IsActiveMember': [0, 1, 1],
            'NumOfProducts': [1, 2, 3],
            'CreditScore': [600, 700, 800],
            'EstimatedSalary': [50000, 75000, 100000],
            'Tenure': [2, 5, 8]
        })
    
    def test_create_interaction_features(self, sample_data):
        """Test interaction feature creation."""
        df_result = create_interaction_features(sample_data)
        
        # Check that interaction features are created
        assert 'Age_Balance' in df_result.columns
        assert 'Age_Active' in df_result.columns
        assert 'Balance_Products' in df_result.columns
        assert 'CreditScore_Age' in df_result.columns
        assert 'Salary_Balance_Ratio' in df_result.columns
        assert 'Age_Tenure_Ratio' in df_result.columns
    
    def test_interaction_feature_values(self, sample_data):
        """Test interaction feature values."""
        df_result = create_interaction_features(sample_data)
        
        # Check Age_Balance
        assert df_result.loc[0, 'Age_Balance'] == 30 * 10000
        
        # Check Age_Active
        assert df_result.loc[1, 'Age_Active'] == 40 * 1
        
        # Check Salary_Balance_Ratio (should handle zero balance)
        assert df_result.loc[0, 'Salary_Balance_Ratio'] >= 0
    
    def test_interaction_features_preserve_original(self, sample_data):
        """Test that original columns are preserved."""
        df_result = create_interaction_features(sample_data)
        
        for col in sample_data.columns:
            assert col in df_result.columns


class TestAggregatedFeatures:
    """Test aggregated feature creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'IsActiveMember': [0, 1, 1],
            'NumOfProducts': [1, 2, 3],
            'Balance': [0, 50000, 100000],
            'EstimatedSalary': [50000, 75000, 100000],
            'CreditScore': [600, 700, 800]
        })
    
    def test_create_aggregated_features(self, sample_data):
        """Test aggregated feature creation."""
        df_result = create_aggregated_features(sample_data)
        
        assert 'EngagementScore' in df_result.columns
        assert 'CustomerValueScore' in df_result.columns
        assert 'RiskScore' in df_result.columns
    
    def test_engagement_score(self, sample_data):
        """Test engagement score calculation."""
        df_result = create_aggregated_features(sample_data)
        
        # Engagement = IsActiveMember * NumOfProducts
        assert df_result.loc[0, 'EngagementScore'] == 0 * 1
        assert df_result.loc[1, 'EngagementScore'] == 1 * 2
    
    def test_customer_value_score_range(self, sample_data):
        """Test that customer value score is in [0, 1] range."""
        df_result = create_aggregated_features(sample_data)
        
        assert (df_result['CustomerValueScore'] >= 0).all()
        assert (df_result['CustomerValueScore'] <= 1).all()
    
    def test_risk_score_range(self, sample_data):
        """Test that risk score is in [0, 1] range."""
        df_result = create_aggregated_features(sample_data)
        
        assert (df_result['RiskScore'] >= 0).all()
        assert (df_result['RiskScore'] <= 1).all()


class TestPolynomialFeatures:
    """Test polynomial feature creation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'Age': [30, 40, 50],
            'CreditScore': [600, 700, 800],
            'Balance': [10000, 50000, 100000],
            'EstimatedSalary': [50000, 75000, 100000]
        })
    
    def test_create_polynomial_features(self, sample_data):
        """Test polynomial feature creation."""
        df_result = create_polynomial_features(sample_data, degree=2)
        
        # Check squared features are created
        assert 'Age_squared' in df_result.columns
        assert 'CreditScore_squared' in df_result.columns
        assert 'Balance_squared' in df_result.columns
        assert 'EstimatedSalary_squared' in df_result.columns
    
    def test_polynomial_feature_values(self, sample_data):
        """Test polynomial feature values."""
        df_result = create_polynomial_features(sample_data, degree=2)
        
        assert df_result.loc[0, 'Age_squared'] == 30 ** 2
        assert df_result.loc[1, 'CreditScore_squared'] == 700 ** 2
    
    def test_polynomial_custom_columns(self, sample_data):
        """Test polynomial features with custom columns."""
        df_result = create_polynomial_features(
            sample_data,
            columns=['Age', 'CreditScore'],
            degree=2
        )
        
        assert 'Age_squared' in df_result.columns
        assert 'CreditScore_squared' in df_result.columns
        # Should not create squared for other columns
        assert 'Balance_squared' not in df_result.columns


class TestEngineerAllFeatures:
    """Test engineer_all_features function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample dataframe."""
        return pd.DataFrame({
            'Age': [30, 40, 50],
            'Balance': [0, 50000, 100000],
            'Tenure': [2, 5, 8],
            'CreditScore': [600, 700, 800],
            'EstimatedSalary': [50000, 75000, 100000],
            'IsActiveMember': [0, 1, 1],
            'NumOfProducts': [1, 2, 3],
            'Exited': [0, 1, 0]
        })
    
    def test_engineer_all_features_default(self, sample_data):
        """Test engineer_all_features with default options."""
        df_result = engineer_all_features(sample_data)
        
        # Check that groups are created
        assert 'AgeGroup' in df_result.columns
        assert 'BalanceGroup' in df_result.columns
        assert 'TenureGroup' in df_result.columns
        
        # Check that interactions are created
        assert 'Age_Balance' in df_result.columns
        
        # Check that aggregated features are created
        assert 'EngagementScore' in df_result.columns
    
    def test_engineer_all_features_selective(self, sample_data):
        """Test engineer_all_features with selective options."""
        df_result = engineer_all_features(
            sample_data,
            include_age_groups=True,
            include_balance_groups=False,
            include_tenure_groups=False,
            include_interactions=False,
            include_aggregated=False,
            include_polynomial=False
        )
        
        assert 'AgeGroup' in df_result.columns
        assert 'BalanceGroup' not in df_result.columns
        assert 'TenureGroup' not in df_result.columns
        assert 'Age_Balance' not in df_result.columns
    
    def test_engineer_all_features_preserves_original(self, sample_data):
        """Test that original columns are preserved."""
        df_result = engineer_all_features(sample_data)
        
        for col in sample_data.columns:
            assert col in df_result.columns


class TestFeatureImportanceSummary:
    """Test feature importance summary."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe with correlation."""
        np.random.seed(42)
        n = 100
        
        # Create features with known correlations
        age = np.random.randint(18, 80, n)
        balance = np.random.uniform(0, 200000, n)
        
        # Create target with correlation to age
        exited = (age > 50).astype(int) + np.random.choice([0, 1], n, p=[0.7, 0.3])
        exited = np.clip(exited, 0, 1)
        
        return pd.DataFrame({
            'Age': age,
            'Balance': balance,
            'CreditScore': np.random.randint(300, 850, n),
            'Exited': exited
        })
    
    def test_get_feature_importance_summary(self, sample_data):
        """Test feature importance summary."""
        result = get_feature_importance_summary(sample_data, target_col='Exited')
        
        assert isinstance(result, pd.DataFrame)
        assert 'Feature' in result.columns
        assert 'Correlation' in result.columns
        assert 'AbsCorrelation' in result.columns
        assert len(result) > 0
    
    def test_feature_importance_sorted(self, sample_data):
        """Test that features are sorted by correlation."""
        result = get_feature_importance_summary(sample_data, target_col='Exited')
        
        # Check that correlations are sorted descending
        correlations = result['Correlation'].values
        assert (correlations[:-1] >= correlations[1:]).all()
    
    def test_feature_importance_top_n(self, sample_data):
        """Test top_n parameter."""
        result = get_feature_importance_summary(sample_data, target_col='Exited', top_n=2)
        
        assert len(result) == 2


class TestFeatureEngineeringWithRealData:
    """Test feature engineering with real dataset."""
    
    @pytest.fixture
    def real_data(self):
        """Load real dataset."""
        return load_data('data/Churn Modeling.csv')
    
    def test_engineer_all_features_real_data(self, real_data):
        """Test feature engineering on real data."""
        df_result = engineer_all_features(real_data)
        
        # Check that new features are created
        assert 'AgeGroup' in df_result.columns
        assert 'BalanceGroup' in df_result.columns
        assert 'HasBalance' in df_result.columns
        
        # Check that original columns are preserved
        assert 'Age' in df_result.columns
        assert 'Balance' in df_result.columns
        assert 'Exited' in df_result.columns
    
    def test_feature_engineering_no_data_loss(self, real_data):
        """Test that no data is lost during feature engineering."""
        df_result = engineer_all_features(real_data)
        
        assert len(df_result) == len(real_data)
    
    def test_feature_importance_real_data(self, real_data):
        """Test feature importance on real data."""
        df_engineered = engineer_all_features(real_data)
        result = get_feature_importance_summary(df_engineered, target_col='Exited')
        
        assert len(result) > 0
        assert all(result['Correlation'] >= 0)  # Absolute correlations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

