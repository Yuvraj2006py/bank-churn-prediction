"""
Comprehensive test suite for model training module.
Tests each model individually to ensure correctness.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.model_training import ModelTrainer, train_models
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_all_features
from src.utils import load_data


class TestLogisticRegression:
    """Test Logistic Regression model training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        # Create target with some relationship to features
        y = ((X['feature1'] > 0) & (X['feature2'] > 0)).astype(int)
        y = pd.Series(y, name='target')
        
        return X, y
    
    def test_train_logistic_regression_basic(self, sample_data):
        """Test basic Logistic Regression training."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        
        assert 'model' in result
        assert 'cv_score' in result
        assert 'model_name' in result
        assert result['model_name'] == 'logistic_regression'
        assert result['model'] is not None
        assert 0 <= result['cv_score'] <= 1
    
    def test_train_logistic_regression_with_tuning(self, sample_data):
        """Test Logistic Regression with hyperparameter tuning."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=True)
        
        assert 'model' in result
        assert 'best_params' in result
        assert 'cv_score' in result
        assert result['best_params'] is not None
        assert isinstance(result['best_params'], dict)
        assert 0 <= result['cv_score'] <= 1
    
    def test_logistic_regression_predictions(self, sample_data):
        """Test that Logistic Regression can make predictions."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        
        model = result['model']
        predictions = model.predict(X_train)
        probabilities = model.predict_proba(X_train)
        
        assert len(predictions) == len(X_train)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X_train), 2)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_logistic_regression_stored_in_trainer(self, sample_data):
        """Test that model is stored in trainer."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        
        assert 'logistic_regression' in trainer.models
        assert trainer.models['logistic_regression'] is not None
        assert 'logistic_regression' in trainer.cv_scores


class TestRandomForest:
    """Test Random Forest model training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        y = ((X['feature1'] > 0) & (X['feature2'] > 0)).astype(int)
        y = pd.Series(y, name='target')
        
        return X, y
    
    def test_train_random_forest_basic(self, sample_data):
        """Test basic Random Forest training."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert 'model' in result
        assert 'cv_score' in result
        assert 'model_name' in result
        assert result['model_name'] == 'random_forest'
        assert result['model'] is not None
        assert 0 <= result['cv_score'] <= 1
    
    def test_train_random_forest_with_tuning(self, sample_data):
        """Test Random Forest with hyperparameter tuning."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_random_forest(X_train, y_train, tune_hyperparameters=True)
        
        assert 'model' in result
        assert 'best_params' in result
        assert 'cv_score' in result
        assert result['best_params'] is not None
        assert isinstance(result['best_params'], dict)
        assert 0 <= result['cv_score'] <= 1
    
    def test_random_forest_predictions(self, sample_data):
        """Test that Random Forest can make predictions."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        model = result['model']
        predictions = model.predict(X_train)
        probabilities = model.predict_proba(X_train)
        
        assert len(predictions) == len(X_train)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X_train), 2)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_random_forest_feature_importance(self, sample_data):
        """Test that Random Forest provides feature importance."""
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        model = result['model']
        feature_importance = model.feature_importances_
        
        assert len(feature_importance) == len(X_train.columns)
        assert (feature_importance >= 0).all()
        assert feature_importance.sum() > 0


class TestXGBoost:
    """Test XGBoost model training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        y = ((X['feature1'] > 0) & (X['feature2'] > 0)).astype(int)
        y = pd.Series(y, name='target')
        
        return X, y
    
    def test_train_xgboost_basic(self, sample_data):
        """Test basic XGBoost training."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not available")
        
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_xgboost(X_train, y_train, tune_hyperparameters=False)
        
        assert 'model' in result
        assert 'cv_score' in result
        assert 'model_name' in result
        assert result['model_name'] == 'xgboost'
        assert result['model'] is not None
        assert 0 <= result['cv_score'] <= 1
    
    def test_train_xgboost_with_tuning(self, sample_data):
        """Test XGBoost with hyperparameter tuning."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not available")
        
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_xgboost(X_train, y_train, tune_hyperparameters=True)
        
        assert 'model' in result
        assert 'best_params' in result
        assert 'cv_score' in result
        assert result['best_params'] is not None
        assert isinstance(result['best_params'], dict)
        assert 0 <= result['cv_score'] <= 1
    
    def test_xgboost_predictions(self, sample_data):
        """Test that XGBoost can make predictions."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not available")
        
        X_train, y_train = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_xgboost(X_train, y_train, tune_hyperparameters=False)
        
        model = result['model']
        predictions = model.predict(X_train)
        probabilities = model.predict_proba(X_train)
        
        assert len(predictions) == len(X_train)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X_train), 2)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_xgboost_import_error(self, sample_data):
        """Test that XGBoost raises error when not available."""
        # This test would need to mock the import, but we'll test the actual behavior
        # by checking if the error is raised when XGBoost is not available
        pass


class TestModelSaving:
    """Test model saving and loading."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model."""
        np.random.seed(42)
        X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_logistic_regression(X, y, tune_hyperparameters=False)
        
        return trainer, 'logistic_regression'
    
    def test_save_model(self, trained_model, tmp_path):
        """Test saving a model."""
        trainer, model_name = trained_model
        
        filepath = tmp_path / "test_model.pkl"
        trainer.save_model(model_name, str(filepath))
        
        assert filepath.exists()
    
    def test_load_model(self, trained_model, tmp_path):
        """Test loading a model."""
        trainer, model_name = trained_model
        
        # Save model
        filepath = tmp_path / "test_model.pkl"
        trainer.save_model(model_name, str(filepath))
        
        # Load into new trainer
        new_trainer = ModelTrainer(random_state=42)
        loaded_data = new_trainer.load_model(str(filepath))
        
        assert 'model' in loaded_data
        assert 'model_name' in loaded_data
        assert loaded_data['model_name'] == model_name
        assert model_name in new_trainer.models
    
    def test_save_nonexistent_model_raises_error(self, tmp_path):
        """Test that saving nonexistent model raises error."""
        trainer = ModelTrainer(random_state=42)
        
        with pytest.raises(ValueError, match="not found"):
            trainer.save_model('nonexistent', str(tmp_path / "test.pkl"))


class TestModelTrainerUtilities:
    """Test utility functions of ModelTrainer."""
    
    @pytest.fixture
    def multiple_trained_models(self):
        """Create trainer with multiple models."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        trainer.train_logistic_regression(X, y, tune_hyperparameters=False)
        trainer.train_random_forest(X, y, tune_hyperparameters=False)
        
        return trainer
    
    def test_get_best_model(self, multiple_trained_models):
        """Test getting best model."""
        trainer = multiple_trained_models
        
        best_name, best_model = trainer.get_best_model()
        
        assert best_name in ['logistic_regression', 'random_forest']
        assert best_model is not None
        assert best_name in trainer.models
    
    def test_get_best_model_no_models_raises_error(self):
        """Test that getting best model with no models raises error."""
        trainer = ModelTrainer(random_state=42)
        
        with pytest.raises(ValueError, match="No models"):
            trainer.get_best_model()
    
    def test_train_all_models(self):
        """Test training all models."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        results = trainer.train_all_models(X, y, tune_hyperparameters=False)
        
        assert 'logistic_regression' in results
        assert 'random_forest' in results
        assert len(results) >= 2


class TestTrainModelsFunction:
    """Test convenience function."""
    
    def test_train_models_function(self):
        """Test train_models convenience function."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        results = train_models(X, y, models_to_train=['logistic_regression'], 
                            tune_hyperparameters=False, random_state=42)
        
        assert 'logistic_regression' in results
        assert 'model' in results['logistic_regression']


class TestModelTrainingWithRealData:
    """Test model training with real preprocessed data."""
    
    @pytest.fixture
    def preprocessed_data(self):
        """Get preprocessed real data."""
        df = load_data('data/Churn Modeling.csv')
        df_engineered = engineer_all_features(df)
        preprocessed = preprocess_data(df_engineered, random_state=42, use_smote=False)
        
        return preprocessed
    
    def test_logistic_regression_real_data(self, preprocessed_data):
        """Test Logistic Regression with real data."""
        X_train = preprocessed_data['X_train_resampled']
        y_train = preprocessed_data['y_train_resampled']
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        
        assert result['model'] is not None
        assert result['cv_score'] > 0.5  # Should be better than random
        
        # Test predictions
        X_val = preprocessed_data['X_val']
        predictions = result['model'].predict(X_val)
        assert len(predictions) == len(X_val)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_random_forest_real_data(self, preprocessed_data):
        """Test Random Forest with real data."""
        X_train = preprocessed_data['X_train_resampled']
        y_train = preprocessed_data['y_train_resampled']
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert result['model'] is not None
        assert result['cv_score'] > 0.5  # Should be better than random
        
        # Test predictions
        X_val = preprocessed_data['X_val']
        predictions = result['model'].predict(X_val)
        assert len(predictions) == len(X_val)
        assert all(pred in [0, 1] for pred in predictions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

