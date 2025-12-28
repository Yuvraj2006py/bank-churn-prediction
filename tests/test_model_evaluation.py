"""
Comprehensive test suite for model evaluation module.
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
from src.model_evaluation import ModelEvaluator, evaluate_models
from src.model_training import ModelTrainer
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_all_features
from src.utils import load_data


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model."""
        X, y = sample_data
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_logistic_regression(X, y, tune_hyperparameters=False)
        
        return result['model']
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.evaluation_results == {}
        assert evaluator.models == {}
        assert evaluator.model_names == []
    
    def test_evaluate_model(self, sample_data, trained_model):
        """Test model evaluation."""
        X, y = sample_data
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(trained_model, X, y, model_name='test_model')
        
        assert 'model_name' in results
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        assert 'confusion_matrix' in results
        assert results['model_name'] == 'test_model'
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
        assert 0 <= results['roc_auc'] <= 1
    
    def test_evaluate_model_stores_results(self, sample_data, trained_model):
        """Test that evaluation stores results."""
        X, y = sample_data
        
        evaluator = ModelEvaluator()
        evaluator.evaluate_model(trained_model, X, y, model_name='test_model')
        
        assert 'test_model' in evaluator.evaluation_results
        assert 'test_model' in evaluator.models
        assert 'test_model' in evaluator.model_names
    
    def test_evaluate_model_confusion_matrix(self, sample_data, trained_model):
        """Test confusion matrix calculation."""
        X, y = sample_data
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(trained_model, X, y, model_name='test_model')
        
        cm = results['confusion_matrix']
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y)
        assert results['tn'] + results['fp'] + results['fn'] + results['tp'] == len(y)
    
    def test_evaluate_model_predictions(self, sample_data, trained_model):
        """Test that predictions are stored."""
        X, y = sample_data
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(trained_model, X, y, model_name='test_model')
        
        assert 'y_true' in results
        assert 'y_pred' in results
        assert 'y_pred_proba' in results
        assert len(results['y_true']) == len(y)
        assert len(results['y_pred']) == len(y)
        assert len(results['y_pred_proba']) == len(y)
        assert all(pred in [0, 1] for pred in results['y_pred'])
        assert all(0 <= prob <= 1 for prob in results['y_pred_proba'])


class TestVisualizations:
    """Test visualization functions."""
    
    @pytest.fixture
    def evaluator_with_results(self):
        """Create evaluator with evaluation results."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_logistic_regression(X, y, tune_hyperparameters=False)
        model = result['model']
        
        evaluator = ModelEvaluator()
        evaluator.evaluate_model(model, X, y, model_name='test_model')
        
        return evaluator, X.columns.tolist()
    
    def test_plot_confusion_matrix(self, evaluator_with_results):
        """Test confusion matrix plotting."""
        evaluator, _ = evaluator_with_results
        
        fig = evaluator.plot_confusion_matrix('test_model')
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_confusion_matrix_normalized(self, evaluator_with_results):
        """Test normalized confusion matrix plotting."""
        evaluator, _ = evaluator_with_results
        
        fig = evaluator.plot_confusion_matrix('test_model', normalize=True)
        assert fig is not None
    
    def test_plot_confusion_matrix_nonexistent_model(self, evaluator_with_results):
        """Test that plotting nonexistent model raises error."""
        evaluator, _ = evaluator_with_results
        
        with pytest.raises(ValueError, match="not found"):
            evaluator.plot_confusion_matrix('nonexistent')
    
    def test_plot_roc_curve(self, evaluator_with_results):
        """Test ROC curve plotting."""
        evaluator, _ = evaluator_with_results
        
        fig = evaluator.plot_roc_curve(['test_model'])
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_precision_recall_curve(self, evaluator_with_results):
        """Test Precision-Recall curve plotting."""
        evaluator, _ = evaluator_with_results
        
        fig = evaluator.plot_precision_recall_curve(['test_model'])
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_feature_importance(self, evaluator_with_results):
        """Test feature importance plotting."""
        evaluator, feature_names = evaluator_with_results
        
        # Train a Random Forest for feature importance
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_random_forest(X, y, tune_hyperparameters=False)
        model = result['model']
        
        evaluator.evaluate_model(model, X, y, model_name='rf_model')
        
        fig = evaluator.plot_feature_importance('rf_model', feature_names)
        assert fig is not None
        assert len(fig.axes) > 0
    
    def test_plot_feature_importance_linear_model(self, evaluator_with_results):
        """Test feature importance for linear models."""
        evaluator, feature_names = evaluator_with_results
        
        fig = evaluator.plot_feature_importance('test_model', feature_names)
        assert fig is not None


class TestModelComparison:
    """Test model comparison functions."""
    
    @pytest.fixture
    def evaluator_with_multiple_models(self):
        """Create evaluator with multiple models."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        evaluator = ModelEvaluator()
        
        # Train and evaluate Logistic Regression
        trainer1 = ModelTrainer(random_state=42)
        result1 = trainer1.train_logistic_regression(X, y, tune_hyperparameters=False)
        evaluator.evaluate_model(result1['model'], X, y, model_name='logistic_regression')
        
        # Train and evaluate Random Forest
        trainer2 = ModelTrainer(random_state=42)
        result2 = trainer2.train_random_forest(X, y, tune_hyperparameters=False)
        evaluator.evaluate_model(result2['model'], X, y, model_name='random_forest')
        
        return evaluator
    
    def test_compare_models(self, evaluator_with_multiple_models):
        """Test model comparison."""
        evaluator = evaluator_with_multiple_models
        
        comparison_df = evaluator.compare_models()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) >= 2
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns
        assert 'ROC-AUC' in comparison_df.columns
    
    def test_compare_models_custom_metric(self, evaluator_with_multiple_models):
        """Test model comparison with custom metric."""
        evaluator = evaluator_with_multiple_models
        
        comparison_df = evaluator.compare_models(metric='f1_score')
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) >= 2
    
    def test_get_best_model(self, evaluator_with_multiple_models):
        """Test getting best model."""
        evaluator = evaluator_with_multiple_models
        
        best_name, best_results = evaluator.get_best_model(metric='roc_auc')
        
        assert best_name in ['logistic_regression', 'random_forest']
        assert 'roc_auc' in best_results
        assert best_results['roc_auc'] >= 0
    
    def test_get_best_model_no_models(self):
        """Test that getting best model with no models raises error."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="No models"):
            evaluator.get_best_model()
    
    def test_generate_report(self, evaluator_with_multiple_models):
        """Test report generation."""
        evaluator = evaluator_with_multiple_models
        
        report = evaluator.generate_report('logistic_regression')
        
        assert isinstance(report, str)
        assert 'logistic_regression' in report
        assert 'Accuracy' in report
        assert 'ROC-AUC' in report
        assert 'CONFUSION MATRIX' in report
    
    def test_generate_report_nonexistent_model(self, evaluator_with_multiple_models):
        """Test that generating report for nonexistent model raises error."""
        evaluator = evaluator_with_multiple_models
        
        with pytest.raises(ValueError, match="not found"):
            evaluator.generate_report('nonexistent')


class TestEvaluateModelsFunction:
    """Test convenience function."""
    
    def test_evaluate_models_function(self):
        """Test evaluate_models convenience function."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y = pd.Series((X['feature1'] > 0).astype(int), name='target')
        
        trainer = ModelTrainer(random_state=42)
        result = trainer.train_logistic_regression(X, y, tune_hyperparameters=False)
        
        models = {'logistic_regression': result['model']}
        
        evaluator = evaluate_models(models, X, y)
        
        assert isinstance(evaluator, ModelEvaluator)
        assert 'logistic_regression' in evaluator.evaluation_results
        assert 'logistic_regression' in evaluator.models


class TestEvaluationWithRealData:
    """Test evaluation with real preprocessed data."""
    
    @pytest.fixture
    def preprocessed_data(self):
        """Get preprocessed real data."""
        df = load_data('data/Churn Modeling.csv')
        df_engineered = engineer_all_features(df)
        preprocessed = preprocess_data(df_engineered, random_state=42, use_smote=False)
        
        return preprocessed
    
    def test_evaluate_logistic_regression_real_data(self, preprocessed_data):
        """Test evaluating Logistic Regression with real data."""
        X_train = preprocessed_data['X_train_resampled']
        y_train = preprocessed_data['y_train_resampled']
        X_test = preprocessed_data['X_test']
        y_test = preprocessed_data['y_test']
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        model = result['model']
        
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate_model(model, X_test, y_test, model_name='logistic_regression')
        
        assert eval_results['accuracy'] > 0.5
        assert eval_results['roc_auc'] > 0.5
        assert eval_results['f1_score'] >= 0
    
    def test_evaluate_multiple_models_real_data(self, preprocessed_data):
        """Test evaluating multiple models with real data."""
        X_train = preprocessed_data['X_train_resampled']
        y_train = preprocessed_data['y_train_resampled']
        X_test = preprocessed_data['X_test']
        y_test = preprocessed_data['y_test']
        
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        
        # Train multiple models
        result_lr = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        result_rf = trainer.train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        models = {
            'logistic_regression': result_lr['model'],
            'random_forest': result_rf['model']
        }
        
        evaluator = evaluate_models(models, X_test, y_test)
        
        assert len(evaluator.evaluation_results) == 2
        assert 'logistic_regression' in evaluator.evaluation_results
        assert 'random_forest' in evaluator.evaluation_results
        
        # Compare models
        comparison = evaluator.compare_models()
        assert len(comparison) == 2
        
        # Get best model
        best_name, best_results = evaluator.get_best_model()
        assert best_name in ['logistic_regression', 'random_forest']
        assert best_results['roc_auc'] > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

