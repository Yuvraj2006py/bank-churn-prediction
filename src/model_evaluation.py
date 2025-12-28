"""
Model Evaluation Module for Bank Churn Prediction.

This module provides:
- Comprehensive evaluation metrics
- Visualization functions (ROC curves, confusion matrices, PR curves)
- Feature importance analysis
- Model comparison and selection
- Evaluation reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Optional, List, Tuple
import warnings
import os

# Set matplotlib backend for testing compatibility
try:
    current_backend = plt.get_backend()
    if ('PYTEST_CURRENT_TEST' in os.environ or 
        'pytest' in os.environ.get('_', '').lower() or
        current_backend == 'TkAgg' or 
        'tk' in current_backend.lower()):
        import matplotlib
        matplotlib.use('Agg', force=True)
except Exception:
    pass

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Comprehensive model evaluation class for churn prediction.
    
    Provides metrics, visualizations, and comparisons.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        self.models = {}
        self.model_names = []
    
    def evaluate_model(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      model_name: str = 'model',
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate a single model and return comprehensive metrics.
        
        Parameters:
        -----------
        model : Any
            Trained model with predict and predict_proba methods
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            True target values
        model_name : str
            Name identifier for the model
        threshold : float
            Probability threshold for binary classification
            
        Returns:
        --------
        dict
            Dictionary with all evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
        except ValueError:
            roc_auc = 0.0
        
        # Precision-Recall AUC
        try:
            pr_auc = average_precision_score(y, y_pred_proba)
        except ValueError:
            pr_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        
        # Metrics with threshold
        accuracy_thresh = accuracy_score(y, y_pred_threshold)
        precision_thresh = precision_score(y, y_pred_threshold, zero_division=0)
        recall_thresh = recall_score(y, y_pred_threshold, zero_division=0)
        f1_thresh = f1_score(y, y_pred_threshold, zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': cm,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'accuracy_threshold': accuracy_thresh,
            'precision_threshold': precision_thresh,
            'recall_threshold': recall_thresh,
            'f1_threshold': f1_thresh,
            'y_true': y.values,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'threshold': threshold
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        self.models[model_name] = model
        if model_name not in self.model_names:
            self.model_names.append(model_name)
        
        return results
    
    def plot_confusion_matrix(self,
                             model_name: str,
                             figsize: Tuple[int, int] = (8, 6),
                             normalize: bool = False) -> plt.Figure:
        """
        Plot confusion matrix for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot
        figsize : tuple
            Figure size
        normalize : bool
            Whether to normalize the confusion matrix
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model '{model_name}' not found in evaluation results.")
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        plt.tight_layout()
        
        return fig
    
    def plot_roc_curve(self,
                      model_names: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ROC curves for one or more models.
        
        Parameters:
        -----------
        model_names : list, optional
            List of model names to plot. If None, plots all models.
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if model_names is None:
            model_names = self.model_names
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
            
            results = self.evaluation_results[model_name]
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = results['roc_auc']
            
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_precision_recall_curve(self,
                                   model_names: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot Precision-Recall curves for one or more models.
        
        Parameters:
        -----------
        model_names : list, optional
            List of model names to plot. If None, plots all models.
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if model_names is None:
            model_names = self.model_names
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
            
            results = self.evaluation_results[model_name]
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = results['pr_auc']
            
            ax.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})")
        
        # Baseline (random classifier)
        baseline = np.sum(results['y_true']) / len(results['y_true'])
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self,
                               model_name: str,
                               feature_names: List[str],
                               top_n: int = 15,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model '{model_name}' does not support feature importance.")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                      color=sns.color_palette('viridis', len(importance_df)))
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {model_name}')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        return fig
    
    def compare_models(self,
                      metric: str = 'roc_auc',
                      ascending: bool = False) -> pd.DataFrame:
        """
        Compare all evaluated models by a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to compare (e.g., 'roc_auc', 'f1_score', 'accuracy')
        ascending : bool
            Whether to sort in ascending order
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name in self.model_names:
            if model_name not in self.evaluation_results:
                continue
            
            results = self.evaluation_results[model_name]
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'PR-AUC': results['pr_auc'],
                'Specificity': results['specificity'],
                'Sensitivity': results['sensitivity']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric, ascending=ascending)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best model based on a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
            
        Returns:
        --------
        tuple
            (model_name, evaluation_results)
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet.")
        
        best_model_name = None
        best_score = -np.inf if metric in ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1_score'] else np.inf
        
        for model_name, results in self.evaluation_results.items():
            if metric not in results:
                continue
            
            score = results[metric]
            if metric in ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1_score']:
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Could not determine best model for metric '{metric}'.")
        
        return best_model_name, self.evaluation_results[best_model_name]
    
    def generate_report(self, model_name: str) -> str:
        """
        Generate a text report for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        str
            Formatted report string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model '{model_name}' not found in evaluation results.")
        
        results = self.evaluation_results[model_name]
        model = self.models[model_name]
        
        report = f"""
{'=' * 60}
MODEL EVALUATION REPORT: {model_name}
{'=' * 60}

PERFORMANCE METRICS:
  Accuracy:        {results['accuracy']:.4f}
  Precision:       {results['precision']:.4f}
  Recall:          {results['recall']:.4f}
  F1-Score:        {results['f1_score']:.4f}
  ROC-AUC:         {results['roc_auc']:.4f}
  PR-AUC:          {results['pr_auc']:.4f}
  Specificity:     {results['specificity']:.4f}
  Sensitivity:     {results['sensitivity']:.4f}

CONFUSION MATRIX:
  True Negatives:  {results['tn']}
  False Positives: {results['fp']}
  False Negatives: {results['fn']}
  True Positives:  {results['tp']}

THRESHOLD-BASED METRICS (threshold = {results['threshold']}):
  Accuracy:        {results['accuracy_threshold']:.4f}
  Precision:       {results['precision_threshold']:.4f}
  Recall:          {results['recall_threshold']:.4f}
  F1-Score:        {results['f1_threshold']:.4f}

MODEL TYPE: {type(model).__name__}
"""
        
        return report
    
    def print_comparison_table(self):
        """Print a formatted comparison table of all models."""
        comparison_df = self.compare_models()
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80 + "\n")


def evaluate_models(models: Dict[str, Any],
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    feature_names: Optional[List[str]] = None) -> ModelEvaluator:
    """
    Convenience function to evaluate multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of {model_name: model} pairs
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    feature_names : list, optional
        List of feature names for importance plots
        
    Returns:
    --------
    ModelEvaluator
        Evaluator with all results
    """
    evaluator = ModelEvaluator()
    
    for model_name, model in models.items():
        evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)
    
    return evaluator

