"""
Model Training Module for Bank Churn Prediction.

This module provides:
- Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning with GridSearchCV
- Model training and persistence
- Cross-validation support
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from typing import Dict, Any, Optional, Tuple, List
import joblib
import warnings
import os

# Optional import for XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model training class for churn prediction.
    
    Supports multiple algorithms with hyperparameter tuning.
    """
    
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        cv_folds : int
            Number of folds for cross-validation
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.training_history = {}
    
    def train_logistic_regression(self,
                                  X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  tune_hyperparameters: bool = True,
                                  cv: Optional[int] = None) -> Dict[str, Any]:
        """
        Train Logistic Regression model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        cv : int, optional
            Number of CV folds (overrides default)
            
        Returns:
        --------
        dict
            Dictionary with model, best params, and CV scores
        """
        print("Training Logistic Regression...")
        
        cv_folds = cv or self.cv_folds
        
        if tune_hyperparameters:
            # Define parameter grid
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            }
            
            # Create base model
            base_model = LogisticRegression(random_state=self.random_state, class_weight='balanced')
            
            # Grid search with cross-validation
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_splitter,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            print(f"  Best parameters: {best_params}")
            print(f"  Best CV score (ROC-AUC): {cv_score:.4f}")
        else:
            # Train without hyperparameter tuning
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=500,
                C=1.0,
                penalty='l2',
                solver='liblinear'
            )
            model.fit(X_train, y_train)
            best_params = None
            
            # Calculate CV score
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='roc_auc')
            cv_score = cv_scores.mean()
            print(f"  CV score (ROC-AUC): {cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store model and results
        self.models['logistic_regression'] = model
        self.best_params['logistic_regression'] = best_params
        self.cv_scores['logistic_regression'] = cv_score
        
        return {
            'model': model,
            'best_params': best_params,
            'cv_score': cv_score,
            'model_name': 'logistic_regression'
        }
    
    def train_random_forest(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           tune_hyperparameters: bool = True,
                           cv: Optional[int] = None) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        cv : int, optional
            Number of CV folds (overrides default)
            
        Returns:
        --------
        dict
            Dictionary with model, best params, and CV scores
        """
        print("Training Random Forest...")
        
        cv_folds = cv or self.cv_folds
        
        if tune_hyperparameters:
            # Define parameter grid (reduced for faster training)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', None]
            }
            
            # Create base model
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            
            # Grid search with cross-validation
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_splitter,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            print(f"  Best parameters: {best_params}")
            print(f"  Best CV score (ROC-AUC): {cv_score:.4f}")
        else:
            # Train without hyperparameter tuning
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            best_params = None
            
            # Calculate CV score
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='roc_auc')
            cv_score = cv_scores.mean()
            print(f"  CV score (ROC-AUC): {cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store model and results
        self.models['random_forest'] = model
        self.best_params['random_forest'] = best_params
        self.cv_scores['random_forest'] = cv_score
        
        return {
            'model': model,
            'best_params': best_params,
            'cv_score': cv_score,
            'model_name': 'random_forest'
        }
    
    def train_xgboost(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     tune_hyperparameters: bool = True,
                     cv: Optional[int] = None) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        cv : int, optional
            Number of CV folds (overrides default)
            
        Returns:
        --------
        dict
            Dictionary with model, best params, and CV scores
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install xgboost to use this model.")
        
        print("Training XGBoost...")
        
        # Sanitize feature names for XGBoost compatibility
        from src.data_preprocessing import sanitize_feature_names
        X_train = X_train.copy()
        X_train.columns = sanitize_feature_names(X_train.columns.tolist())
        
        cv_folds = cv or self.cv_folds
        
        if tune_hyperparameters:
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1, 2]  # For class imbalance
            }
            
            # Create base model
            base_model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            # Grid search with cross-validation
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_splitter,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            print(f"  Best parameters: {best_params}")
            print(f"  Best CV score (ROC-AUC): {cv_score:.4f}")
        else:
            # Train without hyperparameter tuning
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            model.fit(X_train, y_train)
            best_params = None
            
            # Calculate CV score
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='roc_auc')
            cv_score = cv_scores.mean()
            print(f"  CV score (ROC-AUC): {cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store model and results
        self.models['xgboost'] = model
        self.best_params['xgboost'] = best_params
        self.cv_scores['xgboost'] = cv_score
        
        return {
            'model': model,
            'best_params': best_params,
            'cv_score': cv_score,
            'model_name': 'xgboost'
        }
    
    def train_all_models(self,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         models_to_train: Optional[List[str]] = None,
                         tune_hyperparameters: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        models_to_train : list, optional
            List of model names to train. If None, trains all.
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        dict
            Dictionary with results for each model
        """
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest']
            if XGBOOST_AVAILABLE:
                models_to_train.append('xgboost')
        
        results = {}
        
        if 'logistic_regression' in models_to_train:
            results['logistic_regression'] = self.train_logistic_regression(
                X_train, y_train, tune_hyperparameters
            )
        
        if 'random_forest' in models_to_train:
            results['random_forest'] = self.train_random_forest(
                X_train, y_train, tune_hyperparameters
            )
        
        if 'xgboost' in models_to_train:
            if XGBOOST_AVAILABLE:
                results['xgboost'] = self.train_xgboost(
                    X_train, y_train, tune_hyperparameters
                )
            else:
                print("Warning: XGBoost not available, skipping...")
        
        return results
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filepath : str
            Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model_data = {
            'model': self.models[model_name],
            'best_params': self.best_params.get(model_name),
            'cv_score': self.cv_scores.get(model_name),
            'model_name': model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
            
        Returns:
        --------
        dict
            Dictionary with loaded model data
        """
        model_data = joblib.load(filepath)
        
        model_name = model_data.get('model_name', 'unknown')
        self.models[model_name] = model_data['model']
        self.best_params[model_name] = model_data.get('best_params')
        self.cv_scores[model_name] = model_data.get('cv_score')
        
        print(f"Model '{model_name}' loaded from {filepath}")
        return model_data
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best model based on CV scores.
        
        Returns:
        --------
        tuple
            (model_name, model)
        """
        if not self.cv_scores:
            raise ValueError("No models have been trained yet.")
        
        best_model_name = max(self.cv_scores, key=self.cv_scores.get)
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model


def train_models(X_train: pd.DataFrame,
                y_train: pd.Series,
                models_to_train: Optional[List[str]] = None,
                tune_hyperparameters: bool = True,
                random_state: int = 42,
                cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to train multiple models.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    models_to_train : list, optional
        List of model names to train
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
    random_state : int
        Random seed
    cv_folds : int
        Number of CV folds
        
    Returns:
    --------
    dict
        Dictionary with results for each model
    """
    trainer = ModelTrainer(random_state=random_state, cv_folds=cv_folds)
    return trainer.train_all_models(X_train, y_train, models_to_train, tune_hyperparameters)

