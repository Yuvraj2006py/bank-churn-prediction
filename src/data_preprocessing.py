"""
Data Preprocessing Module for Bank Churn Prediction.

This module handles:
- Categorical encoding
- Feature scaling
- Data splitting
- Class imbalance handling
- Preprocessing pipeline management
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, Optional, List, Dict, Any
import joblib
import warnings

# Optional import for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None

warnings.filterwarnings('ignore')


def sanitize_feature_names(feature_names: List[str]) -> List[str]:
    """
    Sanitize feature names to be compatible with XGBoost and other ML libraries.
    
    XGBoost doesn't allow feature names with: [, ], <, >
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
        
    Returns:
    --------
    list
        List of sanitized feature names
    """
    sanitized = []
    for name in feature_names:
        # Replace invalid characters (XGBoost doesn't allow: [, ], <, >)
        sanitized_name = str(name).replace('[', '_lb_').replace(']', '_rb_')
        sanitized_name = sanitized_name.replace('<', '_lt_').replace('>', '_gt_')
        sanitized_name = sanitized_name.replace('+', '_plus_')
        sanitized_name = sanitized_name.replace(' ', '_').replace('-', '_')
        sanitized_name = sanitized_name.replace('(', '_').replace(')', '_')
        sanitized_name = sanitized_name.replace('/', '_div_').replace('*', '_mul_')
        sanitized.append(sanitized_name)
    return sanitized


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for churn prediction.
    
    Handles encoding, scaling, splitting, and imbalance correction.
    """
    
    def __init__(self, 
                 categorical_cols: Optional[List[str]] = None,
                 numeric_cols: Optional[List[str]] = None,
                 target_col: str = 'Exited',
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 use_smote: bool = True):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        categorical_cols : list, optional
            List of categorical column names. If None, will auto-detect.
        numeric_cols : list, optional
            List of numeric column names. If None, will auto-detect.
        target_col : str
            Name of the target column
        test_size : float
            Proportion of data for test set
        val_size : float
            Proportion of remaining data for validation set (after test split)
        random_state : int
            Random seed for reproducibility
        use_smote : bool
            Whether to use SMOTE for class imbalance
        """
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_smote = use_smote
        
        # Preprocessing transformers
        self.preprocessor = None
        self.scaler = None
        self.label_encoders = {}
        self.smote = None
        self.feature_names = []
        
    def _detect_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Auto-detect categorical and numeric columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (categorical_cols, numeric_cols)
        """
        # Exclude target and ID columns
        exclude_cols = [self.target_col, 'RowNumber', 'CustomerId', 'Surname']
        
        if not self.categorical_cols:
            # Auto-detect categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            self.categorical_cols = [col for col in cat_cols if col not in exclude_cols]
        
        if not self.numeric_cols:
            # Auto-detect numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.numeric_cols = [col for col in num_cols if col not in exclude_cols + [self.target_col]]
        
        return self.categorical_cols, self.numeric_cols
    
    def _remove_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ID columns that are not useful for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with ID columns removed
        """
        id_cols = ['RowNumber', 'CustomerId', 'Surname']
        cols_to_remove = [col for col in id_cols if col in df.columns]
        return df.drop(columns=cols_to_remove)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        # Remove ID columns
        df_clean = self._remove_id_columns(df.copy())
        
        # Detect columns if not specified
        self._detect_columns(df_clean)
        
        # Separate features and target
        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]
        
        # Create preprocessing pipeline
        transformers = []
        
        # One-hot encode categorical variables
        if self.categorical_cols:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                 self.categorical_cols)
            )
        
        # Scale numeric variables
        if self.numeric_cols:
            self.scaler = StandardScaler()
            transformers.append(
                ('num', self.scaler, self.numeric_cols)
            )
        
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            
            # Fit and transform
            X_transformed = self.preprocessor.fit_transform(X)
            
            # Get feature names
            feature_names = []
            if self.categorical_cols:
                # Get one-hot encoded feature names
                ohe = self.preprocessor.named_transformers_['cat']
                for i, col in enumerate(self.categorical_cols):
                    categories = ohe.categories_[i]
                    for cat in categories[1:]:  # Skip first category (drop='first')
                        feature_names.append(f"{col}_{cat}")
            
            if self.numeric_cols:
                feature_names.extend(self.numeric_cols)
            
            # Add any remaining columns
            remaining_cols = [col for col in X.columns 
                             if col not in self.categorical_cols + self.numeric_cols]
            feature_names.extend(remaining_cols)
            
            # Sanitize feature names for XGBoost compatibility
            self.feature_names = sanitize_feature_names(feature_names)
            
            # Convert to DataFrame
            X_transformed_df = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        else:
            X_transformed_df = X
        
        # Add target back
        X_transformed_df[self.target_col] = y.values
        
        return X_transformed_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform first.")
        
        # Remove ID columns
        df_clean = self._remove_id_columns(df.copy())
        
        # Separate features and target
        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]
        
        # Transform
        X_transformed = self.preprocessor.transform(X)
        
        # Convert to DataFrame
        X_transformed_df = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        
        # Add target back
        X_transformed_df[self.target_col] = y.values
        
        return X_transformed_df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                      pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe (should be preprocessed)
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target series
            
        Returns:
        --------
        tuple
            (X_resampled, y_resampled)
        """
        if not self.use_smote:
            return X, y
        
        if not SMOTE_AVAILABLE:
            warnings.warn("SMOTE not available. Install imbalanced-learn to use SMOTE. Returning original data.")
            return X, y
        
        if self.smote is None:
            self.smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled_series = pd.Series(y_resampled, name=y.name)
        
        return X_resampled_df, y_resampled_series
    
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save preprocessor
        """
        preprocessor_dict = {
            'preprocessor': self.preprocessor,
            'scaler': self.scaler,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'target_col': self.target_col,
            'feature_names': self.feature_names,
            'smote': self.smote
        }
        joblib.dump(preprocessor_dict, filepath)
    
    def load_preprocessor(self, filepath: str):
        """
        Load preprocessor from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to load preprocessor from
        """
        preprocessor_dict = joblib.load(filepath)
        self.preprocessor = preprocessor_dict['preprocessor']
        self.scaler = preprocessor_dict['scaler']
        self.categorical_cols = preprocessor_dict['categorical_cols']
        self.numeric_cols = preprocessor_dict['numeric_cols']
        self.target_col = preprocessor_dict['target_col']
        self.feature_names = preprocessor_dict['feature_names']
        self.smote = preprocessor_dict['smote']


def preprocess_data(df: pd.DataFrame,
                   target_col: str = 'Exited',
                   test_size: float = 0.2,
                   val_size: float = 0.2,
                   random_state: int = 42,
                   use_smote: bool = True,
                   categorical_cols: Optional[List[str]] = None,
                   numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
    random_state : int
        Random seed
    use_smote : bool
        Whether to use SMOTE
    categorical_cols : list, optional
        Categorical columns
    numeric_cols : list, optional
        Numeric columns
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'preprocessor': DataPreprocessor instance
        - 'X_train': Training features
        - 'X_val': Validation features
        - 'X_test': Test features
        - 'y_train': Training target
        - 'y_val': Validation target
        - 'y_test': Test target
        - 'X_train_resampled': Resampled training features (if SMOTE used)
        - 'y_train_resampled': Resampled training target (if SMOTE used)
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        use_smote=use_smote
    )
    
    # Fit and transform
    df_transformed = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_transformed)
    
    # Handle imbalance
    X_train_resampled, y_train_resampled = preprocessor.handle_imbalance(X_train, y_train)
    
    return {
        'preprocessor': preprocessor,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled
    }

