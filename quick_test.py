"""Quick health check for the project."""
import sys

def test_imports():
    print("Testing imports...")
    try:
        from src.utils import load_data
        from src.feature_engineering import engineer_all_features
        from src.data_preprocessing import preprocess_data
        from src.model_training import ModelTrainer
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_data_loading():
    print("\nTesting data loading...")
    try:
        from src.utils import load_data
        df = load_data('data/Churn Modeling.csv')
        print(f"[OK] Data loaded: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"[FAIL] Data loading error: {e}")
        return False

def test_pipeline():
    print("\nTesting complete pipeline...")
    try:
        from src.utils import load_data
        from src.feature_engineering import engineer_all_features
        from src.data_preprocessing import preprocess_data
        from src.model_training import ModelTrainer
        
        df = load_data('data/Churn Modeling.csv')
        df_eng = engineer_all_features(df)
        preprocessed = preprocess_data(df_eng, random_state=42, use_smote=False)
        trainer = ModelTrainer(random_state=42, cv_folds=3)
        result = trainer.train_logistic_regression(
            preprocessed['X_train_resampled'],
            preprocessed['y_train_resampled'],
            tune_hyperparameters=False
        )
        print(f"[OK] Pipeline works! Model CV score: {result['cv_score']:.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("PROJECT HEALTH CHECK")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_data_loading())
    results.append(test_pipeline())
    
    print("\n" + "=" * 60)
    if all(results):
        print("[SUCCESS] ALL CHECKS PASSED!")
        sys.exit(0)
    else:
        print("[ERROR] SOME CHECKS FAILED")
        sys.exit(1)

