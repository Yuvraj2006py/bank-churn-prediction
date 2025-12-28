"""
Generate visualizations for documentation.

This script creates key visualizations from the dataset and model results
to include in the RESULTS.md documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Import project modules
from src.utils import load_data
from src.feature_engineering import engineer_all_features
from src.data_preprocessing import preprocess_data
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create images directory
images_dir = Path('docs/images')
images_dir.mkdir(parents=True, exist_ok=True)

print("Generating documentation visualizations...")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = load_data('data/Churn Modeling.csv')

# 1. Churn Distribution
print("2. Creating churn distribution chart...")
fig, ax = plt.subplots(figsize=(10, 6))
churn_counts = df['Exited'].value_counts()
colors = ['#27ae60', '#e74c3c']
bars = ax.bar(['Retained', 'Churned'], churn_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax.set_xlabel('Customer Status', fontsize=12, fontweight='bold')
ax.set_title('Customer Churn Distribution', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(images_dir / 'churn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: docs/images/churn_distribution.png")

# 2. Churn by Geography
print("3. Creating geography churn analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Churn rate by geography
geo_churn = df.groupby('Geography')['Exited'].agg(['count', 'sum', 'mean']).reset_index()
geo_churn.columns = ['Geography', 'Total', 'Churned', 'ChurnRate']

ax1.bar(geo_churn['Geography'], geo_churn['ChurnRate'], color=['#3498db', '#e74c3c', '#f39c12'], 
        alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Churn Rate', fontsize=12, fontweight='bold')
ax1.set_xlabel('Geography', fontsize=12, fontweight='bold')
ax1.set_title('Churn Rate by Geography', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(geo_churn['ChurnRate']) * 1.2])
ax1.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (geo, rate) in enumerate(zip(geo_churn['Geography'], geo_churn['ChurnRate'])):
    ax1.text(i, rate, f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Count by geography
ax2.bar(geo_churn['Geography'], geo_churn['Total'], color=['#3498db', '#e74c3c', '#f39c12'], 
        alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax2.set_xlabel('Geography', fontsize=12, fontweight='bold')
ax2.set_title('Customer Count by Geography', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (geo, count) in enumerate(zip(geo_churn['Geography'], geo_churn['Total'])):
    ax2.text(i, count, f'{int(count):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(images_dir / 'geography_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: docs/images/geography_churn.png")

# 3. Age Distribution and Churn
print("4. Creating age analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Age distribution
ax1.hist(df[df['Exited'] == 0]['Age'], bins=20, alpha=0.7, label='Retained', color='#27ae60', edgecolor='black')
ax1.hist(df[df['Exited'] == 1]['Age'], bins=20, alpha=0.7, label='Churned', color='#e74c3c', edgecolor='black')
ax1.set_xlabel('Age', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax1.set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Churn rate by age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
age_churn = df.groupby('AgeGroup')['Exited'].mean().reset_index()
age_churn.columns = ['AgeGroup', 'ChurnRate']

bars = ax2.bar(age_churn['AgeGroup'].astype(str), age_churn['ChurnRate'], 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Churn Rate', fontsize=12, fontweight='bold')
ax2.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax2.set_title('Churn Rate by Age Group', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(images_dir / 'age_churn_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: docs/images/age_churn_analysis.png")

# 4. Balance Analysis
print("5. Creating balance analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Balance distribution (log scale for better visualization)
retained_balance = df[df['Exited'] == 0]['Balance']
churned_balance = df[df['Exited'] == 1]['Balance']

ax1.hist(retained_balance[retained_balance > 0], bins=30, alpha=0.7, label='Retained', 
         color='#27ae60', edgecolor='black')
ax1.hist(churned_balance[churned_balance > 0], bins=30, alpha=0.7, label='Churned', 
         color='#e74c3c', edgecolor='black')
ax1.set_xlabel('Balance ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax1.set_title('Balance Distribution (Non-Zero)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Zero balance vs non-zero balance churn
df['HasBalance'] = (df['Balance'] > 0).astype(int)
balance_churn = df.groupby('HasBalance')['Exited'].agg(['count', 'mean']).reset_index()
balance_churn.columns = ['HasBalance', 'Count', 'ChurnRate']
balance_churn['Label'] = balance_churn['HasBalance'].map({0: 'Zero Balance', 1: 'Has Balance'})

bars = ax2.bar(balance_churn['Label'], balance_churn['ChurnRate'], 
               color=['#e74c3c', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Churn Rate', fontsize=12, fontweight='bold')
ax2.set_xlabel('Balance Status', fontsize=12, fontweight='bold')
ax2.set_title('Churn Rate by Balance Status', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(images_dir / 'balance_churn_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: docs/images/balance_churn_analysis.png")

# 5. Engagement Metrics
print("6. Creating engagement analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Active Member
active_churn = df.groupby('IsActiveMember')['Exited'].agg(['count', 'mean']).reset_index()
active_churn.columns = ['IsActiveMember', 'Count', 'ChurnRate']
active_churn['Label'] = active_churn['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})

bars = axes[0, 0].bar(active_churn['Label'], active_churn['ChurnRate'], 
                      color=['#e74c3c', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Churn Rate', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Member Status', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Churn Rate by Active Member Status', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Number of Products
product_churn = df.groupby('NumOfProducts')['Exited'].agg(['count', 'mean']).reset_index()
product_churn.columns = ['NumOfProducts', 'Count', 'ChurnRate']

bars = axes[0, 1].bar(product_churn['NumOfProducts'].astype(str), product_churn['ChurnRate'], 
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Churn Rate', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Number of Products', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Churn Rate by Number of Products', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Tenure
tenure_churn = df.groupby('Tenure')['Exited'].mean().reset_index()
tenure_churn.columns = ['Tenure', 'ChurnRate']

axes[1, 0].plot(tenure_churn['Tenure'], tenure_churn['ChurnRate'], 
                marker='o', linewidth=2, markersize=8, color='#e74c3c')
axes[1, 0].set_ylabel('Churn Rate', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Tenure (Years)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Churn Rate by Tenure', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].fill_between(tenure_churn['Tenure'], tenure_churn['ChurnRate'], alpha=0.3, color='#e74c3c')

# Credit Score
credit_bins = pd.cut(df['CreditScore'], bins=10)
credit_churn = df.groupby(credit_bins)['Exited'].mean().reset_index()
credit_churn['CreditScore_Mid'] = credit_churn['CreditScore'].apply(lambda x: x.mid)

axes[1, 1].plot(credit_churn['CreditScore_Mid'], credit_churn['Exited'], 
                marker='o', linewidth=2, markersize=8, color='#3498db')
axes[1, 1].set_ylabel('Churn Rate', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Credit Score (Midpoint)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Churn Rate by Credit Score', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].fill_between(credit_churn['CreditScore_Mid'], credit_churn['Exited'], alpha=0.3, color='#3498db')

plt.tight_layout()
plt.savefig(images_dir / 'engagement_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: docs/images/engagement_metrics.png")

# 6. Model Performance Comparison (if models are available)
print("7. Creating model performance comparison...")
try:
    # Load and evaluate models
    df_eng = engineer_all_features(df)
    preprocessed = preprocess_data(df_eng, random_state=42, use_smote=False)
    
    trainer = ModelTrainer(random_state=42, cv_folds=3)
    
    # Train models (quick, no tuning)
    result_lr = trainer.train_logistic_regression(
        preprocessed['X_train_resampled'],
        preprocessed['y_train_resampled'],
        tune_hyperparameters=False
    )
    
    result_rf = trainer.train_random_forest(
        preprocessed['X_train_resampled'],
        preprocessed['y_train_resampled'],
        tune_hyperparameters=False
    )
    
    # Evaluate
    evaluator = ModelEvaluator()
    eval_lr = evaluator.evaluate_model(
        result_lr['model'],
        preprocessed['X_test'],
        preprocessed['y_test'],
        model_name='logistic_regression'
    )
    
    eval_rf = evaluator.evaluate_model(
        result_rf['model'],
        preprocessed['X_test'],
        preprocessed['y_test'],
        model_name='random_forest'
    )
    
    # Try XGBoost
    try:
        result_xgb = trainer.train_xgboost(
            preprocessed['X_train_resampled'],
            preprocessed['y_train_resampled'],
            tune_hyperparameters=False
        )
        eval_xgb = evaluator.evaluate_model(
            result_xgb['model'],
            preprocessed['X_test'],
            preprocessed['y_test'],
            model_name='xgboost'
        )
        models_data = {
            'Logistic Regression': [eval_lr['accuracy'], eval_lr['precision'], eval_lr['recall'], eval_lr['f1_score'], eval_lr['roc_auc']],
            'Random Forest': [eval_rf['accuracy'], eval_rf['precision'], eval_rf['recall'], eval_rf['f1_score'], eval_rf['roc_auc']],
            'XGBoost': [eval_xgb['accuracy'], eval_xgb['precision'], eval_xgb['recall'], eval_xgb['f1_score'], eval_xgb['roc_auc']]
        }
    except:
        models_data = {
            'Logistic Regression': [eval_lr['accuracy'], eval_lr['precision'], eval_lr['recall'], eval_lr['f1_score'], eval_lr['roc_auc']],
            'Random Forest': [eval_rf['accuracy'], eval_rf['precision'], eval_rf['recall'], eval_rf['f1_score'], eval_rf['roc_auc']]
        }
    
    # Create comparison chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#3498db', '#e74c3c', '#f39c12']
    for i, (model_name, values) in enumerate(models_data.items()):
        offset = (i - len(models_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(images_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: docs/images/model_comparison.png")
    
except Exception as e:
    print(f"   [WARNING] Could not generate model comparison: {e}")

# 7. Feature Correlation Heatmap
print("8. Creating correlation heatmap...")
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Exited']
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, ax=ax,
            vmin=-1, vmax=1, cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(images_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: docs/images/correlation_heatmap.png")

print("\n" + "=" * 60)
print("[SUCCESS] All visualizations generated successfully!")
print(f"Images saved to: {images_dir}")
print("=" * 60)

