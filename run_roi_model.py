# run_roi_model.py
"""
Day 4: ROI Band Model Training
Trains a LightGBM classifier to predict ROI band (Low/Medium/High)

Research Questions Addressed:
- RQ3: Can ROI band be predicted from public features?
- RQ2: Can simple models achieve acceptable discrimination & calibration?

Usage:
    python run_roi_model.py

Input:
    - data/processed/creator_features.csv

Output:
    - models/roi_model.pkl
    - models/roi_model_results.json
    - plots/roi_*.png (7 plots)
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Fix Windows encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb  # type: ignore

# Visualization
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

warnings.filterwarnings('ignore')

# Set style for dissertation-quality plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ROI Band labels
ROI_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
ROI_COLORS = {'Low': '#DC3545', 'Medium': '#FFC107', 'High': '#28A745'}


def load_data():
    """Load processed creator features"""
    print(" Loading data...")
    
    df = pd.read_csv('data/processed/creator_features.csv')
    print(f"   ✓ Loaded {len(df)} creators")
    print(f"   ✓ Features: {len(df.columns)} columns")
    
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for ML model
    """
    print("\n Preparing features...")
    
    # Features to use (exclude identifiers and targets)
    exclude_cols = ['channel_id', 'title', 'match_quality', 'roi_band']
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate numeric and categorical
    categorical_cols = ['country', 'niche', 'size_band', 'er_band']
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    # Create feature matrix
    X = df[feature_cols].copy()
    y = df['roi_band'].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    print(f"   ✓ Final feature matrix: {X.shape}")
    print(f"   ✓ Target distribution:")
    for label, count in y.value_counts().sort_index().items():
        print(f"      Class {label} ({ROI_LABELS[label]}): {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols, label_encoders


def train_model(X_train, y_train):
    """
    Train LightGBM multiclass classifier
    """
    print("\n Training LightGBM model (3-class)...")
    
    # Model parameters for multiclass
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'class_weight': 'balanced',  # Handle imbalance
        'verbose': -1,
        'random_state': 42
    }
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train with cross-validation
    print("   Running 5-fold cross-validation...")
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=500,
        nfold=5,
        stratified=True,
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
        seed=42
    )
    
    best_iterations = len(cv_results['valid multi_logloss-mean'])
    best_cv_loss = min(cv_results['valid multi_logloss-mean'])
    print(f"   Best iterations: {best_iterations}")
    print(f"   Best CV Log Loss: {best_cv_loss:.4f}")
    
    # Train final model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_iterations
    )
    
    print(f" * Model trained successfully! *")
    
    return model, params


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Comprehensive multiclass model evaluation
    """
    print("\n Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)  # Shape: (n_samples, 3)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Basic Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Multiclass ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc_ovr = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')
    roc_auc_per_class = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average=None)
    
    # Brier Score (multiclass)
    brier_scores = []
    for i in range(3):
        brier = brier_score_loss(y_test_bin[:, i], y_pred_proba[:, i])
        brier_scores.append(brier)
    avg_brier = np.mean(brier_scores)
    
    # Print Results
    print("\n" + "="*60)
    print(" MODEL PERFORMANCE METRICS (3-CLASS)")
    print("="*60)
    
    print("\n   OVERALL METRICS:")
    print(f"   ┌─────────────────────────────────┐")
    print(f"   │ Accuracy:         {accuracy:.4f}          │")
    print(f"   │ Macro F1-Score:   {f1_macro:.4f}          │")
    print(f"   │ Weighted F1:      {f1_weighted:.4f}          │")
    print(f"   │ Macro ROC-AUC:    {roc_auc_ovr:.4f}          │")
    print(f"   │ Avg Brier Score:  {avg_brier:.4f}          │")
    print(f"   └─────────────────────────────────┘")
    
    print("\n   PER-CLASS METRICS:")
    print(f"   ┌────────────┬───────────┬──────────┬──────────┬──────────┐")
    print(f"   │ Class      │ Precision │ Recall   │ F1-Score │ ROC-AUC  │")
    print(f"   ├────────────┼───────────┼──────────┼──────────┼──────────┤")
    for i, label in ROI_LABELS.items():
        print(f"   │ {label:<10} │ {precision_per_class[i]:.4f}    │ {recall_per_class[i]:.4f}   │ {f1_per_class[i]:.4f}   │ {roc_auc_per_class[i]:.4f}   │")
    print(f"   └────────────┴───────────┴──────────┴──────────┴──────────┘")
    
    # Interpretation
    print("\n" + "="*60)
    print(" INTERPRETATION (for RQ3)")
    print("="*60)
    
    if roc_auc_ovr >= 0.85:
        roc_interpretation = "EXCELLENT - Model strongly discriminates ROI bands"
    elif roc_auc_ovr >= 0.75:
        roc_interpretation = "GOOD - Model adequately discriminates ROI bands"
    elif roc_auc_ovr >= 0.65:
        roc_interpretation = "FAIR - Model has moderate discrimination"
    else:
        roc_interpretation = "POOR - Model struggles to discriminate"
    
    if avg_brier <= 0.15:
        brier_interpretation = "EXCELLENT - Well calibrated probabilities"
    elif avg_brier <= 0.25:
        brier_interpretation = "GOOD - Reasonably calibrated"
    else:
        brier_interpretation = "FAIR - Calibration could improve"
    
    print(f"\n   ROC-AUC ({roc_auc_ovr:.3f}): {roc_interpretation}")
    print(f"   Brier ({avg_brier:.3f}): {brier_interpretation}")
    
    # Classification Report
    print("\n" + "="*60)
    print(" DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, 
                                target_names=['Low (0)', 'Medium (1)', 'High (2)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n   CONFUSION MATRIX:")
    print(f"                      Predicted")
    print(f"                  Low    Med    High")
    print(f"   Actual Low    {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
    print(f"         Med    {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
    print(f"         High   {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")
    
    # Feature Importance
    print("\n" + "="*60)
    print(" TOP 15 FEATURE IMPORTANCE (for RQ3)")
    print("="*60)
    
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n   Rank  Feature                      Importance")
    print("   " + "-"*50)
    for rank, (_, row) in enumerate(feature_imp.head(15).iterrows(), 1):
        print(f"   {rank:2d}.   {row['feature']:<25} {row['importance']:>10.2f}")
    
    # Store results
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'macro_precision': float(precision_macro),
            'macro_recall': float(recall_macro),
            'macro_f1': float(f1_macro),
            'weighted_f1': float(f1_weighted),
            'macro_roc_auc': float(roc_auc_ovr),
            'avg_brier_score': float(avg_brier)
        },
        'per_class': {
            'precision': {ROI_LABELS[i]: float(v) for i, v in enumerate(precision_per_class)},
            'recall': {ROI_LABELS[i]: float(v) for i, v in enumerate(recall_per_class)},
            'f1_score': {ROI_LABELS[i]: float(v) for i, v in enumerate(f1_per_class)},
            'roc_auc': {ROI_LABELS[i]: float(v) for i, v in enumerate(roc_auc_per_class)},
            'brier_score': {ROI_LABELS[i]: float(v) for i, v in enumerate(brier_scores)}
        },
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_imp.to_dict('records'),
        'interpretation': {
            'roc_auc': roc_interpretation,
            'brier': brier_interpretation
        }
    }
    
    return results, y_pred_proba, y_pred, feature_imp, cm


def plot_confusion_matrix(cm, save_path):
    """
    Plot 1: Confusion Matrix Heatmap (3-class)
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low (0)', 'Medium (1)', 'High (2)'],
                yticklabels=['Low (0)', 'Medium (1)', 'High (2)'],
                annot_kws={'size': 16, 'fontweight': 'bold'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_title('Confusion Matrix - ROI Band Model (3-Class)', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(3):
        for j in range(3):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.75, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_roc_curves_multiclass(y_test, y_pred_proba, save_path):
    """
    Plot 2: ROC Curves for each class (One-vs-Rest)
    """
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    colors = ['#DC3545', '#FFC107', '#28A745']
    
    for i, (label, color) in enumerate(zip(['Low', 'Medium', 'High'], colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{label} (AUC = {auc:.3f})')
    
    # Random baseline
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - ROI Band Model (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_feature_importance(feature_imp, save_path, top_n=15):
    """
    Plot 3: Feature Importance Bar Chart
    """
    top_features = feature_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color gradient (green for ROI)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_features)))[::-1]
    
    bars = ax.barh(range(len(top_features)), 
                   top_features['importance'].values,
                   color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=11)
    ax.invert_yaxis()
    
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - ROI Band Model\n(Addresses RQ3)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
        ax.text(val + max(top_features['importance']) * 0.01, i, 
                f'{val:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_prediction_distribution(y_test, y_pred_proba, save_path):
    """
    Plot 4: Prediction Probability Distribution by Class
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = ['Low', 'Medium', 'High']
    colors = ['#DC3545', '#FFC107', '#28A745']
    
    for i, (ax, name, color) in enumerate(zip(axes, class_names, colors)):
        # Get predicted probability for this class
        proba = y_pred_proba[:, i]
        
        # Separate by actual class
        for j, (actual_name, actual_color) in enumerate(zip(class_names, colors)):
            mask = y_test == j
            ax.hist(proba[mask], bins=20, alpha=0.5, label=f'Actual: {actual_name}',
                   color=actual_color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(f'P({name})', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Predicted Probability\nfor {name} ROI', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Distribution by Actual Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_calibration_curves(y_test, y_pred_proba, save_path):
    """
    Plot 5: Calibration Curves for each class
    """
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = ['Low', 'Medium', 'High']
    colors = ['#DC3545', '#FFC107', '#28A745']
    
    for i, (ax, name, color) in enumerate(zip(axes, class_names, colors)):
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_pred_proba[:, i], n_bins=8)
        brier = brier_score_loss(y_test_bin[:, i], y_pred_proba[:, i])
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect')
        
        # Model calibration
        ax.plot(prob_pred, prob_true, 's-', color=color, lw=2, markersize=8,
                label=f'{name} (Brier={brier:.3f})')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f'Calibration: {name} ROI', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Calibration Curves by Class (Addresses RQ2)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_class_performance_comparison(results, save_path):
    """
    Plot 6: Per-class performance comparison
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['Low', 'Medium', 'High']
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    x = np.arange(len(classes))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric == 'Precision':
            values = [results['per_class']['precision'][c] for c in classes]
        elif metric == 'Recall':
            values = [results['per_class']['recall'][c] for c in classes]
        elif metric == 'F1-Score':
            values = [results['per_class']['f1_score'][c] for c in classes]
        else:  # ROC-AUC
            values = [results['per_class']['roc_auc'][c] for c in classes]
        
        bars = ax.bar(x + i*width, values, width, label=metric, color=color, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('ROI Band', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Comparison - ROI Band Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_combined_results(y_test, y_pred_proba, y_pred, cm, feature_imp, results, save_path):
    """
    Combined Plot: All Results in One Figure
    """
    fig = plt.figure(figsize=(18, 12))
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Low', 'Med', 'High'],
                yticklabels=['Low', 'Med', 'High'],
                annot_kws={'size': 12, 'fontweight': 'bold'}, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix', fontweight='bold')
    
    # 2. ROC Curves
    ax2 = fig.add_subplot(gs[0, 1])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    colors = ['#DC3545', '#FFC107', '#28A745']
    for i, (label, color) in enumerate(zip(['Low', 'Med', 'High'], colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{label} ({auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curves (OvR)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-class Performance
    ax3 = fig.add_subplot(gs[0, 2])
    classes = ['Low', 'Medium', 'High']
    f1_scores = [results['per_class']['f1_score'][c] for c in classes]
    bar_colors = ['#DC3545', '#FFC107', '#28A745']
    bars = ax3.bar(classes, f1_scores, color=bar_colors, edgecolor='black')
    for bar, val in zip(bars, f1_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score by Class', fontweight='bold')
    ax3.set_ylim([0, 1.1])
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Feature Importance (Top 10)
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_10 = feature_imp.head(10)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, 10))[::-1]
    ax4.barh(range(10), top_10['importance'].values, color=colors)
    ax4.set_yticks(range(10))
    ax4.set_yticklabels(top_10['feature'].values)
    ax4.invert_yaxis()
    ax4.set_xlabel('Importance (Gain)')
    ax4.set_title('Top 10 Feature Importance (RQ3)', fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    
    # 5. Metrics Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    metrics_text = f"""
    ┌─────────────────────────────────┐
    │     ROI BAND MODEL RESULTS      │
    ├─────────────────────────────────┤
    │ Accuracy:       {results['overall']['accuracy']:.4f}          │
    │ Macro F1:       {results['overall']['macro_f1']:.4f}          │
    │ Macro ROC-AUC:  {results['overall']['macro_roc_auc']:.4f}          │
    │ Avg Brier:      {results['overall']['avg_brier_score']:.4f}          │
    └─────────────────────────────────┘
    
    Answers RQ3: Can ROI band be
    predicted from public features?
    
    ✓ Model achieves {results['overall']['macro_roc_auc']:.1%} ROC-AUC
    ✓ Calibration: Brier = {results['overall']['avg_brier_score']:.3f}
    """
    ax5.text(0.1, 0.5, metrics_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('ROI Band Model - Complete Evaluation Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 60)
    print(" DAY 4: ROI BAND MODEL TRAINING (3-CLASS)")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResearch Questions Addressed:")
    print("   RQ2: Can simple models achieve acceptable calibration?")
    print("   RQ3: Can ROI band be predicted from public features?")
    
    # ========================================
    # Step 1: Load data
    # ========================================
    df = load_data()
    
    # ========================================
    # Step 2: Prepare features
    # ========================================
    X, y, feature_names, label_encoders = prepare_features(df)
    
    # ========================================
    # Step 3: Train/Test split
    # ========================================
    print("\n Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # ========================================
    # Step 4: Train model
    # ========================================
    model, params = train_model(X_train, y_train)
    
    # ========================================
    # Step 5: Evaluate model
    # ========================================
    results, y_pred_proba, y_pred, feature_imp, cm = evaluate_model(
        model, X_test, y_test, feature_names
    )
    
    # ========================================
    # Step 6: Create plots
    # ========================================
    print("\n Creating dissertation-quality plots...")
    
    os.makedirs('plots', exist_ok=True)
    
    # Individual plots
    plot_confusion_matrix(cm, 'plots/roi_confusion_matrix.png')
    print("   ✓ plots/roi_confusion_matrix.png")
    
    plot_roc_curves_multiclass(y_test, y_pred_proba, 'plots/roi_roc_curves.png')
    print("   ✓ plots/roi_roc_curves.png")
    
    plot_feature_importance(feature_imp, 'plots/roi_feature_importance.png')
    print("   ✓ plots/roi_feature_importance.png")
    
    plot_prediction_distribution(y_test, y_pred_proba, 'plots/roi_prediction_distribution.png')
    print("   ✓ plots/roi_prediction_distribution.png")
    
    plot_calibration_curves(y_test, y_pred_proba, 'plots/roi_calibration_curves.png')
    print("   ✓ plots/roi_calibration_curves.png")
    
    plot_class_performance_comparison(results, 'plots/roi_class_comparison.png')
    print("   ✓ plots/roi_class_comparison.png")
    
    # Combined plot
    plot_combined_results(y_test, y_pred_proba, y_pred, cm, feature_imp, results,
                          'plots/roi_all_results.png')
    print("   ✓ plots/roi_all_results.png (combined)")
    
    # ========================================
    # Step 7: Save model and results
    # ========================================
    print("\n Saving model and results...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/roi_model.pkl')
    print(f"   ✓ models/roi_model.pkl")
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/roi_label_encoders.pkl')
    print(f"   ✓ models/roi_label_encoders.pkl")
    
    # Save feature names
    joblib.dump(feature_names, 'models/roi_feature_names.pkl')
    print(f"   ✓ models/roi_feature_names.pkl")
    
    # Save complete results
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_names),
        'n_classes': 3,
        'class_labels': ROI_LABELS,
        'model_params': params
    }
    
    with open('models/roi_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ models/roi_model_results.json")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print(" DAY 4 COMPLETE!")
    print("=" * 60)
    
    print(f"\n KEY RESULTS:")
    print(f"   ┌─────────────────────────────────────┐")
    print(f"   │ Accuracy:         {results['overall']['accuracy']:.4f}              │")
    print(f"   │ Macro F1-Score:   {results['overall']['macro_f1']:.4f}              │")
    print(f"   │ Macro ROC-AUC:    {results['overall']['macro_roc_auc']:.4f}              │")
    print(f"   │ Avg Brier Score:  {results['overall']['avg_brier_score']:.4f}              │")
    print(f"   └─────────────────────────────────────┘")
    
    print(f"\n PER-CLASS F1-SCORES:")
    for cls, score in results['per_class']['f1_score'].items():
        print(f"      {cls}: {score:.4f}")
    
    print(f"\n FILES CREATED:")
    print(f"   Models:")
    print(f"      • models/roi_model.pkl")
    print(f"      • models/roi_model_results.json")
    print(f"   Plots (7 total):")
    print(f"      • plots/roi_confusion_matrix.png")
    print(f"      • plots/roi_roc_curves.png")
    print(f"      • plots/roi_feature_importance.png")
    print(f"      • plots/roi_prediction_distribution.png")
    print(f"      • plots/roi_calibration_curves.png")
    print(f"      • plots/roi_class_comparison.png")
    print(f"      • plots/roi_all_results.png (combined)")
    
    print(f"\ TOP 5 IMPORTANT FEATURES (RQ3):")
    for i, (_, row) in enumerate(feature_imp.head(5).iterrows(), 1):
        print(f"      {i}. {row['feature']}")
    
    print(f"\n Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n NEXT: Run Day 5 - SHAP Explainability")
    print(f"   python run_shap_analysis.py")


if __name__ == "__main__":
    main()