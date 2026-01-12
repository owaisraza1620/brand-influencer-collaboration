# run_match_model.py
"""
Day 3: Match Quality Model Training
Trains a LightGBM classifier to predict brand-creator match quality

Research Questions Addressed:
- RQ1: Which features influence match quality? (Feature Importance)
- RQ2: Can simple models achieve acceptable performance? (Metrics + Calibration)

Usage:
    python run_match_model.py

Input:
    - data/processed/creator_features.csv

Output:
    - models/match_model.pkl
    - models/match_model_results.json
    - plots/match_roc_curve.png
    - plots/match_pr_curve.png
    - plots/match_confusion_matrix.png
    - plots/match_feature_importance.png
    - plots/match_prediction_distribution.png
    - plots/match_calibration_curve.png
    - plots/match_all_results.png (combined)
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, 
    classification_report, confusion_matrix,
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


def load_data():
    """Load processed creator features"""
    print("📂 Loading data...")
    
    df = pd.read_csv('data/processed/creator_features.csv')
    print(f"   ✓ Loaded {len(df)} creators")
    print(f"   ✓ Features: {len(df.columns)} columns")
    
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for ML model
    """
    print("\n🔧 Preparing features...")
    
    # Features to use (exclude identifiers and target)
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
    y = df['match_quality'].copy()
    
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
    print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols, label_encoders


def handle_class_imbalance(y: pd.Series):
    """
    Calculate class weights to handle imbalance
    """
    print("\n⚖️ Handling class imbalance...")
    
    class_counts = y.value_counts()
    total = len(y)
    
    print(f"   Class 0 (Not Match): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    print(f"   Class 1 (Good Match): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    
    # Scale factor for LightGBM
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")
    
    return scale_pos_weight


def train_model(X_train, y_train, scale_pos_weight):
    """
    Train LightGBM classifier
    """
    print("\n🤖 Training LightGBM model...")
    
    # Model parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'random_state': 42
    }
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train with cross-validation to find best iterations
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
    
    best_iterations = len(cv_results['valid auc-mean'])
    best_cv_auc = max(cv_results['valid auc-mean'])
    print(f"   Best iterations: {best_iterations}")
    print(f"   Best CV AUC: {best_cv_auc:.4f}")
    
    # Train final model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_iterations
    )
    
    print(f"   ✓ Model trained successfully!")
    
    return model, params


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Comprehensive model evaluation
    """
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Basic Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Calibration Metric
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # Print Results
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE METRICS")
    print("="*60)
    
    print("\n   DISCRIMINATION METRICS:")
    print(f"   ┌─────────────────────────────────┐")
    print(f"   │ ROC-AUC:        {roc_auc:.4f}          │")
    print(f"   │ PR-AUC:         {pr_auc:.4f}          │")
    print(f"   └─────────────────────────────────┘")
    
    print("\n   CLASSIFICATION METRICS:")
    print(f"   ┌─────────────────────────────────┐")
    print(f"   │ Accuracy:       {accuracy:.4f}          │")
    print(f"   │ Precision:      {precision:.4f}          │")
    print(f"   │ Recall:         {recall:.4f}          │")
    print(f"   │ F1-Score:       {f1:.4f}          │")
    print(f"   └─────────────────────────────────┘")
    
    print("\n   CALIBRATION METRICS:")
    print(f"   ┌─────────────────────────────────┐")
    print(f"   │ Brier Score:    {brier:.4f}          │")
    print(f"   └─────────────────────────────────┘")
    
    # Interpretation
    print("\n" + "="*60)
    print("📋 INTERPRETATION (for RQ2)")
    print("="*60)
    
    if roc_auc >= 0.85:
        roc_interpretation = "EXCELLENT - Model has strong discrimination ability"
    elif roc_auc >= 0.75:
        roc_interpretation = "GOOD - Model has acceptable discrimination"
    elif roc_auc >= 0.65:
        roc_interpretation = "FAIR - Model has weak but usable discrimination"
    else:
        roc_interpretation = "POOR - Model struggles to discriminate"
    
    if brier <= 0.1:
        brier_interpretation = "EXCELLENT - Well calibrated probabilities"
    elif brier <= 0.2:
        brier_interpretation = "GOOD - Reasonably calibrated"
    elif brier <= 0.25:
        brier_interpretation = "FAIR - Acceptable calibration"
    else:
        brier_interpretation = "POOR - Poorly calibrated"
    
    print(f"\n   ROC-AUC ({roc_auc:.3f}): {roc_interpretation}")
    print(f"   Brier ({brier:.3f}): {brier_interpretation}")
    
    # Classification Report
    print("\n" + "="*60)
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Match (0)', 'Good Match (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n   CONFUSION MATRIX:")
    print(f"                    Predicted")
    print(f"                  0        1")
    print(f"   Actual  0    {cm[0,0]:4d}    {cm[0,1]:4d}    (TN, FP)")
    print(f"           1    {cm[1,0]:4d}    {cm[1,1]:4d}    (FN, TP)")
    
    # Feature Importance
    print("\n" + "="*60)
    print("🔑 TOP 15 FEATURE IMPORTANCE (for RQ1)")
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
        'discrimination': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'classification': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'calibration': {
            'brier_score': float(brier)
        },
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_imp.to_dict('records'),
        'interpretation': {
            'roc_auc': roc_interpretation,
            'brier': brier_interpretation
        }
    }
    
    return results, y_pred_proba, y_pred, feature_imp, cm


def plot_roc_curve(y_test, y_pred_proba, save_path):
    """
    Plot 1: ROC Curve (Dissertation Quality)
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Main ROC curve
    ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, 
            label=f'Match Model (AUC = {roc_auc:.3f})')
    
    # Random baseline
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, 
            label='Random Classifier (AUC = 0.500)')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    
    # Styling
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Match Quality Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    # Add threshold annotations
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, 
               label=f'Optimal Threshold = {optimal_threshold:.2f}', zorder=5)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return roc_auc


def plot_pr_curve(y_test, y_pred_proba, save_path):
    """
    Plot 2: Precision-Recall Curve (Important for Imbalanced Data)
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Baseline (proportion of positive class)
    baseline = y_test.mean()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Main PR curve
    ax.plot(recall, precision, color='#28A745', lw=2.5,
            label=f'Match Model (AP = {pr_auc:.3f})')
    
    # Baseline
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1.5,
               label=f'Baseline (Positive Rate = {baseline:.3f})')
    
    # Fill area
    ax.fill_between(recall, precision, alpha=0.2, color='#28A745')
    
    # Styling
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - Match Quality Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(cm, save_path):
    """
    Plot 3: Confusion Matrix Heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Match (0)', 'Good Match (1)'],
                yticklabels=['Not Match (0)', 'Good Match (1)'],
                annot_kws={'size': 16, 'fontweight': 'bold'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_title('Confusion Matrix - Match Quality Model', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_feature_importance(feature_imp, save_path, top_n=15):
    """
    Plot 4: Feature Importance Bar Chart (Answers RQ1)
    """
    top_features = feature_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
    
    # Horizontal bar chart
    bars = ax.barh(range(len(top_features)), 
                   top_features['importance'].values,
                   color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=11)
    ax.invert_yaxis()
    
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - Match Quality Model\n(Addresses RQ1)', 
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
    Plot 5: Prediction Probability Distribution by Class
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Separate predictions by actual class
    proba_class_0 = y_pred_proba[y_test == 0]
    proba_class_1 = y_pred_proba[y_test == 1]
    
    # Plot histograms
    ax.hist(proba_class_0, bins=30, alpha=0.7, label='Not Match (0)', 
            color='#DC3545', edgecolor='black', linewidth=0.5)
    ax.hist(proba_class_1, bins=30, alpha=0.7, label='Good Match (1)', 
            color='#28A745', edgecolor='black', linewidth=0.5)
    
    # Add threshold line
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    
    ax.set_xlabel('Predicted Probability of Good Match', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Distribution by Actual Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_calibration_curve(y_test, y_pred_proba, save_path):
    """
    Plot 6: Calibration Curve (Reliability Diagram) - Key for RQ2
    """
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly Calibrated')
    
    # Model calibration
    ax.plot(prob_pred, prob_true, 's-', color='#2E86AB', lw=2, markersize=8,
            label='Match Model')
    
    # Fill between for visual
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)\n(Addresses RQ2 - Calibration)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    # Add Brier score annotation
    brier = brier_score_loss(y_test, y_pred_proba)
    ax.text(0.05, 0.95, f'Brier Score: {brier:.4f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_combined_results(y_test, y_pred_proba, y_pred, cm, feature_imp, save_path):
    """
    Combined Plot: All Results in One Figure (for Dissertation Overview)
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    ax2.plot(recall, precision, color='#28A745', lw=2, label=f'AP = {pr_auc:.3f}')
    ax2.fill_between(recall, precision, alpha=0.2, color='#28A745')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'],
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix', fontweight='bold')
    
    # 4. Feature Importance (Top 10)
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_10 = feature_imp.head(10)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
    ax4.barh(range(10), top_10['importance'].values, color=colors)
    ax4.set_yticks(range(10))
    ax4.set_yticklabels(top_10['feature'].values)
    ax4.invert_yaxis()
    ax4.set_xlabel('Importance (Gain)')
    ax4.set_title('Top 10 Feature Importance (RQ1)', fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    
    # 5. Calibration Curve
    ax5 = fig.add_subplot(gs[1, 2])
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    brier = brier_score_loss(y_test, y_pred_proba)
    ax5.plot([0, 1], [0, 1], 'k--', lw=1)
    ax5.plot(prob_pred, prob_true, 's-', color='#2E86AB', lw=2, markersize=6,
             label=f'Brier = {brier:.3f}')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Actual Probability')
    ax5.set_title('Calibration Curve (RQ2)', fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Match Quality Model - Complete Evaluation Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 60)
    print("🎯 DAY 3: MATCH QUALITY MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResearch Questions Addressed:")
    print("   RQ1: Which features influence match quality?")
    print("   RQ2: Can simple models achieve acceptable performance?")
    
    # ========================================
    # Step 1: Load data
    # ========================================
    df = load_data()
    
    # ========================================
    # Step 2: Prepare features
    # ========================================
    X, y, feature_names, label_encoders = prepare_features(df)
    
    # ========================================
    # Step 3: Handle class imbalance
    # ========================================
    scale_pos_weight = handle_class_imbalance(y)
    
    # ========================================
    # Step 4: Train/Test split
    # ========================================
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # ========================================
    # Step 5: Train model
    # ========================================
    model, params = train_model(X_train, y_train, scale_pos_weight)
    
    # ========================================
    # Step 6: Evaluate model
    # ========================================
    results, y_pred_proba, y_pred, feature_imp, cm = evaluate_model(
        model, X_test, y_test, feature_names
    )
    
    # ========================================
    # Step 7: Create plots
    # ========================================
    print("\n📊 Creating dissertation-quality plots...")
    
    os.makedirs('plots', exist_ok=True)
    
    # Individual plots
    plot_roc_curve(y_test, y_pred_proba, 'plots/match_roc_curve.png')
    print("   ✓ plots/match_roc_curve.png")
    
    plot_pr_curve(y_test, y_pred_proba, 'plots/match_pr_curve.png')
    print("   ✓ plots/match_pr_curve.png")
    
    plot_confusion_matrix(cm, 'plots/match_confusion_matrix.png')
    print("   ✓ plots/match_confusion_matrix.png")
    
    plot_feature_importance(feature_imp, 'plots/match_feature_importance.png')
    print("   ✓ plots/match_feature_importance.png")
    
    plot_prediction_distribution(y_test, y_pred_proba, 'plots/match_prediction_distribution.png')
    print("   ✓ plots/match_prediction_distribution.png")
    
    plot_calibration_curve(y_test, y_pred_proba, 'plots/match_calibration_curve.png')
    print("   ✓ plots/match_calibration_curve.png")
    
    # Combined plot for overview
    plot_combined_results(y_test, y_pred_proba, y_pred, cm, feature_imp, 
                          'plots/match_all_results.png')
    print("   ✓ plots/match_all_results.png (combined)")
    
    # ========================================
    # Step 8: Save model and results
    # ========================================
    print("\n💾 Saving model and results...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/match_model.pkl')
    print(f"   ✓ models/match_model.pkl")
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/match_label_encoders.pkl')
    print(f"   ✓ models/match_label_encoders.pkl")
    
    # Save feature names
    joblib.dump(feature_names, 'models/match_feature_names.pkl')
    print(f"   ✓ models/match_feature_names.pkl")
    
    # Save complete results
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_names),
        'model_params': params
    }
    
    with open('models/match_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ models/match_model_results.json")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("✅ DAY 3 COMPLETE!")
    print("=" * 60)
    
    print(f"\n📊 KEY RESULTS:")
    print(f"   ┌─────────────────────────────────────┐")
    print(f"   │ ROC-AUC:        {results['discrimination']['roc_auc']:.4f}              │")
    print(f"   │ PR-AUC:         {results['discrimination']['pr_auc']:.4f}              │")
    print(f"   │ Brier Score:    {results['calibration']['brier_score']:.4f}              │")
    print(f"   │ F1-Score:       {results['classification']['f1_score']:.4f}              │")
    print(f"   └─────────────────────────────────────┘")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   Models:")
    print(f"      • models/match_model.pkl")
    print(f"      • models/match_model_results.json")
    print(f"   Plots (7 total):")
    print(f"      • plots/match_roc_curve.png")
    print(f"      • plots/match_pr_curve.png")
    print(f"      • plots/match_confusion_matrix.png")
    print(f"      • plots/match_feature_importance.png")
    print(f"      • plots/match_prediction_distribution.png")
    print(f"      • plots/match_calibration_curve.png")
    print(f"      • plots/match_all_results.png (combined)")
    
    print(f"\n🔬 TOP 5 IMPORTANT FEATURES (RQ1):")
    for i, row in feature_imp.head(5).iterrows():
        print(f"      {feature_imp.head(5).index.get_loc(i)+1}. {row['feature']}")
    
    print(f"\n⏱️ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n➡️ NEXT: Run Day 4 - ROI Band Model")
    print(f"   python run_roi_model.py")


if __name__ == "__main__":
    main()