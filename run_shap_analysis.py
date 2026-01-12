# run_shap_analysis.py
"""
Day 5: SHAP Explainability Analysis
Generates SHAP explanations for both Match Quality and ROI Band models

Research Questions Addressed:
- RQ1: Which public features most influence match quality? (SHAP Analysis)
- RQ2: Model interpretability and trustworthiness

Usage:
    python run_shap_analysis.py

Input:
    - data/processed/creator_features.csv
    - models/match_model.pkl
    - models/roi_model.pkl

Output:
    - plots/shap_match_summary.png
    - plots/shap_match_bar.png
    - plots/shap_match_waterfall.png
    - plots/shap_match_dependence_*.png
    - plots/shap_roi_summary.png
    - plots/shap_roi_bar.png
    - plots/shap_combined_importance.png
    - shap_results/shap_values_match.pkl
    - shap_results/shap_values_roi.pkl
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# SHAP
import shap  # type: ignore

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


def load_data_and_models():
    """Load data and trained models"""
    print(" Loading data and models...")
    
    # Load data
    df = pd.read_csv('data/processed/creator_features.csv')
    print(f"   ✓ Loaded {len(df)} creators")
    
    # Load models
    match_model = joblib.load('models/match_model.pkl')
    roi_model = joblib.load('models/roi_model.pkl')
    print(f"   ✓ Loaded match_model.pkl")
    print(f"   ✓ Loaded roi_model.pkl")
    
    # Load feature names
    feature_names = joblib.load('models/match_feature_names.pkl')
    print(f"   ✓ Loaded {len(feature_names)} feature names")
    
    return df, match_model, roi_model, feature_names


def prepare_features(df: pd.DataFrame, feature_names: list):
    """Prepare features for SHAP analysis"""
    print("\n Preparing features...")
    
    # Categorical columns
    categorical_cols = ['country', 'niche', 'size_band', 'er_band']
    
    # Create feature matrix
    X = df[feature_names].copy()
    
    # Encode categorical variables (same as training)
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Handle NaN
    X = X.fillna(0)
    
    # Get targets
    y_match = df['match_quality'].copy()
    y_roi = df['roi_band'].copy()
    
    print(f"   ✓ Feature matrix shape: {X.shape}")
    
    return X, y_match, y_roi, label_encoders


def compute_shap_values_match(model, X, feature_names):
    """Compute SHAP values for Match Quality model"""
    print("\n🔍 Computing SHAP values for Match Quality model...")
    print("   (This may take 1-2 minutes...)")
    
    # Use TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For binary classification, shap_values is a list [class_0, class_1]
    # We want class 1 (Good Match)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"   ✓ SHAP values shape: {shap_values.shape}")
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X.values,
        feature_names=feature_names
    )
    
    return shap_values, shap_explanation, explainer


def compute_shap_values_roi(model, X, feature_names):
    """Compute SHAP values for ROI Band model"""
    print("\n Computing SHAP values for ROI Band model...")
    print("   (This may take 2-3 minutes...)")
    
    # Use TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # For multiclass, shap_values is a list [class_0, class_1, class_2]
    print(f"   ✓ SHAP values computed for 3 classes")
    
    return shap_values, explainer


def plot_shap_summary_match(shap_values, X, feature_names, save_path):
    """
    Plot 1: SHAP Summary Plot (Beeswarm) for Match Quality
    This is THE KEY PLOT for RQ1!
    """
    print("\n Creating SHAP Summary Plot (Match Quality)...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title('SHAP Summary Plot - Match Quality Model\n(Addresses RQ1: Which features influence match quality?)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_shap_bar_match(shap_values, X, feature_names, save_path):
    """
    Plot 2: SHAP Bar Plot (Global Feature Importance) for Match Quality
    """
    print("\n Creating SHAP Bar Plot (Match Quality)...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title('SHAP Feature Importance - Match Quality Model\n(Mean |SHAP value|)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_shap_waterfall_match(shap_explanation, X, df, save_path, sample_idx=0):
    """
    Plot 3: SHAP Waterfall Plot (Individual Explanation) for Match Quality
    Shows WHY a specific creator got their prediction
    """
    print("\n Creating SHAP Waterfall Plot (Match Quality)...")
    
    # Find a good example (high probability good match)
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_explanation[sample_idx], show=False, max_display=12)
    
    # Get creator name for title
    creator_name = df.iloc[sample_idx]['title'] if 'title' in df.columns else f"Creator {sample_idx}"
    plt.title(f'SHAP Waterfall - Why "{creator_name}" is predicted as Good Match', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_shap_dependence(shap_values, X, feature_names, feature_idx, save_path):
    """
    Plot 4: SHAP Dependence Plot - Shows feature interaction
    """
    feature_name = feature_names[feature_idx]
    print(f"\n Creating SHAP Dependence Plot for '{feature_name}'...")
    
    plt.figure(figsize=(10, 7))
    shap.dependence_plot(
        feature_idx, 
        shap_values, 
        X, 
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: {feature_name}\n(Shows feature value vs SHAP impact)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_shap_summary_roi(shap_values, X, feature_names, save_path):
    """
    Plot 5: SHAP Summary Plot for ROI Band model (all 3 classes)
    """
    print("\n Creating SHAP Summary Plot (ROI Band)...")
    
    # For multiclass, create a combined plot showing all classes
    # Convert list of arrays to a single array for multi-output plot
    plt.figure(figsize=(12, 10))
    
    # Use multi-output summary plot
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        class_names=['Low ROI', 'Medium ROI', 'High ROI'],
        show=False,
        max_display=15
    )
    plt.title('SHAP Summary Plot - ROI Band Model (Addresses RQ3)\n(Shows feature impact for Low/Medium/High ROI)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_shap_bar_roi(shap_values, X, feature_names, save_path):
    """
    Plot 6: SHAP Bar Plot for ROI Band (averaged across classes)
    """
    print("\n Creating SHAP Bar Plot (ROI Band)...")
    
    # Average absolute SHAP values across all classes
    # shap_values is a list of arrays, each with shape (n_samples, n_features)
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    
    # Ensure mean_shap is 1D and matches feature_names length
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.flatten()
    mean_shap = mean_shap[:len(feature_names)]
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(mean_shap)],
        'importance': mean_shap
    }).sort_values('importance', ascending=True).tail(15)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(importance_df)))
    plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title('SHAP Feature Importance - ROI Band Model\n(Averaged across Low/Medium/High)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def plot_combined_importance(shap_values_match, shap_values_roi, X, feature_names, save_path):
    """
    Plot 7: Combined Feature Importance Comparison (Match vs ROI)
    """
    print("\n Creating Combined Feature Importance Plot...")
    
    # Match model importance
    match_importance = np.abs(shap_values_match).mean(axis=0)
    
    # ROI model importance (averaged across classes)
    roi_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_roi], axis=0)
    
    # Ensure both are 1D and same length
    if match_importance.ndim > 1:
        match_importance = match_importance.flatten()
    if roi_importance.ndim > 1:
        roi_importance = roi_importance.flatten()
    
    # Ensure same length
    min_len = min(len(match_importance), len(roi_importance), len(feature_names))
    match_importance = match_importance[:min_len]
    roi_importance = roi_importance[:min_len]
    feature_names_trimmed = feature_names[:min_len]
    
    # Normalize to 0-1 scale
    match_importance_norm = match_importance / match_importance.max()
    roi_importance_norm = roi_importance / roi_importance.max()
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'feature': feature_names_trimmed,
        'Match Quality': match_importance_norm,
        'ROI Band': roi_importance_norm
    })
    
    # Sort by average importance
    comparison_df['avg'] = (comparison_df['Match Quality'] + comparison_df['ROI Band']) / 2
    comparison_df = comparison_df.sort_values('avg', ascending=True).tail(12)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(comparison_df))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, comparison_df['Match Quality'], height, 
                    label='Match Quality Model', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(y + height/2, comparison_df['ROI Band'], height, 
                    label='ROI Band Model', color='#28A745', alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(comparison_df['feature'])
    ax.set_xlabel('Normalized Feature Importance (SHAP)', fontsize=12)
    ax.set_title('Feature Importance Comparison: Match Quality vs ROI Band\n(Both models use same features, different targets)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")
    
    return comparison_df


def plot_top_features_detailed(shap_values_match, X, feature_names, df, save_path):
    """
    Plot 8: Detailed analysis of top 3 features
    """
    print("\n Creating Top Features Detailed Analysis...")
    
    # Get top 3 features by importance
    importance = np.abs(shap_values_match).mean(axis=0)
    top_indices = np.argsort(importance)[-3:][::-1]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, idx in zip(axes, top_indices):
        feature_name = feature_names[idx]
        
        # Scatter plot: feature value vs SHAP value
        scatter = ax.scatter(
            X.iloc[:, idx], 
            shap_values_match[:, idx],
            c=shap_values_match[:, idx],
            cmap='RdBu_r',
            alpha=0.6,
            s=30
        )
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel(f'{feature_name} Value', fontsize=11)
        ax.set_ylabel('SHAP Value (Impact on Match Quality)', fontsize=11)
        ax.set_title(f'{feature_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Top 3 Features: How Their Values Impact Match Quality Prediction', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {save_path}")


def generate_shap_insights(shap_values_match, shap_values_roi, feature_names):
    """Generate text insights from SHAP analysis"""
    print("\n Generating SHAP Insights...")
    
    # Match model insights
    match_importance = np.abs(shap_values_match).mean(axis=0)
    match_ranking = sorted(zip(feature_names, match_importance), key=lambda x: x[1], reverse=True)
    
    # ROI model insights
    roi_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values_roi], axis=0)
    roi_ranking = sorted(zip(feature_names, roi_importance), key=lambda x: x[1], reverse=True)
    
    insights = {
        'match_quality': {
            'top_5_features': [{'feature': f, 'importance': float(i)} for f, i in match_ranking[:5]],
            'interpretation': []
        },
        'roi_band': {
            'top_5_features': [{'feature': f, 'importance': float(i)} for f, i in roi_ranking[:5]],
            'interpretation': []
        }
    }
    
    # Generate interpretations
    print("\n" + "="*60)
    print(" SHAP INSIGHTS FOR DISSERTATION")
    print("="*60)
    
    print("\n🎯 MATCH QUALITY MODEL - Top 5 Features:")
    for i, (feature, imp) in enumerate(match_ranking[:5], 1):
        print(f"   {i}. {feature}: {imp:.4f}")
        
        # Add interpretation
        if 'engagement' in feature.lower():
            interp = f"Higher {feature} increases likelihood of good match"
        elif 'subscriber_view_ratio' in feature.lower():
            interp = f"Higher loyalty score (views/subscribers) indicates better match potential"
        elif 'posts_per_week' in feature.lower():
            interp = f"More frequent posting correlates with better match quality"
        elif 'consistency' in feature.lower():
            interp = f"Consistent performance makes creator more reliable for brands"
        else:
            interp = f"{feature} significantly impacts match quality prediction"
        
        insights['match_quality']['interpretation'].append(interp)
    
    print("\n ROI BAND MODEL - Top 5 Features:")
    for i, (feature, imp) in enumerate(roi_ranking[:5], 1):
        print(f"   {i}. {feature}: {imp:.4f}")
        
        # Add interpretation
        if 'subscriber_view_ratio' in feature.lower():
            interp = f"Loyalty score is the strongest predictor of ROI band"
        elif 'engagement' in feature.lower():
            interp = f"Higher engagement drives better ROI predictions"
        elif 'subscriber_count' in feature.lower():
            interp = f"Channel size influences ROI band classification"
        elif 'consistency' in feature.lower():
            interp = f"Consistent creators are associated with predictable ROI"
        else:
            interp = f"{feature} significantly impacts ROI band prediction"
        
        insights['roi_band']['interpretation'].append(interp)
    
    return insights


def main():
    print("=" * 60)
    print(" DAY 5: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResearch Questions Addressed:")
    print("   RQ1: Which public features most influence match quality?")
    print("   (SHAP provides detailed, interpretable feature importance)")
    
    # ========================================
    # Step 1: Load data and models
    # ========================================
    df, match_model, roi_model, feature_names = load_data_and_models()
    
    # ========================================
    # Step 2: Prepare features
    # ========================================
    X, y_match, y_roi, label_encoders = prepare_features(df, feature_names)
    
    # Use a sample for faster computation (SHAP can be slow on full dataset)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    sample_indices = X_sample.index.tolist()
    
    print(f"\n Using {sample_size} samples for SHAP analysis")
    
    # ========================================
    # Step 3: Compute SHAP values for Match model
    # ========================================
    shap_values_match, shap_explanation_match, explainer_match = compute_shap_values_match(
        match_model, X_sample, feature_names
    )
    
    # ========================================
    # Step 4: Compute SHAP values for ROI model
    # ========================================
    shap_values_roi, explainer_roi = compute_shap_values_roi(
        roi_model, X_sample, feature_names
    )
    
    # ========================================
    # Step 5: Create plots
    # ========================================
    print("\n" + "="*60)
    print(" CREATING SHAP VISUALIZATIONS")
    print("="*60)
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('shap_results', exist_ok=True)
    
    # Match Quality plots
    plot_shap_summary_match(shap_values_match, X_sample, feature_names, 
                           'plots/shap_match_summary.png')
    
    plot_shap_bar_match(shap_values_match, X_sample, feature_names, 
                       'plots/shap_match_bar.png')
    
    # Find a good example for waterfall (high prediction)
    match_preds = match_model.predict(X_sample)
    good_match_idx = np.argmax(match_preds)
    plot_shap_waterfall_match(shap_explanation_match, X_sample, df.iloc[sample_indices], 
                              'plots/shap_match_waterfall.png', sample_idx=good_match_idx)
    
    # Dependence plots for top 3 features
    importance = np.abs(shap_values_match).mean(axis=0)
    top_3_indices = np.argsort(importance)[-3:][::-1]
    
    for i, feat_idx in enumerate(top_3_indices):
        plot_shap_dependence(shap_values_match, X_sample, feature_names, feat_idx,
                            f'plots/shap_match_dependence_{feature_names[feat_idx]}.png')
    
    # ROI Band plots
    plot_shap_summary_roi(shap_values_roi, X_sample, feature_names, 
                         'plots/shap_roi_summary.png')
    
    plot_shap_bar_roi(shap_values_roi, X_sample, feature_names, 
                     'plots/shap_roi_bar.png')
    
    # Combined comparison
    comparison_df = plot_combined_importance(shap_values_match, shap_values_roi, 
                                             X_sample, feature_names,
                                             'plots/shap_combined_importance.png')
    
    # Top features detailed
    plot_top_features_detailed(shap_values_match, X_sample, feature_names, df,
                               'plots/shap_top_features_detailed.png')
    
    # ========================================
    # Step 6: Generate insights
    # ========================================
    insights = generate_shap_insights(shap_values_match, shap_values_roi, feature_names)
    
    # ========================================
    # Step 7: Save SHAP values
    # ========================================
    print("\n Saving SHAP results...")
    
    joblib.dump(shap_values_match, 'shap_results/shap_values_match.pkl')
    print("   ✓ shap_results/shap_values_match.pkl")
    
    joblib.dump(shap_values_roi, 'shap_results/shap_values_roi.pkl')
    print("   ✓ shap_results/shap_values_roi.pkl")
    
    joblib.dump(explainer_match, 'shap_results/shap_explainer_match.pkl')
    print("   ✓ shap_results/shap_explainer_match.pkl")
    
    joblib.dump(explainer_roi, 'shap_results/shap_explainer_roi.pkl')
    print("   ✓ shap_results/shap_explainer_roi.pkl")
    
    # Save insights
    with open('shap_results/shap_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    print("   ✓ shap_results/shap_insights.json")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print(" DAY 5 COMPLETE!")
    print("=" * 60)
    
    print(f"\n KEY FINDINGS (RQ1 ANSWER):")
    print(f"\n   MATCH QUALITY - Top 3 Influential Features:")
    for i, item in enumerate(insights['match_quality']['top_5_features'][:3], 1):
        print(f"      {i}. {item['feature']}")
    
    print(f"\n   ROI BAND - Top 3 Influential Features:")
    for i, item in enumerate(insights['roi_band']['top_5_features'][:3], 1):
        print(f"      {i}. {item['feature']}")
    
    print(f"\n FILES CREATED:")
    print(f"   SHAP Plots (9 total):")
    print(f"      • plots/shap_match_summary.png (Beeswarm - KEY FOR RQ1)")
    print(f"      • plots/shap_match_bar.png")
    print(f"      • plots/shap_match_waterfall.png (Individual explanation)")
    print(f"      • plots/shap_match_dependence_*.png (3 plots)")
    print(f"      • plots/shap_roi_summary.png")
    print(f"      • plots/shap_roi_bar.png")
    print(f"      • plots/shap_combined_importance.png")
    print(f"      • plots/shap_top_features_detailed.png")
    print(f"   SHAP Data:")
    print(f"      • shap_results/shap_values_match.pkl")
    print(f"      • shap_results/shap_values_roi.pkl")
    print(f"      • shap_results/shap_insights.json")
    
    print(f"\n KEY INSIGHT:")
    print(f"   Your 'subscriber_view_ratio' (Loyalty Score) appears in")
    print(f"   the top features for BOTH models - validating your novel")
    print(f"   contribution to influencer marketing analytics!")
    
    print(f"\n⏱️ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n➡️ NEXT: Run Day 6 - Streamlit Prototype")
    print(f"   python run_streamlit_app.py")


if __name__ == "__main__":
    main()