# run_feature_engineering.py
"""
Day 2: Feature Engineering + Synthetic Labels
Computes creator-level features and creates training labels

Usage:
    python run_feature_engineering.py

Input:
    - data/raw/channels.csv
    - data/raw/videos.csv

Output:
    - data/processed/creator_features.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fix Windows encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def load_raw_data():
    """Load collected channel and video data"""
    print("Loading raw data...")
    
    channels = pd.read_csv('data/raw/channels.csv')
    videos = pd.read_csv('data/raw/videos.csv')
    
    print(f"   ✓ Channels: {len(channels)} rows")
    print(f"   ✓ Videos: {len(videos)} rows")
    
    return channels, videos

    """
    for computing engagement features
    """
def compute_engagement_features(videos_df: pd.DataFrame) -> pd.DataFrame:
    
    print("\n Computing engagement features...")
    
    # Ensure numeric types
    videos_df['view_count'] = pd.to_numeric(videos_df['view_count'], errors='coerce').fillna(0)
    videos_df['like_count'] = pd.to_numeric(videos_df['like_count'], errors='coerce').fillna(0)
    videos_df['comment_count'] = pd.to_numeric(videos_df['comment_count'], errors='coerce').fillna(0)
    
    # Calculate engagement rate per video
    videos_df['engagement_rate'] = (
        (videos_df['like_count'] + videos_df['comment_count']) / 
        (videos_df['view_count'] + 1)  # +1 to avoid division by zero
    )
    
    # Aggregate by channel
    engagement_features = videos_df.groupby('channel_id').agg({
        'view_count': ['mean', 'median', 'std', 'sum'],
        'like_count': ['mean', 'sum'],
        'comment_count': ['mean', 'sum'],
        'engagement_rate': ['mean', 'median', 'std'],
        'video_id': 'count'
    }).reset_index()
    
    # Flatten column names
    engagement_features.columns = [
        'channel_id',
        'avg_views', 'median_views', 'std_views', 'total_video_views',
        'avg_likes', 'total_likes',
        'avg_comments', 'total_comments',
        'avg_engagement_rate', 'median_engagement_rate', 'er_stability',
        'videos_collected'
    ]
    
    # View stability (coefficient of variation)
    engagement_features['view_stability'] = (
        engagement_features['std_views'] / (engagement_features['avg_views'] + 1)
    )
    print("------------------------------------------------------")
    print(f" ✓ Computed for {len(engagement_features)} channels")
    print("------------------------------------------------------")
    return engagement_features


def compute_consistency_features(videos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute view consistency (% of videos above 50% of channel average)
    """
    print("\n Computing consistency features...")
    
    videos_df['view_count'] = pd.to_numeric(videos_df['view_count'], errors='coerce').fillna(0)
    
    consistency_data = []
    
    for channel_id, group in videos_df.groupby('channel_id'):
        avg_views = group['view_count'].mean()
        threshold = avg_views * 0.5
        
        # % of videos above threshold
        above_threshold = (group['view_count'] >= threshold).mean()
        
        consistency_data.append({
            'channel_id': channel_id,
            'view_consistency': above_threshold
        })
    
    consistency_df = pd.DataFrame(consistency_data)
    print(f"   ✓ Computed for {len(consistency_df)} channels")
    
    return consistency_df


def compute_posting_frequency(videos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute posting frequency (posts per week)
    """
    print("------------------------------------------------------")
    print("\n Computing posting frequency...")
    print("------------------------------------------------------")

    # Parse dates
    videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], errors='coerce')
    
    frequency_data = []
    
    for channel_id, group in videos_df.groupby('channel_id'):
        group = group.dropna(subset=['published_at'])
        
        if len(group) < 2:
            posts_per_week = 0
        else:
            date_range = (group['published_at'].max() - group['published_at'].min()).days
            weeks = max(date_range / 7, 1)
            posts_per_week = len(group) / weeks
        
        frequency_data.append({
            'channel_id': channel_id,
            'posts_per_week': posts_per_week
        })
    
    frequency_df = pd.DataFrame(frequency_data)
    print(f"   ✓ Computed for {len(frequency_df)} channels")
    
    return frequency_df


def compute_loyalty_features(channels_df: pd.DataFrame, engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LOYALTY SCORE features (key innovation!)
    
    subscriber_view_ratio = avg_views / subscriber_count(a geberal calculation since we don't have 
    access to creator's analytical features, as they allow access the calulation may change)

    Higher ratio = more loyal/engaged audience
    """
    print("------------------------------------------------------")
    print("\n Computing LOYALTY features (innovation!)...")
    print("------------------------------------------------------")

    # Merge channel subscriber data with engagement data
    loyalty_df = engagement_df[['channel_id', 'avg_views']].merge(
        channels_df[['channel_id', 'subscriber_count']], 
        on='channel_id', 
        how='left' #left join to ensure all channels are included
    )
    
    # Ensure numeric
    loyalty_df['subscriber_count'] = pd.to_numeric(loyalty_df['subscriber_count'], errors='coerce').fillna(1)
    
    # Subscriber-View Ratio (loyalty proxy)
    # Higher = views come from loyal subscribers
    # Lower = views from non-subscribers (viral but less loyal)
    loyalty_df['subscriber_view_ratio'] = (
        loyalty_df['avg_views'] / (loyalty_df['subscriber_count'] + 1)
    )
    
    # Cap extreme values
    loyalty_df['subscriber_view_ratio'] = loyalty_df['subscriber_view_ratio'].clip(0, 10)
    
    result = loyalty_df[['channel_id', 'subscriber_view_ratio']]
    print(f"   ✓ Computed for {len(result)} channels")
    
    return result


def compute_size_bands(channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize channels by subscriber count
    """
    print("------------------------------------------------------")
    print("\n Computing size bands...")
    print("------------------------------------------------------")

    channels_df['subscriber_count'] = pd.to_numeric(channels_df['subscriber_count'], errors='coerce').fillna(0)
    
    def get_size_band(subs):
        if subs < 10000:
            return 'nano'
        elif subs < 100000:
            return 'micro'
        elif subs < 1000000:
            return 'mid'
        else:
            return 'mega'
    
    size_df = channels_df[['channel_id', 'subscriber_count']].copy()
    size_df['size_band'] = size_df['subscriber_count'].apply(get_size_band)
    
    print(f"   Distribution:")
    for band, count in size_df['size_band'].value_counts().items():
        print(f"      • {band}: {count}")
    
    return size_df[['channel_id', 'size_band']]


def compute_er_bands(engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize channels by engagement rate
    """
    print("------------------------------------------------------")
    print("\n Computing engagement rate bands...")
    print("------------------------------------------------------")
    def get_er_band(er):
        if er < 0.02:
            return 'low'
        elif er < 0.05:
            return 'medium'
        else:
            return 'high'
    
    er_df = engagement_df[['channel_id', 'avg_engagement_rate']].copy()
    er_df['er_band'] = er_df['avg_engagement_rate'].apply(get_er_band)
    
    print(f"   Distribution:")
    for band, count in er_df['er_band'].value_counts().items():
        print(f"      • {band}: {count}")
    
    return er_df[['channel_id', 'er_band']]


def create_match_quality_labels(features_df: pd.DataFrame) -> pd.Series:
    """
    Create synthetic MATCH QUALITY labels using weak supervision
    
    Labeling Functions (vote on label):
    - LF1: High engagement rate → good match
    - LF2: Loyal audience → good match
    - LF3: Consistent performance → good match
    - LF4: Active posting → good match
    - LF5: Established channel → good match
    - LF6: Micro with high ER → good match
    """
    print("\n🏷️ Creating MATCH QUALITY labels (weak supervision)...")
    
    labels = []
    
    for _, row in features_df.iterrows():
        votes = []
        
        # LF1: High engagement rate
        if row['avg_engagement_rate'] > 0.04:
            votes.append(1)
        elif row['avg_engagement_rate'] < 0.01:
            votes.append(0)
        
        # LF2: Loyal audience (subscriber_view_ratio)
        if row['subscriber_view_ratio'] > 0.3:
            votes.append(1)
        elif row['subscriber_view_ratio'] < 0.05:
            votes.append(0)
        
        # LF3: Consistent performance
        if row['er_stability'] < 0.02 and row['view_consistency'] > 0.6:
            votes.append(1)
        elif row['er_stability'] > 0.05:
            votes.append(0)
        
        # LF4: Active posting
        if row['posts_per_week'] >= 1:
            votes.append(1)
        elif row['posts_per_week'] < 0.25:
            votes.append(0)
        
        # LF5: Micro with high ER (sweet spot)
        if row['size_band'] == 'micro' and row['avg_engagement_rate'] > 0.05:
            votes.append(1)
        
        # Majority vote
        if len(votes) == 0:
            # Default: based on ER
            label = 1 if row['avg_engagement_rate'] > 0.02 else 0
        else:
            label = 1 if np.mean(votes) >= 0.5 else 0
        
        labels.append(label)
    
    labels_series = pd.Series(labels, name='match_quality')
    
    print(f"   Distribution:")
    print(f"      • Good match (1): {sum(labels)}")
    print(f"      • Not match (0): {len(labels) - sum(labels)}")
    
    return labels_series


def create_roi_band_labels(features_df: pd.DataFrame) -> pd.Series:
    """
    Create synthetic ROI BAND labels (Low/Medium/High)
    
    Score-based approach:
    - High ER → +2
    - Loyal audience → +2
    - Consistent → +1
    - Micro size → +1 (often better ROI)
    - Mega size → -1 (often lower ROI per $)
    """
    print("\n🏷️ Creating ROI BAND labels (weak supervision)...")
    
    labels = []
    
    for _, row in features_df.iterrows():
        score = 0
        
        # High engagement → higher ROI
        if row['avg_engagement_rate'] > 0.05:
            score += 2
        elif row['avg_engagement_rate'] > 0.025:
            score += 1
        
        # Loyal audience → higher conversion
        if row['subscriber_view_ratio'] > 0.4:
            score += 2
        elif row['subscriber_view_ratio'] > 0.15:
            score += 1
        
        # Consistency → predictable ROI
        if row['view_consistency'] > 0.7:
            score += 1
        
        # Size factor
        if row['size_band'] == 'micro':
            score += 1  # Micro often has better ROI
        elif row['size_band'] == 'mega':
            score -= 1  # Mega often has lower ROI per $
        
        # Map score to band
        if score >= 4:
            label = 2  # High
        elif score >= 2:
            label = 1  # Medium
        else:
            label = 0  # Low
        
        labels.append(label)
    
    labels_series = pd.Series(labels, name='roi_band')
    
    print(f"   Distribution:")
    print(f"      • High (2): {labels.count(2)}")
    print(f"      • Medium (1): {labels.count(1)}")
    print(f"      • Low (0): {labels.count(0)}")
    
    return labels_series


def main():
    print("=" * 60)
    print("🔧 DAY 2: FEATURE ENGINEERING + SYNTHETIC LABELS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================
    # Step 1: Load raw data
    # ========================================
    channels_df, videos_df = load_raw_data()
    
    # ========================================
    # Step 2: Compute features
    # ========================================
    print("\n" + "=" * 60)
    print("📊 COMPUTING FEATURES")
    print("=" * 60)
    
    # Engagement features
    engagement_features = compute_engagement_features(videos_df)
    
    # Consistency features
    consistency_features = compute_consistency_features(videos_df)
    
    # Posting frequency
    frequency_features = compute_posting_frequency(videos_df)
    
    # Loyalty features (INNOVATION!)
    loyalty_features = compute_loyalty_features(channels_df, engagement_features)
    
    # Size bands
    size_bands = compute_size_bands(channels_df)
    
    # ER bands
    er_bands = compute_er_bands(engagement_features)
    
    # ========================================
    # Step 3: Merge all features
    # ========================================
    print("\n📊 Merging all features...")
    
    # Start with channels base info
    features_df = channels_df[['channel_id', 'title', 'subscriber_count', 'total_views', 
                               'video_count', 'country', 'niche']].copy()
    
    # Ensure subscriber_count is numeric
    features_df['subscriber_count'] = pd.to_numeric(features_df['subscriber_count'], errors='coerce').fillna(0)
    features_df['total_views'] = pd.to_numeric(features_df['total_views'], errors='coerce').fillna(0)
    
    # Compute channel age
    if 'created_at' in channels_df.columns:
        channels_df['created_at'] = pd.to_datetime(channels_df['created_at'], errors='coerce')
        # Make both timezone-naive for comparison
        now = pd.Timestamp.now().tz_localize(None) if pd.Timestamp.now().tz else pd.Timestamp.now()
        channels_df['created_at'] = channels_df['created_at'].dt.tz_localize(None) if channels_df['created_at'].dt.tz else channels_df['created_at']
        channels_df['channel_age_months'] = (
            (now - channels_df['created_at']).dt.days / 30
        ).fillna(0)
        features_df['channel_age_months'] = channels_df['channel_age_months']
    else:
        features_df['channel_age_months'] = 24  # Default
    
    # Merge engagement features
    features_df = features_df.merge(engagement_features, on='channel_id', how='left')
    
    # Merge consistency features
    features_df = features_df.merge(consistency_features, on='channel_id', how='left')
    
    # Merge frequency features
    features_df = features_df.merge(frequency_features, on='channel_id', how='left')
    
    # Merge loyalty features
    features_df = features_df.merge(loyalty_features, on='channel_id', how='left')
    
    # Merge size bands
    features_df = features_df.merge(size_bands, on='channel_id', how='left')
    
    # Merge ER bands
    features_df = features_df.merge(er_bands, on='channel_id', how='left')
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    print(f"   ✓ Total features: {len(features_df.columns)}")
    print(f"   ✓ Total creators: {len(features_df)}")
    
    # ========================================
    # Step 4: Create synthetic labels
    # ========================================
    print("\n" + "=" * 60)
    print("🏷️ CREATING SYNTHETIC LABELS")
    print("=" * 60)
    
    # Match quality labels
    features_df['match_quality'] = create_match_quality_labels(features_df)
    
    # ROI band labels
    features_df['roi_band'] = create_roi_band_labels(features_df)
    
    # ========================================
    # Step 5: Save processed data
    # ========================================
    print("\n📁 Saving processed data...")
    
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv('data/processed/creator_features.csv', index=False)
    
    print(f"   ✓ Saved to data/processed/creator_features.csv")
    
    # ========================================
    # Step 6: Summary
    # ========================================
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Creators: {len(features_df)}")
    print(f"   Features: {len(features_df.columns)}")
    
    print(f"\n📋 Feature Columns:")
    for col in features_df.columns:
        print(f"   • {col}")
    
    print(f"\n📈 Key Statistics:")
    print(f"   Avg Engagement Rate: {features_df['avg_engagement_rate'].mean():.4f}")
    print(f"   Avg Subscriber-View Ratio: {features_df['subscriber_view_ratio'].mean():.4f}")
    print(f"   Avg Posts/Week: {features_df['posts_per_week'].mean():.2f}")
    
    print(f"\n🏷️ Label Distribution:")
    print(f"   Match Quality: {features_df['match_quality'].value_counts().to_dict()}")
    print(f"   ROI Band: {features_df['roi_band'].value_counts().to_dict()}")
    
    print(f"\n⏱️ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n➡️ NEXT: Run Day 3 - Match Quality Model")
    print(f"   python run_match_model.py")


if __name__ == "__main__":
    main()