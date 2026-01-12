# src/data_collection/free_channel_sources.py
"""
Load YouTube channel IDs from seed CSV files
Total: 1,031 channels across 11 niches
"""

import pandas as pd
import os
from typing import List, Dict, Optional
from pathlib import Path


# Niche categories mapping (CSV filename -> niche name)
NICHE_CATEGORIES = {
    'Finance': 'finance',
    'Investment': 'finance',
    'Money': 'finance',
    'crypto': 'finance',
    'education': 'education',
    'health': 'health',
    'oral_health': 'health',
    'physio': 'health',
    'motivation': 'lifestyle',
    'self-improvement': 'lifestyle',
    'vlogs': 'lifestyle',
}

# Grouped niches for analysis
NICHE_GROUPS = {
    'finance': ['Finance', 'Investment', 'Money', 'crypto'],
    'health': ['health', 'oral_health', 'physio'],
    'lifestyle': ['motivation', 'self-improvement', 'vlogs'],
    'education': ['education'],
}


def get_seed_channels_dir() -> Path:
    """Get the seed channels directory path"""
    # Try multiple possible locations
    possible_paths = [
        Path('data/seed_channels'),
        Path('../data/seed_channels'),
        Path('../../data/seed_channels'),
        Path(__file__).parent.parent.parent / 'data' / 'seed_channels',
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Default
    return Path('data/seed_channels')


def load_channels_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load channels from a single CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with channel data
    """
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename if needed
    if 'channelid' in df.columns:
        df = df.rename(columns={'channelid': 'channel_id'})
    
    return df


def load_all_seed_channels(seed_dir: str = None) -> pd.DataFrame:
    """
    Load all channel IDs from seed CSV files
    
    Args:
        seed_dir: Directory containing seed CSV files
        
    Returns:
        DataFrame with all channels and their niche labels
    """
    if seed_dir is None:
        seed_dir = get_seed_channels_dir()
    else:
        seed_dir = Path(seed_dir)
    
    all_channels = []
    
    # Load each CSV file
    for csv_file in seed_dir.glob('*.csv'):
        niche_name = csv_file.stem  # filename without extension
        
        try:
            df = load_channels_from_csv(str(csv_file))
            df['source_file'] = niche_name
            df['niche'] = NICHE_CATEGORIES.get(niche_name, niche_name.lower())
            all_channels.append(df)
            print(f"  ✓ Loaded {len(df)} channels from {niche_name}.csv")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")
    
    if not all_channels:
        print("⚠️ No channel files found!")
        return pd.DataFrame()
    
    # Combine all
    combined = pd.concat(all_channels, ignore_index=True)
    
    # Remove duplicates by channel_id
    combined = combined.drop_duplicates(subset=['channel_id'], keep='first')
    
    print(f"\n📊 Total unique channels: {len(combined)}")
    
    return combined


def get_channel_ids_by_niche(niche: str = None, seed_dir: str = None) -> List[str]:
    """
    Get channel IDs, optionally filtered by niche
    
    Args:
        niche: Filter by niche (finance, health, lifestyle, education) or None for all
        seed_dir: Directory containing seed CSV files
        
    Returns:
        List of channel IDs
    """
    df = load_all_seed_channels(seed_dir)
    
    if df.empty:
        return []
    
    if niche:
        df = df[df['niche'] == niche.lower()]
    
    return df['channel_id'].tolist()


def get_channels_summary(seed_dir: str = None) -> Dict:
    """
    Get summary statistics of seed channels
    
    Returns:
        Dictionary with channel counts by niche
    """
    df = load_all_seed_channels(seed_dir)
    
    if df.empty:
        return {}
    
    summary = {
        'total_channels': len(df),
        'by_niche': df['niche'].value_counts().to_dict(),
        'by_source': df['source_file'].value_counts().to_dict(),
    }
    
    # Add subscriber stats if available
    if 'subscribercount' in df.columns:
        df['subscribercount'] = pd.to_numeric(df['subscribercount'], errors='coerce')
        summary['subscriber_stats'] = {
            'min': int(df['subscribercount'].min()),
            'max': int(df['subscribercount'].max()),
            'mean': int(df['subscribercount'].mean()),
            'median': int(df['subscribercount'].median()),
        }
    
    return summary


def filter_by_subscriber_count(
    df: pd.DataFrame, 
    min_subs: int = 0, 
    max_subs: int = float('inf')
) -> pd.DataFrame:
    """
    Filter channels by subscriber count
    
    Args:
        df: DataFrame with channels
        min_subs: Minimum subscribers
        max_subs: Maximum subscribers
        
    Returns:
        Filtered DataFrame
    """
    if 'subscribercount' not in df.columns:
        print("⚠️ No subscriber count column found")
        return df
    
    df['subscribercount'] = pd.to_numeric(df['subscribercount'], errors='coerce')
    
    filtered = df[
        (df['subscribercount'] >= min_subs) & 
        (df['subscribercount'] <= max_subs)
    ]
    
    return filtered


def get_micro_creators(seed_dir: str = None, min_subs: int = 5000, max_subs: int = 100000) -> pd.DataFrame:
    """
    Get only micro-creators (5K-100K subscribers)
    
    Returns:
        DataFrame of micro-creators
    """
    df = load_all_seed_channels(seed_dir)
    
    if df.empty:
        return df
    
    micro = filter_by_subscriber_count(df, min_subs=min_subs, max_subs=max_subs)
    print(f"🎯 Micro-creators ({min_subs:,}-{max_subs:,} subs): {len(micro)}")
    
    return micro


# ============================================================
# Quick Access Functions
# ============================================================

def get_all_channel_ids(seed_dir: str = None) -> List[str]:
    """Get all channel IDs as a simple list"""
    df = load_all_seed_channels(seed_dir)
    return df['channel_id'].tolist() if not df.empty else []


def get_channels_with_metadata(seed_dir: str = None) -> pd.DataFrame:
    """
    Get channels with pre-existing metadata from CSVs
    (subscriber count, view count, etc.)
    """
    return load_all_seed_channels(seed_dir)


# ============================================================
# Main - Test loading
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("📂 Loading Seed Channels")
    print("="*60)
    
    # Load all channels
    channels = load_all_seed_channels()
    
    if not channels.empty:
        print("\n" + "="*60)
        print("📊 Summary")
        print("="*60)
        
        summary = get_channels_summary()
        
        print(f"\nTotal channels: {summary['total_channels']}")
        
        print("\nBy niche:")
        for niche, count in summary['by_niche'].items():
            print(f"  • {niche}: {count}")
        
        print("\nBy source file:")
        for source, count in summary['by_source'].items():
            print(f"  • {source}: {count}")
        
        if 'subscriber_stats' in summary:
            stats = summary['subscriber_stats']
            print(f"\nSubscriber stats:")
            print(f"  • Min: {stats['min']:,}")
            print(f"  • Max: {stats['max']:,}")
            print(f"  • Mean: {stats['mean']:,}")
            print(f"  • Median: {stats['median']:,}")
        
        # Test micro-creator filter
        print("\n" + "="*60)
        print("🎯 Micro-Creators (5K-100K subs)")
        print("="*60)
        micro = get_micro_creators()
        
        if not micro.empty:
            print(f"\nBy niche:")
            print(micro['niche'].value_counts())
