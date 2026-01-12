# run_collection.py
"""
Day 1: Data Collection Script
Collects YouTube channel and video data from 1,031 seed channels

Usage:
    python run_collection.py

Requirements:
    - Set YOUTUBE_API_KEY in .env file
    - Seed CSVs in data/seed_channels/
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, 'src')

from data_collection.free_channel_sources import (  # type: ignore
    load_all_seed_channels,
    get_channels_summary
)
from data_collection.youtube_api import YouTubeDataCollector  # type: ignore


def main():
    print("=" * 60)
    print("🚀 DAY 1: DATA COLLECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================
    # Step 1: Load environment
    # ========================================
    print("\n📋 Step 1: Loading configuration...")
    load_dotenv()
    
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    if not API_KEY or API_KEY == 'your_api_key_here':
        print("\n❌ ERROR: YouTube API key not found!")
        print("\nTo fix this:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create project → Enable 'YouTube Data API v3'")
        print("3. Create API Key (Credentials → Create Credentials → API Key)")
        print("4. Create .env file with: YOUTUBE_API_KEY=your_key_here")
        print("\n💡 It's FREE - no credit card required!")
        return
    
    print(f"✅ API Key loaded: {API_KEY[:10]}...")
    
    # ========================================
    # Step 2: Load seed channels
    # ========================================
    print("\n📋 Step 2: Loading seed channels...")
    
    seed_channels = load_all_seed_channels('data/seed_channels')
    
    if seed_channels.empty:
        print("❌ ERROR: No seed channels found in data/seed_channels/")
        return
    
    # Show summary
    summary = get_channels_summary('data/seed_channels')
    
    print(f"\n📊 Channels by Niche:")
    for niche, count in summary['by_niche'].items():
        print(f"   • {niche}: {count}")
    
    print(f"\n   Total: {summary['total_channels']} channels")
    
    if 'subscriber_stats' in summary:
        stats = summary['subscriber_stats']
        print(f"\n📈 Subscriber Range:")
        print(f"   • Min: {stats['min']:,}")
        print(f"   • Max: {stats['max']:,}")
        print(f"   • Median: {stats['median']:,}")
    
    # ========================================
    # Step 3: Initialize collector
    # ========================================
    print("\n📋 Step 3: Initializing YouTube API collector...")
    
    collector = YouTubeDataCollector(api_key=API_KEY)
    
    print(f"   Daily quota: {collector.daily_quota:,} units")
    print(f"   Estimated usage: ~{len(seed_channels) * 4:,} units")
    print(f"   ✅ Within free tier limit!")
    
    # ========================================
    # Step 4: Prepare collection
    # ========================================
    print("\n📋 Step 4: Preparing collection...")
    
    channel_ids = seed_channels['channel_id'].tolist()
    niche_map = dict(zip(seed_channels['channel_id'], seed_channels['niche']))
    
    print(f"   Channels to collect: {len(channel_ids)}")
    print(f"   Videos per channel: 30")
    
    # ========================================
    # Step 5: Run collection
    # ========================================
    print("\n" + "=" * 60)
    print("📥 Step 5: COLLECTING DATA (this will take 30-60 minutes)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Collect!
    channels_df, videos_df = collector.collect_channels_batch(
        channel_ids=channel_ids,
        niche_map=niche_map,
        save_path='data/raw',
        videos_per_channel=30
    )
    
    # ========================================
    # Step 6: Save results
    # ========================================
    print("\n📋 Step 6: Saving results...")
    
    channels_df.to_csv('data/raw/channels.csv', index=False)
    videos_df.to_csv('data/raw/videos.csv', index=False)
    
    print(f"   ✅ data/raw/channels.csv ({len(channels_df)} rows)")
    print(f"   ✅ data/raw/videos.csv ({len(videos_df)} rows)")
    
    # ========================================
    # Step 7: Summary
    # ========================================
    print("\n" + "=" * 60)
    print("✅ DATA COLLECTION COMPLETE!")
    print("=" * 60)
    
    print(f"\n📊 Final Statistics:")
    print(f"   Channels collected: {len(channels_df)}")
    print(f"   Videos collected: {len(videos_df)}")
    print(f"   Avg videos/channel: {len(videos_df)/max(len(channels_df),1):.1f}")
    print(f"   API quota used: {collector.quota_used:,} units")
    
    if not channels_df.empty:
        print(f"\n📂 By Niche:")
        for niche, count in channels_df['niche'].value_counts().items():
            print(f"   • {niche}: {count}")
        
        print(f"\n📈 Subscriber Stats:")
        print(f"   • Min: {channels_df['subscriber_count'].min():,}")
        print(f"   • Max: {channels_df['subscriber_count'].max():,}")
        print(f"   • Median: {channels_df['subscriber_count'].median():,.0f}")
    
    print(f"\n⏱️ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n➡️ NEXT: Run Day 2 - Feature Engineering")
    print(f"   python run_feature_engineering.py")


if __name__ == "__main__":
    main()
