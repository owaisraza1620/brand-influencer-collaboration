# src/data_collection/youtube_api.py
"""
YouTube Data API v3 Collector
COMPLETELY FREE - Uses free tier quota (10,000 units/day)
No credit card required!

Updated to work with seed channel CSVs
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional
import os


class YouTubeDataCollector:
    """
    Collects YouTube channel and video data using the FREE YouTube Data API v3
    
    Daily Quota: 10,000 units (free)
    Cost per call:
        - channels.list: 1 unit
        - videos.list: 1 unit  
        - playlistItems.list: 1 unit
        - search.list: 100 units (AVOID!)
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the collector
        
        Args:
            api_key: Your FREE YouTube Data API v3 key
                     Get it from: https://console.cloud.google.com/
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.quota_used = 0
        self.daily_quota = 10000
        self.request_delay = 0.1  # Seconds between requests
        
    def get_remaining_quota(self) -> int:
        """Get remaining API quota for today"""
        return self.daily_quota - self.quota_used
    
    def _check_quota(self, cost: int = 1):
        """Check if we have enough quota"""
        if self.quota_used + cost > self.daily_quota:
            raise Exception(f"Daily quota exceeded! Used: {self.quota_used}, Limit: {self.daily_quota}")
    
    def get_channel_stats(self, channel_id: str) -> Optional[Dict]:
        """
        Get channel-level statistics and metadata
        
        Cost: 1 unit
        """
        self._check_quota(1)
        
        try:
            response = self.youtube.channels().list(
                part='statistics,snippet,contentDetails,brandingSettings',
                id=channel_id
            ).execute()
            self.quota_used += 1
            time.sleep(self.request_delay)
            
            if response.get('items'):
                channel = response['items'][0]
                stats = channel.get('statistics', {})
                snippet = channel.get('snippet', {})
                branding = channel.get('brandingSettings', {}).get('channel', {})
                
                return {
                    'channel_id': channel_id,
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', '')[:500],
                    'custom_url': snippet.get('customUrl', ''),
                    'country': snippet.get('country', 'Unknown'),
                    'created_at': snippet.get('publishedAt', ''),
                    'subscriber_count': int(stats.get('subscriberCount', 0)),
                    'total_views': int(stats.get('viewCount', 0)),
                    'video_count': int(stats.get('videoCount', 0)),
                    'hidden_subscriber_count': stats.get('hiddenSubscriberCount', False),
                    'keywords': branding.get('keywords', ''),
                    'collected_at': datetime.now().isoformat()
                }
            return None
            
        except HttpError as e:
            print(f"API Error for channel {channel_id}: {e}")
            return None
        except Exception as e:
            print(f"Error for channel {channel_id}: {e}")
            return None
    
    def get_channel_videos(self, channel_id: str, max_results: int = 30) -> List[Dict]:
        """
        Get recent videos from a channel
        
        Cost: 2-3 units per channel
        """
        self._check_quota(3)
        
        try:
            # Step 1: Get uploads playlist ID
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            self.quota_used += 1
            
            if not channel_response.get('items'):
                return []
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Step 2: Get video IDs from playlist
            video_ids = []
            next_page_token = None
            
            while len(video_ids) < max_results:
                self._check_quota(1)
                
                playlist_response = self.youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=next_page_token
                ).execute()
                self.quota_used += 1
                time.sleep(self.request_delay)
                
                for item in playlist_response.get('items', []):
                    video_ids.append(item['contentDetails']['videoId'])
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            if not video_ids:
                return []
            
            # Step 3: Get video details in batches
            videos = []
            for i in range(0, len(video_ids), 50):
                self._check_quota(1)
                
                batch_ids = video_ids[i:i+50]
                video_response = self.youtube.videos().list(
                    part='statistics,snippet,contentDetails',
                    id=','.join(batch_ids)
                ).execute()
                self.quota_used += 1
                time.sleep(self.request_delay)
                
                for video in video_response.get('items', []):
                    stats = video.get('statistics', {})
                    snippet = video.get('snippet', {})
                    content = video.get('contentDetails', {})
                    
                    videos.append({
                        'video_id': video['id'],
                        'channel_id': channel_id,
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', '')[:300],
                        'published_at': snippet.get('publishedAt', ''),
                        'category_id': snippet.get('categoryId', ''),
                        'tags': snippet.get('tags', [])[:10],
                        'duration': content.get('duration', ''),
                        'definition': content.get('definition', ''),
                        'view_count': int(stats.get('viewCount', 0)),
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0)),
                        'collected_at': datetime.now().isoformat()
                    })
            
            return videos
            
        except HttpError as e:
            print(f"API Error fetching videos for {channel_id}: {e}")
            return []
        except Exception as e:
            print(f"Error fetching videos for {channel_id}: {e}")
            return []
    
    def collect_channels_batch(
        self, 
        channel_ids: List[str],
        niche_map: Dict[str, str] = None,
        save_path: str = None,
        videos_per_channel: int = 30
    ) -> tuple:
        """
        Collect data for multiple channels
        
        Args:
            channel_ids: List of channel IDs to collect
            niche_map: Optional dict mapping channel_id -> niche
            save_path: Path to save intermediate results
            videos_per_channel: Videos to fetch per channel
            
        Returns:
            Tuple of (channels_df, videos_df)
        """
        channels_data = []
        videos_data = []
        
        total = len(channel_ids)
        
        for idx, channel_id in enumerate(channel_ids):
            print(f"\n[{idx+1}/{total}] Processing {channel_id}...")
            print(f"    Quota used: {self.quota_used}/{self.daily_quota}")
            
            # Check quota before continuing
            if self.quota_used > self.daily_quota * 0.9:
                print("⚠️ Approaching quota limit! Stopping collection.")
                break
            
            # Get channel info
            channel_info = self.get_channel_stats(channel_id)
            if channel_info:
                # Add niche if provided
                if niche_map and channel_id in niche_map:
                    channel_info['niche'] = niche_map[channel_id]
                
                channels_data.append(channel_info)
                print(f"    ✓ Channel: {channel_info['title']} ({channel_info['subscriber_count']:,} subs)")
                
                # Get videos
                videos = self.get_channel_videos(channel_id, max_results=videos_per_channel)
                videos_data.extend(videos)
                print(f"    ✓ Videos: {len(videos)} collected")
            else:
                print(f"    ✗ Failed to get channel info")
            
            # Save intermediate results every 50 channels
            if save_path and (idx + 1) % 50 == 0:
                self._save_intermediate(channels_data, videos_data, save_path)
        
        channels_df = pd.DataFrame(channels_data)
        videos_df = pd.DataFrame(videos_data)
        
        print(f"\n{'='*60}")
        print(f"✅ Collection Complete!")
        print(f"{'='*60}")
        print(f"Channels collected: {len(channels_df)}")
        print(f"Videos collected: {len(videos_df)}")
        print(f"Total API quota used: {self.quota_used}")
        
        return channels_df, videos_df
    
    def _save_intermediate(self, channels_data, videos_data, base_path):
        """Save intermediate results"""
        os.makedirs(base_path, exist_ok=True)
        pd.DataFrame(channels_data).to_csv(f"{base_path}/channels_temp.csv", index=False)
        pd.DataFrame(videos_data).to_csv(f"{base_path}/videos_temp.csv", index=False)
        print(f"    💾 Intermediate save completed")


# YouTube Category ID Mapping
YOUTUBE_CATEGORIES = {
    '1': 'Film & Animation',
    '2': 'Autos & Vehicles', 
    '10': 'Music',
    '15': 'Pets & Animals',
    '17': 'Sports',
    '19': 'Travel & Events',
    '20': 'Gaming',
    '22': 'People & Blogs',
    '23': 'Comedy',
    '24': 'Entertainment',
    '25': 'News & Politics',
    '26': 'Howto & Style',
    '27': 'Education',
    '28': 'Science & Technology',
    '29': 'Nonprofits & Activism',
}

def get_category_name(category_id: str) -> str:
    """Convert category ID to name"""
    return YOUTUBE_CATEGORIES.get(str(category_id), 'Unknown')


# ============================================================
# Main - Example Usage
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    from free_channel_sources import load_all_seed_channels
    
    load_dotenv()
    
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️ Please set YOUTUBE_API_KEY in .env file")
        print("Get your FREE key from: https://console.cloud.google.com/")
        exit(1)
    
    # Load seed channels
    print("Loading seed channels...")
    seed_channels = load_all_seed_channels()
    
    if seed_channels.empty:
        print("No seed channels found!")
        exit(1)
    
    # Create niche map
    niche_map = dict(zip(seed_channels['channel_id'], seed_channels['niche']))
    channel_ids = seed_channels['channel_id'].tolist()
    
    print(f"\nFound {len(channel_ids)} channels to collect")
    
    # Initialize collector
    collector = YouTubeDataCollector(api_key=API_KEY)
    
    # Collect data
    channels_df, videos_df = collector.collect_channels_batch(
        channel_ids=channel_ids,
        niche_map=niche_map,
        save_path='data/raw',
        videos_per_channel=30
    )
    
    # Save final results
    os.makedirs('data/raw', exist_ok=True)
    channels_df.to_csv('data/raw/channels.csv', index=False)
    videos_df.to_csv('data/raw/videos.csv', index=False)
    
    print("\n📁 Data saved to data/raw/")
