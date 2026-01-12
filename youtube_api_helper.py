"""
YouTube API Helper Module
Fetches channel and video data from YouTube Data API v3
Computes features and saves new creators to database
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeAPI:
    """YouTube Data API v3 wrapper"""
    
    def __init__(self, api_key: str):
        """
        Initialize YouTube API client
        
        Args:
            api_key: YouTube Data API v3 key from Google Cloud Console
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def get_channel_id_from_handle(self, handle: str) -> str:
        """
        Convert @handle to channel ID
        
        Args:
            handle: YouTube handle (e.g., "@MrBeast")
            
        Returns:
            Channel ID (e.g., "UCX6OQ3DkcsbYNE6H8uQQuVA")
        """
        # Remove @ if present
        handle = handle.lstrip('@')
        
        try:
            request = self.youtube.search().list(
                part='snippet',
                q=handle,
                type='channel',
                maxResults=1
            )
            response = request.execute()
            
            if response['items']:
                return response['items'][0]['snippet']['channelId']
            return None
        except HttpError as e:
            print(f"Error fetching channel ID: {e}")
            return None
    
    def fetch_channel_data(self, channel_input: str) -> dict:
        """
        Fetch channel statistics and metadata
        
        Args:
            channel_input: Channel ID (UC...) or @handle
            
        Returns:
            Dictionary with channel data
        """
        # Check if it's a handle or channel ID
        if channel_input.startswith('@') or (not channel_input.startswith('UC')):
            channel_id = self.get_channel_id_from_handle(channel_input)
            if not channel_id:
                return None
        else:
            channel_id = channel_input
        
        try:
            # Get channel statistics
            request = self.youtube.channels().list(
                part='snippet,statistics',
                id=channel_id
            )
            response = request.execute()
            
            if not response['items']:
                return None
            
            channel = response['items'][0]
            snippet = channel['snippet']
            stats = channel['statistics']
            
            # Extract country from snippet (if available)
            country = snippet.get('country', 'Unknown')
            
            # Get thumbnail URL
            thumbnails = snippet.get('thumbnails', {})
            thumbnail_url = ''
            if thumbnails:
                thumbnail_url = thumbnails.get('default', {}).get('url', '') or \
                               thumbnails.get('medium', {}).get('url', '') or \
                               thumbnails.get('high', {}).get('url', '')
            
            return {
                'channel_id': channel_id,
                'title': snippet['title'],
                'subscriber_count': int(stats.get('subscriberCount', 0)),
                'total_views': int(stats.get('viewCount', 0)),
                'video_count': int(stats.get('videoCount', 0)),
                'country': country,
                'description': snippet.get('description', ''),
                'created_at': snippet.get('publishedAt', ''),
                'custom_url': snippet.get('customUrl', ''),
                'thumbnail': thumbnail_url
            }
        except HttpError as e:
            print(f"Error fetching channel data: {e}")
            return None
    
    def fetch_video_data(self, channel_id: str, max_results: int = 30) -> list:
        """
        Fetch recent videos from a channel
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch (default: 30)
            
        Returns:
            List of video dictionaries
        """
        try:
            # Get uploads playlist ID
            channel_request = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            )
            channel_response = channel_request.execute()
            
            if not channel_response['items']:
                return []
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from uploads playlist
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                playlist_request = self.youtube.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                )
                playlist_response = playlist_request.execute()
                
                video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
                
                if not video_ids:
                    break
                
                # Get video statistics
                video_request = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                )
                video_response = video_request.execute()
                
                for video in video_response['items']:
                    videos.append({
                        'video_id': video['id'],
                        'channel_id': channel_id,
                        'title': video['snippet']['title'],
                        'published_at': video['snippet']['publishedAt'],
                        'view_count': int(video['statistics'].get('viewCount', 0)),
                        'like_count': int(video['statistics'].get('likeCount', 0)),
                        'comment_count': int(video['statistics'].get('commentCount', 0))
                    })
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return videos[:max_results]
        except HttpError as e:
            print(f"Error fetching video data: {e}")
            return []
    
    def fetch_creator_data(self, channel_input: str, max_videos: int = 30):
        """
        Fetch both channel and video data
        
        Args:
            channel_input: Channel ID or @handle
            max_videos: Maximum videos to fetch
            
        Returns:
            Tuple of (channel_data dict, videos_data list)
        """
        channel_data = self.fetch_channel_data(channel_input)
        
        if not channel_data:
            return None, []
        
        videos_data = self.fetch_video_data(channel_data['channel_id'], max_videos)
        
        return channel_data, videos_data


def compute_features_from_api_data(channel_data: dict, videos_data: list, niche: str) -> dict:
    """
    Compute all features from channel and video data (same logic as Day 2)
    
    Args:
        channel_data: Dictionary with channel info
        videos_data: List of video dictionaries
        niche: Creator's niche/category
        
    Returns:
        Dictionary with all computed features
    """
    if not videos_data:
        # Return default values if no videos
        return {
            'channel_id': channel_data['channel_id'],
            'title': channel_data['title'],
            'subscriber_count': channel_data['subscriber_count'],
            'total_views': channel_data['total_views'],
            'video_count': channel_data['video_count'],
            'country': channel_data.get('country', 'Unknown'),
            'niche': niche,
            'avg_views': 0,
            'median_views': 0,
            'std_views': 0,
            'total_video_views': 0,
            'avg_likes': 0,
            'total_likes': 0,
            'avg_comments': 0,
            'total_comments': 0,
            'avg_engagement_rate': 0.01,
            'median_engagement_rate': 0.01,
            'er_stability': 0.1,
            'view_stability': 0.1,
            'view_consistency': 0.5,
            'posts_per_week': 0,
            'subscriber_view_ratio': 0.01,
            'size_band': 'nano',
            'er_band': 'low',
            'channel_age_months': 24
        }
    
    # Convert to DataFrame for easier processing
    videos_df = pd.DataFrame(videos_data)
    
    # Ensure numeric types
    videos_df['view_count'] = pd.to_numeric(videos_df['view_count'], errors='coerce').fillna(0)
    videos_df['like_count'] = pd.to_numeric(videos_df['like_count'], errors='coerce').fillna(0)
    videos_df['comment_count'] = pd.to_numeric(videos_df['comment_count'], errors='coerce').fillna(0)
    
    # Calculate engagement rate per video
    videos_df['engagement_rate'] = (
        (videos_df['like_count'] + videos_df['comment_count']) / 
        (videos_df['view_count'] + 1)
    )
    
    # Compute engagement features
    avg_views = videos_df['view_count'].mean()
    median_views = videos_df['view_count'].median()
    std_views = videos_df['view_count'].std()
    total_video_views = videos_df['view_count'].sum()
    
    avg_likes = videos_df['like_count'].mean()
    total_likes = videos_df['like_count'].sum()
    
    avg_comments = videos_df['comment_count'].mean()
    total_comments = videos_df['comment_count'].sum()
    
    avg_engagement_rate = videos_df['engagement_rate'].mean()
    median_engagement_rate = videos_df['engagement_rate'].median()
    er_stability = videos_df['engagement_rate'].std()
    
    # View stability (coefficient of variation)
    view_stability = std_views / (avg_views + 1)
    
    # View consistency (% of videos above 50% of average)
    threshold = avg_views * 0.5
    view_consistency = (videos_df['view_count'] >= threshold).mean()
    
    # Posting frequency (posts per week)
    videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], errors='coerce')
    videos_df = videos_df.dropna(subset=['published_at'])
    
    if len(videos_df) < 2:
        posts_per_week = 0
    else:
        date_range = (videos_df['published_at'].max() - videos_df['published_at'].min()).days
        weeks = max(date_range / 7, 1)
        posts_per_week = len(videos_df) / weeks
    
    # Subscriber-view ratio (loyalty score)
    subscriber_count = float(channel_data['subscriber_count'])
    subscriber_view_ratio = avg_views / (subscriber_count + 1)
    subscriber_view_ratio = min(subscriber_view_ratio, 10)  # Cap extreme values
    
    # Size band
    if subscriber_count < 10000:
        size_band = 'nano'
    elif subscriber_count < 100000:
        size_band = 'micro'
    elif subscriber_count < 1000000:
        size_band = 'mid'
    else:
        size_band = 'mega'
    
    # ER band
    if avg_engagement_rate < 0.02:
        er_band = 'low'
    elif avg_engagement_rate < 0.05:
        er_band = 'medium'
    else:
        er_band = 'high'
    
    # Channel age
    if 'created_at' in channel_data and channel_data['created_at']:
        try:
            created_at = pd.to_datetime(channel_data['created_at'])
            now_utc = datetime.now(timezone.utc)
            if created_at.tz is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            channel_age_months = (now_utc - created_at).days / 30
        except:
            channel_age_months = 24
    else:
        channel_age_months = 24
    
    # Compile all features (include videos_collected for model compatibility)
    features = {
        'channel_id': channel_data['channel_id'],
        'title': channel_data['title'],
        'subscriber_count': subscriber_count,
        'total_views': channel_data['total_views'],
        'video_count': channel_data['video_count'],
        'videos_collected': len(videos_data),  # Add this for model compatibility
        'country': channel_data.get('country', 'Unknown'),
        'niche': niche,
        'avg_views': avg_views,
        'median_views': median_views,
        'std_views': std_views,
        'total_video_views': total_video_views,
        'avg_likes': avg_likes,
        'total_likes': total_likes,
        'avg_comments': avg_comments,
        'total_comments': total_comments,
        'avg_engagement_rate': avg_engagement_rate,
        'median_engagement_rate': median_engagement_rate,
        'er_stability': er_stability,
        'view_stability': view_stability,
        'view_consistency': view_consistency,
        'posts_per_week': posts_per_week,
        'subscriber_view_ratio': subscriber_view_ratio,
        'size_band': size_band,
        'er_band': er_band,
        'channel_age_months': channel_age_months
    }
    
    return features


def save_new_creator(channel_data: dict, videos_data: list, features: dict) -> bool:
    """
    Save new creator data to CSV files (append mode)
    
    Args:
        channel_data: Channel dictionary
        videos_data: List of video dictionaries
        features: Computed features dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directories exist
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # 1. Append to channels.csv
        channels_file = 'data/raw/channels.csv'
        if os.path.exists(channels_file):
            channels_df = pd.read_csv(channels_file)
            # Check if channel already exists
            if channel_data['channel_id'] in channels_df['channel_id'].values:
                print(f"Channel {channel_data['channel_id']} already exists in database")
                return False
        else:
            channels_df = pd.DataFrame()
        
        # Create channel row
        channel_row = pd.DataFrame([{
            'channel_id': channel_data['channel_id'],
            'title': channel_data['title'],
            'subscriber_count': channel_data['subscriber_count'],
            'total_views': channel_data['total_views'],
            'video_count': channel_data['video_count'],
            'country': channel_data.get('country', 'Unknown'),
            'created_at': channel_data.get('created_at', ''),
            'niche': features.get('niche', 'Unknown')
        }])
        
        channels_df = pd.concat([channels_df, channel_row], ignore_index=True)
        channels_df.to_csv(channels_file, index=False)
        
        # 2. Append to videos.csv
        videos_file = 'data/raw/videos.csv'
        if os.path.exists(videos_file):
            videos_df = pd.read_csv(videos_file)
        else:
            videos_df = pd.DataFrame()
        
        if videos_data:
            videos_rows = pd.DataFrame(videos_data)
            videos_df = pd.concat([videos_df, videos_rows], ignore_index=True)
            videos_df.to_csv(videos_file, index=False)
        
        # 3. Append to creator_features.csv (MAIN DATABASE)
        features_file = 'data/processed/creator_features.csv'
        if os.path.exists(features_file):
            features_df = pd.read_csv(features_file)
            # Check if channel already exists
            if features['channel_id'] in features_df['channel_id'].values:
                print(f"Channel {features['channel_id']} already exists in features database")
                return False
        else:
            features_df = pd.DataFrame()
        
        # Create features row (ensure all required columns exist)
        features_row = pd.DataFrame([features])
        
        # If existing file has more columns, add missing ones
        if not features_df.empty:
            for col in features_df.columns:
                if col not in features_row.columns:
                    features_row[col] = 0
        
        features_df = pd.concat([features_df, features_row], ignore_index=True)
        features_df.to_csv(features_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error saving creator: {e}")
        return False
