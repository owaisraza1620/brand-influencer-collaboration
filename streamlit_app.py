# streamlit_app.py
"""
Brand-Creator Matchmaker - Professional Streamlit Application
AI-Powered Influencer Selection Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Brand-Creator Matchmaker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PROFESSIONAL CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Remove default padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Header */
    .brand-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .brand-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .brand-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    .sidebar .sidebar-content {
        background: #f8fafc;
    }
    
    /* Section Headers */
    h2 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.75rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox and Inputs */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 6px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #f1f5f9;
        padding: 0.25rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        border-radius: 6px;
        background-color: transparent;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info Boxes */
    .stSuccess {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Creator Card */
    .creator-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .creator-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA AND MODELS
# ============================================
@st.cache_data
def load_data():
    """Load creator features data with proper cleaning"""
    try:
        df = pd.read_csv('data/processed/creator_features.csv')
        
        # Clean and convert numeric columns
        numeric_cols = ['subscriber_count', 'total_views', 'video_count', 
                       'avg_views', 'median_views', 'avg_engagement_rate',
                       'median_engagement_rate', 'subscriber_view_ratio',
                       'view_consistency', 'posts_per_week']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Clean categorical columns
        categorical_cols = ['niche', 'country', 'size_band', 'er_band']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'NaN', 'None', ''], 'Unknown')
                df = df[df[col] != 'Unknown']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        match_model = joblib.load('models/match_model.pkl')
        roi_model = joblib.load('models/roi_model.pkl')
        match_encoders = joblib.load('models/match_label_encoders.pkl')
        feature_names = joblib.load('models/match_feature_names.pkl')
        return match_model, roi_model, match_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# ============================================
# HELPER FUNCTIONS
# ============================================
def format_number(num):
    """Format large numbers for display"""
    if pd.isna(num) or num == '' or num is None:
        return "0"
    
    try:
        num = float(num)
    except (ValueError, TypeError):
        return "0"
    
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))

def get_roi_label(roi_band):
    """Get ROI label with icon"""
    labels = {
        0: '🔴 Low',
        1: '🟡 Medium', 
        2: '🟢 High'
    }
    return labels.get(int(roi_band), '🟡 Medium')

def get_roi_label_text_only(roi_band):
    """Get ROI label without icon (for metric display)"""
    labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    return labels.get(int(roi_band), 'Medium')

def get_match_color(score):
    """Get color for match score"""
    if score >= 0.8:
        return "#10b981"
    elif score >= 0.5:
        return "#f59e0b"
    else:
        return "#ef4444"

def prepare_features_for_prediction(creator_row, feature_names, encoders):
    """Prepare features for ML prediction - handles missing features"""
    # Convert to Series if dict
    if isinstance(creator_row, dict):
        creator_row = pd.Series(creator_row)
    
    # Create feature vector with default values
    features = pd.Series(index=feature_names, dtype=float)
    
    # Fill in available features from creator_row
    for feat in feature_names:
        if feat in creator_row.index:
            features[feat] = creator_row[feat]
        else:
            # Set default values for missing features
            if feat == 'videos_collected':
                features[feat] = creator_row.get('video_count', 0)  # Use video_count as proxy
            elif feat in ['channel_id', 'title']:
                features[feat] = 0  # These are identifiers, not used in model
            elif feat in ['match_quality', 'roi_band']:
                features[feat] = 0  # These are targets, not features
            else:
                features[feat] = 0  # Default to 0 for numeric features
    
    # Encode categorical variables
    categorical_cols = ['country', 'niche', 'size_band', 'er_band']
    for col in categorical_cols:
        if col in features.index and col in encoders:
            try:
                val = str(features[col])
                if val in encoders[col].classes_:
                    features[col] = encoders[col].transform([val])[0]
                else:
                    features[col] = 0
            except:
                features[col] = 0
    
    # Convert to numeric, fill NaN with 0
    features = pd.to_numeric(features, errors='coerce').fillna(0)
    
    return features.values.reshape(1, -1)

def predict_for_creator(creator_row, match_model, roi_model, feature_names, encoders):
    """Get predictions for a single creator"""
    X = prepare_features_for_prediction(creator_row, feature_names, encoders)
    
    try:
        match_proba = match_model.predict_proba(X)[0]
        match_score = float(match_proba[1])
    except AttributeError:
        match_proba = match_model.predict(X)[0]
        match_score = float(match_proba)
        match_score = max(0.0, min(1.0, match_score))
    
    try:
        roi_proba = roi_model.predict_proba(X)[0]
        roi_band = int(np.argmax(roi_proba))
    except AttributeError:
        roi_proba = roi_model.predict(X)[0]
        if isinstance(roi_proba, np.ndarray):
            roi_band = int(np.argmax(roi_proba))
        else:
            roi_band = int(roi_proba)
    
    return match_score, roi_band

# ============================================
# MAIN APP
# ============================================
def main():
    # Load data and models
    df = load_data()
    if df.empty:
        st.error("❌ Could not load creator data.")
        return
    
    match_model, roi_model, match_encoders, feature_names = load_models()
    if match_model is None:
        st.error("❌ Could not load ML models.")
        return
    
    # ============================================
    # PROFESSIONAL HEADER
    # ============================================
    st.markdown("""
    <div class="brand-header">
        <h1 class="brand-title">🎯 Brand-Creator Matchmaker</h1>
        <p class="brand-subtitle">AI-Powered Influencer Selection Platform | Powered by LightGBM & SHAP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Creators", f"{len(df):,}")
    with col2:
        st.metric("Categories", len(df['niche'].unique()))
    with col3:
        st.metric("Countries", len(df['country'].unique()))
    with col4:
        avg_er = df['avg_engagement_rate'].mean() * 100 if 'avg_engagement_rate' in df.columns else 0
        st.metric("Avg Engagement", f"{avg_er:.2f}%")
    
    st.markdown("---")
    
    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.markdown("### 🎯 Campaign Setup")
        st.markdown("---")
        
        # Category Selection
        categories = sorted([x for x in df['niche'].unique() if pd.notna(x) and str(x) != 'nan'])
        if not categories:
            st.error("No categories available")
            return
        
        selected_category = st.selectbox(
            "**Brand Category**",
            options=categories,
            help="Select the category that matches your brand"
        )
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        # Size Filter
        size_bands = sorted([str(x) for x in df['size_band'].dropna().unique() if pd.notna(x) and str(x) != 'nan'])
        selected_size = st.multiselect(
            "Creator Size",
            options=size_bands,
            default=size_bands,
            help="Filter by creator size category"
        )
        
        # Engagement Filter
        min_er = st.slider(
            "Min Engagement Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5
        )
        
        # Subscriber Filter
        min_subs = st.number_input(
            "Min Subscribers",
            min_value=0,
            max_value=10_000_000,
            value=0,
            step=10000,
            format="%d"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        top_n = st.slider(
            "Top Recommendations",
            min_value=5,
            max_value=50,
            value=20
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        category_count = len(df[df['niche'] == selected_category])
        st.info(f"""
        **Category:** {selected_category}
        
        **Creators:** {category_count:,}
        
        **Total:** {len(df):,} creators
        """)
    
    # ============================================
    # MAIN TABS
    # ============================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Recommendations", 
        "🔍 Analysis",
        "⚖️ Compare",
        "➕ Add Creator"
    ])
    
    # ============================================
    # TAB 1: RECOMMENDATIONS
    # ============================================
    with tab1:
        st.markdown(f"## Top {selected_category} Creators")
        
        # Filter data
        filtered_df = df[df['niche'] == selected_category].copy()
        
        if selected_size:
            filtered_df = filtered_df[filtered_df['size_band'].isin(selected_size)]
        
        if 'avg_engagement_rate' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['avg_engagement_rate'] * 100 >= min_er]
        
        if 'subscriber_count' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['subscriber_count'] >= min_subs]
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No creators match your filters. Try adjusting the criteria.")
        else:
            # Calculate predictions
            with st.spinner("Computing ML predictions..."):
                predictions = []
                for idx, (row_idx, row) in enumerate(filtered_df.iterrows()):
                    try:
                        match_score, roi_band = predict_for_creator(
                            row, match_model, roi_model, feature_names, match_encoders
                        )
                        predictions.append({
                            'idx': row_idx,
                            'match_score': match_score,
                            'roi_band': roi_band
                        })
                    except:
                        predictions.append({
                            'idx': row_idx,
                            'match_score': 0.5,
                            'roi_band': 1
                        })
            
            pred_df = pd.DataFrame(predictions).set_index('idx')
            
            if 'roi_band' in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=['roi_band'])
            if 'match_score' in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=['match_score'])
            
            filtered_df = filtered_df.join(pred_df, how='left')
            filtered_df = filtered_df.sort_values('match_score', ascending=False)
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Creators Found", len(filtered_df))
            with col2:
                avg_er = filtered_df['avg_engagement_rate'].mean() * 100 if 'avg_engagement_rate' in filtered_df.columns else 0
                st.metric("Avg Engagement", f"{avg_er:.2f}%")
            with col3:
                high_roi = len(filtered_df[filtered_df['roi_band'] == 2])
                st.metric("High ROI", high_roi)
            with col4:
                avg_match = filtered_df['match_score'].mean()
                st.metric("Avg Match", f"{avg_match:.1%}")
            
            st.markdown("---")
            
            # Display creators
            st.markdown(f"### Top {min(top_n, len(filtered_df))} Recommendations")
            
            for rank, (idx, creator) in enumerate(filtered_df.head(top_n).iterrows(), 1):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([0.5, 2.5, 1.2, 1.2, 1])
                    
                    with col1:
                        if rank <= 3:
                            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                            st.markdown(f"### {medals[rank]}")
                        else:
                            st.markdown(f"### #{rank}")
                    
                    with col2:
                        title = str(creator.get('title', 'Unknown'))
                        st.markdown(f"**{title}**")
                        country = str(creator.get('country', 'Unknown'))
                        video_count = int(creator.get('video_count', 0))
                        subs = format_number(creator.get('subscriber_count', 0))
                        st.caption(f"{country} • {video_count} videos • {subs} subscribers")
                    
                    with col3:
                        st.markdown("**Match Score**")
                        match_score = creator.get('match_score', 0)
                        match_color = get_match_color(match_score)
                        st.markdown(f"<span style='color: {match_color}; font-size: 1.5rem; font-weight: 700;'>{match_score:.1%}</span>", unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("**ROI Potential**")
                        roi_band = int(creator.get('roi_band', 1))
                        roi_label = get_roi_label(roi_band)
                        st.markdown(f"<h3 style='margin: 0; font-size: 1.2rem;'>{roi_label}</h3>", unsafe_allow_html=True)
                    
                    with col5:
                        st.markdown("**Size**")
                        size_band = str(creator.get('size_band', 'Unknown')).upper()
                        st.markdown(f"**{size_band}**")
                    
                    with st.expander(f"📊 View Details"):
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown("**Performance Metrics**")
                            st.write(f"Engagement Rate: {creator.get('avg_engagement_rate', 0) * 100:.2f}%")
                            st.write(f"Loyalty Score: {creator.get('subscriber_view_ratio', 0):.3f}")
                            st.write(f"Posts per Week: {creator.get('posts_per_week', 0):.1f}")
                            st.write(f"View Consistency: {creator.get('view_consistency', 0):.2f}")
                            st.write(f"Total Views: {format_number(creator.get('total_views', 0))}")
                        
                        with detail_col2:
                            st.markdown("**Key Insights**")
                            er = creator.get('avg_engagement_rate', 0) * 100
                            loyalty = creator.get('subscriber_view_ratio', 0)
                            posts = creator.get('posts_per_week', 0)
                            
                            if er > 3:
                                st.write(f"✅ High engagement ({er:.1f}%)")
                            if loyalty > 0.3:
                                st.write(f"✅ Strong loyalty ({loyalty:.2f})")
                            if posts >= 2:
                                st.write(f"✅ Active posting ({posts:.1f}/week)")
                    
                    st.markdown("---")
    
    # ============================================
    # TAB 2: CREATOR ANALYSIS
    # ============================================
    with tab2:
        st.markdown("## Creator Analysis")
        
        creator_options = [str(x) for x in df['title'].unique() if pd.notna(x)]
        selected_creator = st.selectbox("Select Creator", options=creator_options)
        
        if selected_creator:
            creator_data = df[df['title'] == selected_creator].iloc[0]
            
            match_score, roi_band = predict_for_creator(
                creator_data, match_model, roi_model, feature_names, match_encoders
            )
            
            st.markdown(f"### {selected_creator}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Category", str(creator_data.get('niche', 'Unknown')))
            with col2:
                st.metric("Subscribers", format_number(creator_data.get('subscriber_count', 0)))
            with col3:
                st.metric("Match Score", f"{match_score:.1%}")
            with col4:
                st.metric("ROI Band", get_roi_label(roi_band))
            with col5:
                st.metric("Country", str(creator_data.get('country', 'Unknown')))
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Performance Metrics")
                metrics = {
                    'Total Views': format_number(creator_data.get('total_views', 0)),
                    'Video Count': int(creator_data.get('video_count', 0)),
                    'Avg Views': format_number(creator_data.get('avg_views', 0)),
                    'Engagement Rate': f"{creator_data.get('avg_engagement_rate', 0) * 100:.2f}%",
                    'Posts per Week': f"{creator_data.get('posts_per_week', 0):.1f}"
                }
                for metric, value in metrics.items():
                    st.write(f"**{metric}:** {value}")
            
            with col2:
                st.markdown("#### Quality Indicators")
                quality = {
                    'Loyalty Score': f"{creator_data.get('subscriber_view_ratio', 0):.3f}",
                    'View Consistency': f"{creator_data.get('view_consistency', 0):.2f}",
                    'Size Band': str(creator_data.get('size_band', 'Unknown')).upper()
                }
                for metric, value in quality.items():
                    st.write(f"**{metric}:** {value}")
    
    # ============================================
    # TAB 3: COMPARE CREATORS
    # ============================================
    with tab3:
        st.markdown("## Compare Creators")
        
        creator_options = [str(x) for x in df['title'].unique() if pd.notna(x)]
        selected_creators = st.multiselect(
            "Select Creators (2-5)",
            options=creator_options,
            max_selections=5
        )
        
        if len(selected_creators) >= 2:
            comparison_data = []
            
            for creator_name in selected_creators:
                creator = df[df['title'] == creator_name].iloc[0]
                match_score, roi_band = predict_for_creator(
                    creator, match_model, roi_model, feature_names, match_encoders
                )
                
                comparison_data.append({
                    'Creator': creator_name,
                    'Category': str(creator.get('niche', 'Unknown')),
                    'Subscribers': int(creator.get('subscriber_count', 0)),
                    'Engagement %': creator.get('avg_engagement_rate', 0) * 100,
                    'Match Score': match_score * 100,
                    'ROI': get_roi_label(roi_band)  # Already includes icon
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    comparison_df, 
                    x='Creator', 
                    y='Match Score',
                    title='Match Score Comparison',
                    color='Match Score',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    comparison_df, 
                    x='Creator', 
                    y='Engagement %',
                    title='Engagement Rate Comparison',
                    color='Engagement %',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            best = max(comparison_data, key=lambda x: x['Match Score'])
            st.success(f"🏆 **Best Match:** {best['Creator']} ({best['Match Score']:.1f}% match, {best['ROI']} ROI)")
        
        elif len(selected_creators) == 1:
            st.info("Please select at least 2 creators to compare")
        else:
            st.info("Select 2-5 creators from the dropdown above")
    
    # ============================================
    # TAB 4: ADD NEW CREATOR
    # ============================================
    with tab4:
        st.markdown("## Add New Creator")
        st.info("Enter YouTube Channel ID or @handle to fetch creator data")
        
        # Get API key from environment variable first, then allow user input as fallback
        api_key_from_env = os.getenv('YOUTUBE_API_KEY', '')
        
        # Only show API key input if not set in environment
        if not api_key_from_env:
            api_key = st.text_input(
                "🔑 YouTube API Key", 
                type="password",
                help="Enter your YouTube Data API v3 key. For demo purposes, this can be set as YOUTUBE_API_KEY environment variable."
            )
            if not api_key:
                st.warning("⚠️ API key required. Enter your YouTube API key above or set YOUTUBE_API_KEY environment variable.")
        else:
            api_key = api_key_from_env
            st.success("✅ YouTube API key loaded from environment variable")
            with st.expander("🔧 Change API Key"):
                api_key = st.text_input(
                    "Enter new API Key",
                    type="password",
                    value="",
                    help="Leave empty to use environment variable"
                )
                if api_key:
                    st.info("Using manually entered API key for this session")
                else:
                    api_key = api_key_from_env
        
        col1, col2 = st.columns([3, 1])
        with col1:
            channel_input = st.text_input(
                "Channel ID or @Handle", 
                placeholder="UC_x5XG1OV2P6uZZ5FSM9Ttw or @MrBeast",
                help="Enter YouTube Channel ID (starts with UC) or @handle"
            )
        with col2:
            niche_options = sorted([str(x) for x in df['niche'].unique() if pd.notna(x)])
            selected_niche = st.selectbox("Category", options=niche_options)
        
        if st.button("🚀 Fetch & Analyze", type="primary"):
            if not api_key:
                st.error("❌ Please enter YouTube API key or set YOUTUBE_API_KEY environment variable")
            elif not channel_input:
                st.error("❌ Please enter Channel ID or @handle")
            else:
                try:
                    from youtube_api_helper import YouTubeAPI, compute_features_from_api_data, save_new_creator
                    
                    with st.spinner("🔄 Fetching data from YouTube API..."):
                        api = YouTubeAPI(api_key=api_key)
                        channel_data, videos_data = api.fetch_creator_data(channel_input)
                        
                        if channel_data:
                            st.success(f"✅ Found channel: **{channel_data['title']}**")
                            
                            # Display channel info
                            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                            with info_col1:
                                st.metric("Subscribers", format_number(channel_data.get('subscriber_count', 0)))
                            with info_col2:
                                st.metric("Total Views", format_number(channel_data.get('total_views', 0)))
                            with info_col3:
                                st.metric("Videos", channel_data.get('video_count', 0))
                            with info_col4:
                                st.metric("Country", channel_data.get('country', 'Unknown'))
                            
                            st.info(f"📹 Fetched {len(videos_data)} recent videos for analysis")
                            
                            # Compute features
                            with st.spinner("⚙️ Computing features..."):
                                features = compute_features_from_api_data(
                                    channel_data, videos_data, niche=selected_niche
                                )
                            
                            # Generate predictions
                            with st.spinner("🤖 Generating ML predictions..."):
                                creator_row = pd.Series(features)
                                match_score, roi_band = predict_for_creator(
                                    creator_row, match_model, roi_model, feature_names, match_encoders
                                )
                            
                            st.markdown("---")
                            st.markdown("### 🎯 ML Predictions")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                match_color = get_match_color(match_score)
                                st.markdown(f"**Match Score**")
                                st.markdown(f"<h2 style='color: {match_color};'>{match_score:.1%}</h2>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"**ROI Band**")
                                st.markdown(f"<h2>{get_roi_label(roi_band)}</h2>", unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"**Size Band**")
                                st.markdown(f"<h2>{str(features.get('size_band', 'Unknown')).upper()}</h2>", unsafe_allow_html=True)
                            
                            # Display key features
                            st.markdown("---")
                            st.markdown("### 📊 Computed Features")
                            feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
                            with feat_col1:
                                st.metric("Engagement Rate", f"{features.get('avg_engagement_rate', 0) * 100:.2f}%")
                            with feat_col2:
                                st.metric("Loyalty Score", f"{features.get('subscriber_view_ratio', 0):.3f}")
                            with feat_col3:
                                st.metric("Posts/Week", f"{features.get('posts_per_week', 0):.1f}")
                            with feat_col4:
                                st.metric("View Consistency", f"{features.get('view_consistency', 0):.2f}")
                            
                            # Save option
                            st.markdown("---")
                            st.markdown("### 💾 Save to Database")
                            
                            if st.button("💾 Save Creator", type="secondary"):
                                features['match_quality'] = match_score
                                features['roi_band'] = roi_band
                                success = save_new_creator(channel_data, videos_data, features)
                                if success:
                                    st.success("✅ Creator saved successfully! The creator will now appear in recommendations.")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to save. Please check file permissions.")
                        else:
                            st.error("❌ Channel not found. Please check the Channel ID or @handle.")
                except ImportError:
                    st.error("❌ **youtube_api_helper.py not found!** Please ensure the file exists in the project directory.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p><strong>Brand-Creator Matchmaker</strong> | AI-Powered Influencer Selection</p>
        <p style='font-size: 0.85rem;'>Powered by LightGBM & SHAP | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
