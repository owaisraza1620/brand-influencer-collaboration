# 🎯 AI Brand-Creator Matchmaker

> **MSc Dissertation Project - Sheffield Hallam University**
> 
> AI-driven predictive analytics for brand-creator matchmaking and ROI band prediction

## Cost: 100% FREE

This entire project uses **FREE** tools, APIs, and libraries.

---

## Dataset Overview

**Final Dataset Statistics:**

| Dataset | Count | Description |
|---------|-------|-------------|
| **Raw Channels** | 1,006 | Collected from YouTube Data API v3 |
| **Raw Videos** | 29,627 | ~30 recent videos per channel |
| **Processed Creators** | 1,007 | Feature-engineered dataset |

**Source Channels:** 1,031 seed channels across 11 niches:

| Niche | Source Files | Seed Channels |
|-------|--------------|---------------|
| **Finance** | Finance.csv, Investment.csv, Money.csv, crypto.csv | 97 |
| **Health** | health.csv, oral_health.csv, physio.csv | 587 |
| **Lifestyle** | motivation.csv, self-improvement.csv, vlogs.csv | 251 |
| **Education** | education.csv | 96 |
| **Total** | 11 CSV files | **1,031** |

*Note: 1,006 channels successfully collected and processed (25 channels may have been unavailable or failed during collection)*

---

## Quick Start

### 1. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get FREE YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create API Key (Credentials → Create Credentials → API Key)
5. Set as environment variable or in `.env` file

**NO CREDIT CARD REQUIRED!**

### 3. Configure API Key

**Option A: Environment Variable (Recommended)**
```bash
# Windows (PowerShell):
$env:YOUTUBE_API_KEY="your_api_key_here"

# Windows (CMD):
set YOUTUBE_API_KEY=your_api_key_here

# Mac/Linux:
export YOUTUBE_API_KEY="your_api_key_here"
```

**Option B: .env File**
```bash
# Create .env file in project root
echo "YOUTUBE_API_KEY=your_api_key_here" > .env
```

### 4. Run Complete Pipeline

The project follows a 7-step pipeline:

```bash
# Step 1: Data Collection
python run_collection.py

# Step 2: Feature Engineering
python run_feature_engineering.py

# Step 3: Train Match Model
python run_match_model.py

# Step 4: Train ROI Model
python run_roi_model.py

# Step 5: SHAP Analysis
python run_shap_analysis.py

# Step 6: Run Streamlit Prototype
streamlit run streamlit_app.py
```

---

## Streamlit Web Application

### Run Locally

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Features

The Streamlit app provides 4 main tabs:

1. ** Recommendations** - Brand-side creator discovery and ranking
   - Filter by niche, size, engagement rate, subscribers
   - Ranked by ML model match probability
   - Top-K display (default: 20, adjustable 5-50)

2. ** Analysis** - Individual creator deep-dive
   - Detailed metrics and predictions
   - Feature breakdown
   - ROI band prediction

3. ** Compare** - Side-by-side creator comparison
   - Compare multiple creators
   - Visual comparison charts

4. ** Add Creator** - Creator-side registration
   - Enter YouTube Channel ID or @handle
   - Real-time feature computation
   - ML predictions
   - Save to database (CSV)

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set `YOUTUBE_API_KEY` in Streamlit Secrets
5. Deploy!

See `dissertation notes/STREAMLIT_DEPLOYMENT.md` for detailed instructions.

---

## Project Structure

```
brand-creator-matchmaker/
├── data/
│   ├── seed_channels/          # 11 CSV files with channel IDs
│   │   ├── Finance.csv
│   │   ├── Investment.csv
│   │   ├── Money.csv
│   │   ├── crypto.csv
│   │   ├── education.csv
│   │   ├── health.csv
│   │   ├── oral_health.csv
│   │   ├── physio.csv
│   │   ├── motivation.csv
│   │   ├── self-improvement.csv
│   │   └── vlogs.csv
│   ├── raw/                    # Collected data
│   │   ├── channels.csv        # 1,006 channels
│   │   └── videos.csv          # 29,627 videos
│   └── processed/              # Feature-engineered data
│       └── creator_features.csv # 1,007 creators
├── models/                     # Trained ML models
│   ├── match_model.pkl
│   ├── roi_model.pkl
│   ├── match_feature_names.pkl
│   ├── match_label_encoders.pkl
│   ├── roi_feature_names.pkl
│   ├── roi_label_encoders.pkl
│   └── *_model_results.json
├── plots/                      # Model evaluation plots
│   ├── match_*.png
│   ├── roi_*.png
│   └── shap_*.png
├── shap_results/              # SHAP explainability outputs
│   ├── shap_explainer_*.pkl
│   ├── shap_values_*.pkl
│   └── shap_insights.json
├── src/
│   ├── data_collection/
│   │   ├── youtube_api.py      # YouTube API collector
│   │   └── free_channel_sources.py  # Load seed channels
│   ├── features/               # Feature engineering
│   ├── labels/                 # Weak supervision labels
│   ├── models/                 # ML models
│   ├── evaluation/             # Calibration metrics
│   ├── explainability/         # SHAP analysis
│   └── portfolio/              # Portfolio experiment
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_collection.ipynb
├── dissertation notes/        # Dissertation documentation
├── run_collection.py          # Pipeline Step 1
├── run_feature_engineering.py # Pipeline Step 2
├── run_match_model.py         # Pipeline Step 3
├── run_roi_model.py           # Pipeline Step 4
├── run_shap_analysis.py       # Pipeline Step 5
├── streamlit_app.py           # Streamlit web application
├── youtube_api_helper.py      # YouTube API helper functions
├── requirements.txt
└── README.md
```

---

## Research Questions

| RQ | Question | Method |
|----|----------|--------|
| **RQ1** | Which features influence match quality? | SHAP analysis |
| **RQ2** | Can simple models achieve acceptable performance? | AUC, Brier, ECE |
| **RQ3** | Can ROI band be predicted? | 3-class classifier |
| **RQ4** | Micro basket vs mega creators? | Portfolio simulation |

---

## Pipeline Workflow

| Step | Script | Input | Output |
|------|--------|-------|--------|
| **1. Data Collection** | `run_collection.py` | Seed channel CSVs | `channels.csv`, `videos.csv` |
| **2. Feature Engineering** | `run_feature_engineering.py` | Raw CSVs | `creator_features.csv` |
| **3. Match Model** | `run_match_model.py` | `creator_features.csv` | `match_model.pkl` + plots |
| **4. ROI Model** | `run_roi_model.py` | `creator_features.csv` | `roi_model.pkl` + plots |
| **5. SHAP Analysis** | `run_shap_analysis.py` | Trained models | SHAP plots + insights |
| **6. Streamlit App** | `streamlit run streamlit_app.py` | Models + data | Web interface |

---

## Technology Stack

### Core Libraries
- **Data Processing:** pandas, numpy, scipy
- **Machine Learning:** scikit-learn, lightgbm
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn, plotly
- **Web App:** streamlit

### APIs & Services
- **YouTube Data API v3** (FREE tier)
- **Streamlit Cloud** (FREE hosting)

### Development Tools
- Python 3.10+
- Jupyter Notebooks
- Git & GitHub

---

## Key Features

### ML Models
- **Match Quality Model:** Binary classifier (LightGBM)
  - Predicts brand-creator match probability
  - Trained on weak supervision labels
  - Calibrated with Platt scaling

- **ROI Band Model:** 3-class classifier (LightGBM)
  - Predicts ROI potential: Low (0), Medium (1), High (2)
  - Multiclass classification
  - Per-class calibration curves

### Explainability
- **SHAP Analysis:** Feature importance rankings
- **SHAP Plots:** Bar charts, summary plots, dependence plots
- **Model Interpretability:** Understand prediction drivers

### Web Application
- **Real-time Predictions:** ML models loaded in-memory
- **Interactive Filtering:** Niche, size, engagement, subscribers
- **Top-K Ranking:** Configurable result limits
- **Creator Registration:** Add new creators via YouTube API
- **CSV Storage:** Prototype data persistence

---

## Security & Privacy

- API keys stored in environment variables (not committed)
- `.env` file excluded via `.gitignore`
- No sensitive data in repository
- YouTube API used for public data only

---

## Documentation

Additional documentation available in `dissertation notes/`:
- `STREAMLIT_DEPLOYMENT.md` - Deployment guide
- `FINAL_IMPLEMENTATION_ANSWERS.md` - Implementation details
- `IMPLEMENTATION_WEAKNESSES_LIMITATIONS.md` - Known limitations

---

## Known Limitations

1. **CSV Storage:** Prototype uses CSV files (not production database)
2. **Weak Labels:** Match quality labels are heuristic-based (not ground truth)
3. **Single API Key:** No multi-user authentication
4. **No Caching:** Predictions computed on-demand (no caching layer)

See `dissertation notes/IMPLEMENTATION_WEAKNESSES_LIMITATIONS.md` for details.

---

## License

This project is for academic research purposes (MSc Dissertation).

---

## Author

**Md Owais Raza Abulhasan Ansari**  
MSc Student, Sheffield Hallam University  
Supervisor: Dr. Keith Harris

---

## Acknowledgments

- YouTube Data API v3 (Google Cloud)
- Streamlit Community Cloud
- Open-source ML libraries (LightGBM, SHAP, scikit-learn)

---

## Contact

For questions or issues, please refer to the dissertation documentation or contact the author.

---

**Last Updated:** December 2025  
**Dataset Collection Date:** December 2025  
**Status:** Complete & Deployed
