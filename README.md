# 🎯 AI Brand-Creator Matchmaker

> **MSc Dissertation Project - Sheffield Hallam University**
> 
> AI-driven predictive analytics for brand-creator matchmaking and ROI band prediction

## 💰 Cost: 100% FREE

This entire project uses **FREE** tools, APIs, and libraries.

---

## 📊 Dataset Overview

**1,031 YouTube Channels** across 11 niches:

| Niche | Source Files | Channels |
|-------|--------------|----------|
| **Finance** | Finance.csv, Investment.csv, Money.csv, crypto.csv | 97 |
| **Health** | health.csv, oral_health.csv, physio.csv | 587 |
| **Lifestyle** | motivation.csv, self-improvement.csv, vlogs.csv | 251 |
| **Education** | education.csv | 96 |
| **Total** | 11 CSV files | **1,031** |

---

## 🚀 Quick Start

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
5. Copy to `.env` file

**NO CREDIT CARD REQUIRED!**

### 3. Configure API Key

```bash
# Copy example and add your key
cp .env.example .env
# Edit .env and replace 'your_api_key_here' with your actual key
```

### 4. Run Data Collection

```bash
cd src/data_collection
python youtube_api.py
```

---

## 📁 Project Structure

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
│   ├── raw/                    # Collected data (channels.csv, videos.csv)
│   └── processed/              # Feature-engineered data
├── models/                     # Trained ML models
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
├── app.py                      # Streamlit prototype
├── requirements.txt
└── README.md
```

---

## 🔬 Research Questions

| RQ | Question | Method |
|----|----------|--------|
| **RQ1** | Which features influence match quality? | SHAP analysis |
| **RQ2** | Can simple models achieve acceptable performance? | AUC, Brier, ECE |
| **RQ3** | Can ROI band be predicted? | 3-class classifier |
| **RQ4** | Micro basket vs mega creators? | Portfolio simulation |

---

## 📈 7-Day Roadmap

| Day | Task | Output |
|-----|------|--------|
| 1 | Data Collection | channels.csv, videos.csv |
| 2 | Feature Engineering | creator_features.csv |
| 3 | Match Model | match_model.pkl |
| 4 | ROI Model + Calibration | roi_model.pkl |
| 5 | SHAP Analysis | Feature rankings |
| 6 | Portfolio Experiment | RQ4 results |
| 7 | Streamlit Prototype | Working demo |

---

## 👤 Author

**Md Owais Raza Abulhasan Ansari**  
MSc Student, Sheffield Hallam University  
Supervisor: Dr. Keith Harris
