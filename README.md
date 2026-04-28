# Election Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql&logoColor=white)
![FRED API](https://img.shields.io/badge/Data-FRED%20API-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A non-partisan, data-driven US presidential election prediction dashboard built with Python and Streamlit. Uses XGBoost trained on county-level historical results (2000–2024) combined with live economic and approval data to forecast state-by-state outcomes and Electoral College totals.

## Features

- **Interactive Dashboard** — 4-tab Streamlit app with choropleth maps, EC breakdowns, and confidence intervals
- **Live Data** — Pulls GDP, unemployment, CPI, and consumer confidence from the FRED API in real time
- **XGBoost Model** — Trained on 71,840 county-level rows with Leave-One-Election-Out (LOEO) cross-validation and recency weighting
- **Confidence Intervals** — Three models (median, 2.5th, 97.5th percentile) for 95% CI on every state prediction
- **Approval Adjustment** — Incumbent party vote share adjusted live based on current approval polling
- **PostgreSQL Backend** — Cleaned historical data stored and queried from a local PostgreSQL database
- **Prediction History** — Save predictions and track EC vote trends over time

## Project Structure

```
Election-Prediction/
│
├── src/
│   ├── app.py                  # Streamlit dashboard (4 tabs)
│   ├── config.py               # DB connection string + FRED API key
│   ├── data_cleaning.py        # Merges raw CSVs, outputs cleaned_polls.csv
│   ├── feature_engineering.py  # Builds ML features, outputs featured_data.csv
│   ├── modeling.py             # Trains XGBoost models, saves to src/models/
│   ├── database.py             # Bulk-inserts cleaned data into PostgreSQL
│   ├── data_sources.py         # Live FRED + approval data fetching
│   ├── electoral_votes.py      # EV allocations and EC calculation
│   └── data/
│       ├── polling_data/
│       │   ├── complete_data.csv           # MIT Election Lab state-level 1976–2020
│       │   ├── 2024president.csv           # 2024 state-level results
│       │   ├── countypres_2000-2024.csv    # County-level results 2000–2024
│       │   ├── GDP.csv / unemployment.csv  # Local FRED economic data
│       │   └── Silver Bulletin Trump approval polls - Sheet1.csv
│       ├── cleaned_polls.csv   # Output of data_cleaning.py
│       └── featured_data.csv   # Output of feature_engineering.py
│
└── src/models/
    ├── xgb_model_mid.pkl       # Main XGBoost model (MSE objective)
    ├── xgb_model_lo.pkl        # 2.5th percentile quantile model
    ├── xgb_model_hi.pkl        # 97.5th percentile quantile model
    └── training_features.pkl   # Feature column list (must stay in sync)
```

## Prerequisites

- Python 3.8+
- PostgreSQL (running on port 5433)
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/keithpotz/Election-Perdiction.git
cd Election-Perdiction
```

### 2. Configure database and API key

Edit `src/config.py`:

```python
DB_CONNECTION_STRING = "postgresql://username:password@localhost:5433/election_db"
FRED_API_KEY = "your_fred_api_key_here"
```

Or set the environment variable:

```bash
export DB_CONNECTION_STRING='postgresql://username:password@localhost:5433/election_db'
```

### 3. Run the data pipeline (in order)

```bash
# Clean and merge raw vote data
python src/data_cleaning.py

# Build ML features (fetches CPI + consumer confidence from FRED)
python src/feature_engineering.py

# Train XGBoost models (saves 3 pkl files to src/models/)
python src/modeling.py

# Load cleaned data into PostgreSQL
python src/database.py
```

### 4. Launch the dashboard

```bash
streamlit run src/app.py
```

## Model Details

| Setting | Value |
|---|---|
| Algorithm | XGBoost |
| Training rows | 71,840 county-level records (2000–2024) |
| Validation | Leave-One-Election-Out (LOEO) |
| Sample weighting | Recency-weighted (2024 = 5x, 2000 = 1x) |
| Target | Candidate vote share (0–100%) |

**Features:** `year`, `totalvotes`, `gdp_growth`, `unemployment`, `inflation`, `consumer_confidence`, `is_incumbent`, `incumbent_on_ballot`, `consecutive_terms`, `state_lean`, `county_lean`, `turnout_delta`, one-hot state, one-hot party

## Data Sources

- [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) — State-level results 1976–2020
- [MIT MEDSL County Results](https://electionlab.mit.edu/data) — County-level results 2000–2024
- [FRED API](https://fred.stlouisfed.org/) — GDP, unemployment, CPI (CPIAUCSL), consumer confidence (UMCSENT)
- [Silver Bulletin](https://www.silverscaling.com/) — Trump approval polling (manual CSV download)

## Troubleshooting

**`could not connect to server` on startup**
PostgreSQL isn't running, or is on the wrong port. This project expects port `5433`. Check your running instances and update `DB_CONNECTION_STRING` in `src/config.py` or your environment variable to match.

**`relation "elections" does not exist`**
The database tables haven't been created yet. Run `python src/database.py` to bulk-insert the cleaned data and initialize the schema.

**`FileNotFoundError: featured_data.csv`**
The feature engineering step hasn't been run yet, or was run out of order. Follow the pipeline order strictly: `data_cleaning.py` → `feature_engineering.py` → `modeling.py` → `database.py`.

**`KeyError` or feature mismatch when making a prediction**
The model `.pkl` files are out of sync with `training_features.pkl`. This happens when `feature_engineering.py` is updated but the models aren't retrained. Re-run `python src/modeling.py` to regenerate all three model files.

**`FRED API` returns no data / `KeyError` on economic indicators**
Your FRED API key is missing or invalid. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) and set it in `src/config.py` or as `FRED_API_KEY` in your environment.

**Approval rating chart shows no data**
The Silver Bulletin approval CSV must be downloaded manually and placed at `src/data/polling_data/Silver Bulletin Trump approval polls - Sheet1.csv`. The filename must match exactly.

**`streamlit: command not found`**
Streamlit isn't installed or your virtual environment isn't activated. Run `pip install -r requirements.txt` inside your active environment.

## Contributions

Open source — forks and pull requests welcome.
