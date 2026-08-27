# Election Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql&logoColor=white)
![FRED API](https://img.shields.io/badge/Data-FRED%20API-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A non-partisan, data-driven US election prediction dashboard built with Python and Streamlit. Uses XGBoost trained on county-level historical results (2000–2024) combined with live economic, polling, and approval data to forecast state-by-state presidential outcomes and Electoral College totals plus a 2026 midterm House/Senate projection.

## Features

- **Interactive Dashboard** — 5-tab Streamlit app: Home, Historical Results, Make a Prediction, Prediction History, 2026 Midterms
- **Live Data** — Pulls GDP, unemployment, CPI, and consumer confidence from the FRED API, plus presidential approval and the national generic congressional ballot from Silver Bulletin's live poll database, all in real time (no manual downloads required)
- **XGBoost Model** — Trained on 71,840+ county-level rows with recency-weighted sampling; national-conditions inputs (economy, incumbency context, midterm environment) are modeled separately as a transparent, hand-calibrated national swing rather than fed into the tree, since they're election-year-constant values with too little independent variation for a per-county model to learn from
- **Confidence Intervals** — Three quantile models (median, 2.5th, 97.5th percentile) for real quantile-regression CIs on every state prediction
- **2026 Midterm Projections** — A national House seats-votes swing model and a per-state Senate race model (state lean + incumbency + national environment), cross-checked against real open-seat status and a consensus of ~12 professional forecasters scraped from Wikipedia
- **PostgreSQL Backend** — Cleaned historical data stored and queried from a local PostgreSQL database
- **Prediction History** — Save predictions and track EC vote trends over time

## Project Structure

```
Election-Prediction/
│
├── src/
│   ├── app.py                  # Streamlit dashboard (5 tabs)
│   ├── config.py               # DB connection string + FRED API key
│   ├── data_cleaning.py        # Merges raw CSVs, outputs cleaned_polls.csv
│   ├── feature_engineering.py  # Builds ML features, outputs featured_data.csv
│   ├── modeling.py             # Trains XGBoost models, saves to src/models/
│   ├── fundamentals.py         # National-conditions swing adjustment (economy, incumbency, midterm carryover)
│   ├── midterms.py             # 2026 House/Senate projection models
│   ├── race_status_2026.py     # Scrapes real 2026 Senate open-seat/consensus-rating data from Wikipedia
│   ├── database.py             # Bulk-inserts cleaned data into PostgreSQL
│   ├── data_sources.py         # Live FRED + Silver Bulletin (approval, generic ballot) data fetching
│   ├── electoral_votes.py      # EV allocations and EC calculation
│   └── data/
│       ├── polling_data/
│       │   ├── complete_data.csv           # MIT Election Lab state-level 1976–2020
│       │   ├── 2024president.csv           # 2024 state-level results
│       │   ├── countypres_2000-2024.csv    # County-level presidential results 2000–2024
│       │   ├── house_1976_2024.csv         # MIT/MEDSL US House results 1976–2024
│       │   ├── senate_1976_2024.csv        # MIT/MEDSL US Senate results 1976–2024
│       │   ├── senate_race_status_2026.csv # Cached scrape: 2026 open seats + consensus ratings
│       │   ├── GDP.csv / unemployment.csv  # Local FRED economic data
│       │   └── Silver Bulletin Trump approval polls - Sheet1.csv  # Offline fallback only; live fetch is primary
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

### 3. Get the midterm data files (one-time, manual)

`house_1976_2024.csv` and `senate_1976_2024.csv` power the 2026 Midterms tab. They come from Harvard Dataverse and require accepting a one-time "guestbook" form on the site (no API access exists), so they aren't fetched automatically:

- House: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
- Senate: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PEJ5QU

Download each `.tab` file and place it in `src/data/polling_data/` under the filenames above (the House file is tab-delimited despite its extension and needs no manual reformatting — `feature_engineering.py`/`midterms.py` handle it as-is). If these files are missing, the presidential model and its Home/Prediction/History tabs still work fine — only the Midterms tab needs them.

### 4. Run the data pipeline (in order)

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

### 5. Launch the dashboard

```bash
streamlit run src/app.py
```

## Model Details

| Setting | Value |
|---|---|
| Algorithm | XGBoost (3 models: median + 2.5%/97.5% quantile) |
| Training rows | 71,840+ county-level records (2000–2024) |
| Validation | Random 80/20 split |
| Sample weighting | Recency-weighted (2024 = 5x, 2000 = 1x) |
| Target | Candidate vote share (0–100%) |

**Tree-model features** (things that genuinely vary row-by-row): `year`, `totalvotes`, `is_incumbent`, `state_lean`, `county_lean`, `turnout_delta`, one-hot state, one-hot party.

**National-conditions inputs** (`gdp_growth`, `unemployment`, `inflation`, `consumer_confidence`, `incumbent_on_ballot`, `consecutive_terms`, `prior_midterm_house_margin`, `approval`) are deliberately **not** fed into the tree — they're identical for every county in a given election year, so across 2000–2024 there are only 7 historical values behind each one, far too few for a per-county model to learn a reliable relationship from. Instead they're applied as a transparent, hand-calibrated national swing on the incumbent party's vote share (`src/fundamentals.py`), sanity-checked against real historical election outcomes.

## 2026 Midterms

- **House**: a national "seats-votes" swing model (Dem seat share as a function of the national generic-ballot margin), fit on ~25 historical House cycles, with a Monte Carlo simulation over the model's residuals for a control probability.
- **Senate**: a per-state model of margin ~ presidential state lean + incumbency + national environment, evaluated for each of the ~33 Class II seats up in 2026 (derived from the data itself, not a hardcoded roster). Deliberately scoped to just the contested seats, not a full 100-seat chamber-control claim, since the other 67 seats' current holders aren't derivable from historical general-election results alone.
- Real 2026 open-seat/retirement status and a consensus rating averaged across ~12 professional forecasters (Cook Political Report, Sabato's Crystal Ball, Inside Elections, etc.) are scraped from Wikipedia and shown alongside the model's own prediction, not blended into it — so you can see where the structural model and expert consensus agree or disagree.

## Data Sources

- [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) — Presidential, House, and Senate results 1976–2024 (state, county, and district level)
- [FRED API](https://fred.stlouisfed.org/) — GDP, unemployment, CPI (CPIAUCSL), consumer confidence (UMCSENT)
- [Silver Bulletin](https://www.natesilver.net/) — Presidential approval and generic congressional ballot, fetched live from Silver Bulletin's public poll database (falls back to a manually-downloaded CSV if offline)
- [Wikipedia](https://en.wikipedia.org/wiki/2026_United_States_Senate_elections) — 2026 Senate race status (open seats) and forecaster consensus ratings

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

**Approval / generic ballot chart shows no data**
Both are fetched live from Silver Bulletin by default and need internet access. Approval falls back to a manually-downloaded CSV at `src/data/polling_data/Silver Bulletin Trump approval polls - Sheet1.csv` (filename must match exactly) if the live fetch fails; the generic ballot has no offline fallback.

**2026 Midterms tab shows an error or is missing data**
`house_1976_2024.csv`/`senate_1976_2024.csv` are missing — see Setup step 3 above. The Senate race-status scrape (open seats/consensus ratings) requires internet access on first run; after that it's cached to `senate_race_status_2026.csv` and doesn't need to re-fetch.

**`streamlit: command not found`**
Streamlit isn't installed or your virtual environment isn't activated. Run `pip install -r requirements.txt` inside your active environment.

## Contributions

Open source — forks and pull requests welcome.
