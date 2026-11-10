# The Ledger Forecast

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Model](https://img.shields.io/badge/Model-Polls%20%2B%20Fundamentals-orange)
![Simulations](https://img.shields.io/badge/Simulations-20%2C000-purple)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-lightgrey)

**Polls first. No betting lines. Every forecast on the record.**

A non-partisan, data-driven US election forecast. It covers every 2026 Senate race, all 36 governor's races, and all 435 House districts, and grades itself publicly against the results. It runs as a Streamlit dashboard and publishes on The Ledger (Substack).

The project started as an XGBoost model of presidential results. For 2026 it was rebuilt as a **polls-primary forecast**:
- It averages every public poll and corrects for each pollster's track record.
- It blends the poll average with a fundamentals model, weighting each by how accurate it has been.
- It simulates the election 20,000 times with a shared national polling error.

The full method is in [`forecasts/METHODOLOGY.md`](forecasts/METHODOLOGY.md), also published on Substack as [How our 2026 Midterm Forecast Works](https://theledgerforcast.substack.com/p/how-our-2026-midterm-forecast-works).

---

## How the 2026 forecast did

*Preliminary: 486 of 506 races graded as of November 10, 2026. The rest (Alaska, Mississippi Senate, the Louisiana runoff and some House seats) are still being counted. Full report: [`forecasts/GRADES_2026.md`](forecasts/GRADES_2026.md).*

**Chamber control:** Democrats won both chambers, and the final forecast (Nov 3) favored them in both.

| | Final forecast | Result |
|---|---|---|
| Senate | Democrats 67.5% (median 52 seats, 90% range 47–56) | Democrats, 54 seats so far ✓ |
| House | Democrats 91.9% (median 234 seats, 90% range 215–259) | Democrats, 257 seats so far ✓ |

**Race by race (final forecast):**

| Method | Races | Winners right | Brier (lower is better) | Log loss |
|---|---|---|---|---|
| **The Ledger Forecast** | 486 | **93%** | **0.050** | **0.177** |
| Expert consensus (Cook, Sabato, Inside Elections, …) | 432 | 94% | 0.055 | 0.195 |
| Polls only (our corrected average, polled races) | 59 | 95% | 0.045 | 0.168 |
| Fundamentals only | 67 | 87% | 0.104 | 0.329 |

- **Beat the expert consensus** on probability quality: overall, in the House (0.050 vs. 0.055) and in the Senate (0.046 vs. 0.053). The experts did better on governors (0.047 vs. 0.054).
- **Got better every week.** Brier improved from about 0.057 in early October to 0.050 on election day.
- **The 99%+ calls held:** 325 races, 99% right.

**What went wrong** (stated plainly, since that's the point of grading):
- **Every miss went the same way.** All 33 misses were races we had going Republican that Democrats won. Democrats beat our forecast margin by about 5 points on average, in the House, the Senate and governors alike.
- **Our poll correction backfired.** We pulled state polls 1.8 points toward Republicans, because polls overstated Democrats in 2016–2024. In 2026 they understated them. In Senate races, raw averages missed by D+3.8 and our corrected averages by D+5.1.
- **Too confident in unpolled seats.** Three House seats with no polls (FL-6, FL-19, PA-11) were at 99.6%+ Republican, and Democrats won all three.
- **Ranges too narrow.** 79% of Senate and governor results fell inside our 90% ranges.
- **Too cautious on the chambers.** Nate Silver's FLIPR (benchmark only, never an input) had Democrats at 82% for the Senate and 96% for the House. Same national error, seen from the other side.

The fixes are tracked in [`post-election-plan.md`](post-election-plan.md):
- rethink the correction toward Republicans
- widen the shared national error
- cap win chances in unpolled races

Every change goes in the [changelog](forecasts/CHANGELOG.md) before the next forecast uses it.

### Verifying the record

Every daily 2026 forecast is in `forecasts/2026/`, one set of files per day, each stamped with its model version and publish time. Here is what that does and doesn't prove:

- **The 2026 files were uploaded after the election.** They were kept locally during the campaign and added to this repository in November 2026. The dates in the filenames and inside the files are our own record. GitHub can't confirm when they were created.
- **The method was public before the election.** [How our 2026 Midterm Forecast Works](https://theledgerforcast.substack.com/p/how-our-2026-midterm-forecast-works) was published on Substack on October 13, 2026, three weeks before election day. Substack, not us, dates it. It describes the method we graded, including the correction toward Republicans that backfired.
- **The forecast numbers were posted on X (Twitter) almost daily during the campaign** at [@keithpotz](https://x.com/keithpotz), as share images from the dashboard. X, not us, dates those posts, and the numbers in them match the snapshot files for the same days.
- **From now on, each forecast is uploaded the day it's published,** so GitHub's own timestamps show it existed before the election it forecasts.

**Historical backtest** (every polled Senate and governor race, 2000–2022, forecast 28 days out using only other cycles' data): RMSE 6.6 points, 89% of winners, Brier 0.067. See [`forecasts/CALIBRATION.md`](forecasts/CALIBRATION.md).

---

## How it works

1. **Polling averages.** Every poll is weighted by:
   - **recency:** weight halves every 14 days
   - **sample size**
   - **pollster rating:** built from about 18,000 polls with known results, 1998–2024
   - **firm:** one firm can't dominate a race by polling often

   Each pollster's historical lean is half-corrected, and campaign-sponsored polls count half.
2. **Fundamentals.** For each race:
   - **Inputs:** the state's partisan lean, incumbency, the incumbent's past over- or under-performance, and the national environment from the generic ballot.
   - **Training data:** Senate and governor races since 2000.
   - **House:** districts use Cook PVI on the 2026 maps, including the nine states that redrew.
3. **Blend.** Polls and fundamentals are combined by inverse-variance weighting. Well-polled races lean on polls; unpolled races lean on fundamentals.
4. **Simulate.** 20,000 elections, each with three errors:
   - a national error shared by every race
   - a regional error shared by nearby states
   - each race's own error

   From these come chamber odds, seat ranges, the tipping-point race and the most common paths to a majority.
5. **Publish and grade.** Each day's forecast is saved to `forecasts/2026/` with its model version. After the election every snapshot is scored against official results, alongside the expert consensus, polls-only and fundamentals-only baselines.

### Independence policy

No data from organizations with business ties to prediction or betting markets. That excludes Silver Bulletin, RealClearPolitics, Decision Desk HQ and VoteHub. Expert ratings are shown for comparison but never feed the model. No races are adjusted by hand. See `src/independence.py`.

---

## Features

The Streamlit dashboard (`streamlit run src/app.py`) has these tabs:

| Tab | What it shows |
|---|---|
| 📊 2026 Overview | Chamber odds, seat ranges, key races |
| 🏛️ Senate / 🏘️ House / 🏢 Governors | Every race with its polling average, win chance over time, the snake chart, the tipping point and paths to a majority |
| 📋 Polls | Every poll used, with its corrections and weight, plus the pollster ratings |
| 🎚️ What if? | Shift the polls from D+6 to R+6 and see what happens |
| 🗳️ Election Night | Record calls as they come in; the 20,000 simulations re-weight live |
| 🏁 Results | Grading: chamber outcomes, scorecard vs. alternatives, calibration, margin errors, polling-error map, pollster scorecard, every miss |
| 🔎 Comparisons | Our forecast vs. each expert forecaster and each pollster |
| 📈 Trends | Approval, generic ballot, economy, historical midterm seat losses |
| 📣 Post Builder | Social posts and share images (race lists, results report cards) |
| 📏 Track Record | Forecast history over the campaign |
| 🗓️ 2028 Outlook / 🔮 2028 Prediction / 📜 Historical Results / 📈 Prediction History | The presidential model (2024 county results plus a national swing) and historical results |

Other tools:
- **Daily update:** `python src/daily_update.py` refreshes everything in one command:
  - race status, polls and national inputs
  - the data-quality checks
  - publishing the day's snapshot

  `run_daily_update.cmd` wraps it for Windows Task Scheduler.
- **Grading:** `python src/grade_2026.py` scores every snapshot against `forecasts/2026/results_2026.csv` and writes `forecasts/GRADES_2026.md`.
- **Calibration:** `python src/calibration.py` runs the 2000–2022 backtest and writes `forecasts/CALIBRATION.md`.
- **Writing:**
  - `src/weekly.py` explains what moved between two snapshots
  - `src/launch_post.py` drafts a full Substack forecast post
  - `src/share_cards.py` makes race cards for social media
- **Rasmussen check:** `python src/rasmussen_check.py` finds Rasmussen polls that Wikipedia hasn't listed yet.

---

## Project structure

```
Election-Prediction/
├── forecasts/
│   ├── 2026/                     # Daily snapshots (<date>_senate/governor/house + summary.json),
│   │                             #   topline_log.csv, results_2026.csv, grades_2026.csv, share cards
│   ├── METHODOLOGY.md            # How the Ledger Forecast works
│   ├── CALIBRATION.md            # 2000–2022 backtest and calibration
│   ├── GRADES_2026.md            # 2026 report card
│   ├── CHANGELOG.md              # Every method change, dated, with its effect
│   └── calibration/              # Backtest outputs and chart
├── src/
│   ├── app.py                    # Streamlit dashboard
│   ├── dashboard_*.py            # Tab views: 2026, polls, election night, results
│   ├── daily_update.py           # One-command refresh + checks + publish
│   ├── forecast_2026.py          # Every 2026 race: blend + 20,000 simulations
│   ├── blend.py                  # Polls/fundamentals inverse-variance blend, shared error sizes
│   ├── simulations.py            # Tipping point, paths to a majority, election-night re-weighting
│   ├── polls_2026.py             # Senate/governor poll scraper (Wikipedia)
│   ├── polls_house_districts.py  # House district poll scraper
│   ├── polls_average.py          # Weighted, corrected polling averages + backtest
│   ├── pollster_ratings.py       # Pollster accuracy and lean ratings
│   ├── national_polls.py         # Generic ballot and approval (no market-tied sources)
│   ├── midterms.py / governors.py / house_districts.py   # Fundamentals models
│   ├── race_status_2026.py       # Nominees, open seats, expert ratings (Wikipedia)
│   ├── calibration.py            # Historical backtest
│   ├── grade_2026.py             # Grading against results
│   ├── publish_forecast.py       # Writes the daily snapshot
│   ├── independence.py           # Source-exclusion rules
│   ├── posts.py / result_posts.py / share_cards.py / weekly.py / launch_post.py  # Publishing
│   ├── brand.py                  # Publication and model names
│   ├── fundamentals.py / modeling.py / feature_engineering.py / data_cleaning.py  # Presidential model
│   ├── database.py               # PostgreSQL loader
│   └── data/
│       ├── polling_data/         # MIT results 1976–2024, FiveThirtyEight poll archive, 2024 polls,
│       │                         #   pollster ratings, 2026 race status and expert ratings
│       └── polls_2026/           # 2026 polls, manual polls, approval, generic-ballot log, data checks
├── tests/                        # pytest suite
├── post-election-plan.md         # What's next after 2026
└── election-forecaster-roadmap.md
```

---

## Setup

### Prerequisites
- Python 3.13 (other recent 3.x versions likely work)
- PostgreSQL on port 5433 (only for the historical/presidential tabs)
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
- Google Chrome, which `kaleido` uses to export share images

```bash
git clone https://github.com/keithpotz/Election-Prediction.git
cd Election-Prediction
pip install -r requirements.txt
```

### Config
Create `src/config.py`. It's in `.gitignore`, so your keys stay local:

```python
DB_CONNECTION_STRING = "postgresql://username:password@127.0.0.1:5433/election_db"
FRED_API_KEY = "your_fred_api_key_here"
```

### Historical House and Senate results (one-time, manual)
`house_1976_2024.csv` and `senate_1976_2024.csv` come from Harvard Dataverse, which requires a one-time guestbook form (there's no API). Download them to `src/data/polling_data/` under those names:
- House: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
- Senate: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PEJ5QU

### Run it

```bash
python src/daily_update.py      # refresh polls and race data, run checks, publish today's snapshot
streamlit run src/app.py        # open the dashboard
python src/grade_2026.py        # grade the forecasts once results are entered
python -m pytest tests          # run the tests
```

For the presidential tabs, build the historical pipeline once, in this order:

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/modeling.py
python src/database.py
```

---

## Data sources

- **[Wikipedia](https://en.wikipedia.org/wiki/2026_United_States_Senate_elections):**
  - 2026 race polls (Senate, governor, House districts), compiled from pollsters' own releases
  - nominees, open seats, expert ratings
  - presidential approval polls
- **Pollsters' own releases:** polls not yet on Wikipedia, entered by hand with links (e.g. Rasmussen).
- **FiftyPlusOne and NBC News:** generic-ballot averages, read from Wikipedia's 2026 House elections page.
- **[UCSB American Presidency Project](https://www.presidency.ucsb.edu/):** presidential approval, plus the historical approval and midterm seat-change table.
- **[FiveThirtyEight's pollster-ratings archive](https://github.com/fivethirtyeight/data/tree/master/pollster-ratings):** historical polls with known results, 1998–2022, used for the pollster ratings. 2024 polls were collected separately.
- **[MIT Election Data + Science Lab](https://electionlab.mit.edu/data):** presidential, House and Senate results, 1976–2024.
- **[FRED](https://fred.stlouisfed.org/):** GDP, unemployment, CPI, consumer confidence.
- **Official state results** (via Wikipedia's cited race pages): the 2026 results used for grading.

---

## Troubleshooting

**`daily_update.py` crashes on reading tables:** install `lxml` (`pip install lxml`). Wikipedia parsing needs it.

**Wikipedia returns 403:** requests need a descriptive `User-Agent`. The scrapers set one; if you add a new one, reuse `race_status_2026._fetch_html()`.

**Share images fail to export:** `kaleido` needs Google Chrome installed locally.

**`could not connect to server`:** PostgreSQL isn't running or isn't on port 5433. Check `DB_CONNECTION_STRING` in `src/config.py`. Only the historical and presidential tabs need the database.

**`FileNotFoundError: featured_data.csv` or a model feature mismatch:** run the presidential pipeline in order (`data_cleaning` → `feature_engineering` → `modeling` → `database`).

**Midterm tabs missing history:** `house_1976_2024.csv` / `senate_1976_2024.csv` are missing. See Setup.

**`streamlit: command not found`:** activate your virtual environment and run `pip install -r requirements.txt`.

---

## License

GPL-3.0. See [LICENSE](LICENSE). Forks and pull requests are welcome.
