# Election Forecaster Roadmap
### From portfolio project to a real public forecasting operation

Goal: build Election-Prediction into something that functions like a real forecaster (Silver / Economist / Split Ticket style), with a public track record, honest calibration, and a real methodology — not a timeline-driven sprint, a standing project to work through.

---

## Phase 1: Get real polling data into the pipeline

This is the foundation. Without it, the model is fundamentals-only, not a forecast.

- [ ] Pick a live polling data source
  - [ ] RealClearPolitics public averages
  - [ ] Individual pollster APIs / public data feeds (e.g. YouGov)
  - [ ] Build a scraper if no clean API exists
- [ ] Design a polling data schema (pollster, date, state, sample size, methodology, result)
- [ ] Build a pollster rating / weighting table
  - [ ] Use FiveThirtyEight's published pollster-ratings methodology as a reference (legacy writeups still public)
  - [ ] Weight by recency, sample size, historical accuracy
- [ ] Build a house-effect correction layer
  - [ ] Track each pollster's historical deviation from final results
  - [ ] Apply a correction before polls feed into the model

---

## Phase 2: Restructure the model architecture

Flip from fundamentals-primary to polling-primary, fundamentals-as-prior.

- [ ] Build the state-level weighted polling average as the primary signal
- [ ] Keep the existing XGBoost fundamentals model as a secondary "prior" layer
- [ ] Design the blend function: fundamentals dominate far from election day, polls dominate close to it
- [ ] Decide the blend-weight schedule (e.g. linear ramp, or tied to days-until-election)
- [ ] Re-validate with LOEO once the blended model is in place

---

## Phase 3: Fix the uncertainty modeling

This is what separates "point estimate with decoration" from a real probabilistic forecast.

- [ ] Model correlated state-level error (shared national "miss direction" applied across all states in simulation, not independent per-state intervals)
- [ ] Add time-decay to forecast uncertainty (confidence interval narrows as election day approaches — separate from training recency weighting)
- [ ] Build and publish the calibration report from LOEO backtests
  - [ ] Bucket predicted probabilities (0-10%, 10-20%, etc.)
  - [ ] Check actual win rate per bucket against predicted
  - [ ] Turn this into a publishable chart
- [ ] Add a dumb-baseline comparison (e.g. "last election + incumbency" or simple polling average) and report honestly whether the model beats it

---

## Phase 4: Build the public-facing track record

Start this in parallel with Phase 1-3, not after. The track record is the actual credibility asset.

- [ ] Start publishing predictions for 2026 midterm races (Senate, Governors) on the current architecture, with clear caveats about what it does/doesn't account for yet
- [ ] Set up a public, timestamped prediction log (signed commit, pinned post, or similar — something that proves no post-hoc editing)
- [ ] Publish a dated methodology changelog every time the model changes
- [ ] Grade past predictions publicly and honestly, including the misses
- [ ] Add tipping-point state analysis once correlated-error modeling (Phase 3) is in place

---

## Phase 5: Build visibility in the forecasting space

- [ ] Set up the public writing home (Substack or similar) for methodology writeups and race-by-race post-mortems
- [ ] Cross-post calibration charts and writeups to communities where this audience already exists (Hacker News, r/fivethirtyeight-adjacent subreddits, political-data X/Twitter)
- [ ] Engage with existing independent forecasters (Strength In Numbers, Split Ticket, Silver Bulletin) — for visibility and feedback, not immediate competition
- [ ] Revisit after one full cycle (2026 midterms) graded: what worked, what needs to change before 2028

---

## Notes

- This supersedes the "freelance portfolio piece" framing — this is now the primary direction for the project.
- Model details as of last review: XGBoost, 71,840 county-level rows (2000-2024), LOEO cross-validation, recency-weighted (2024=5x, 2000=1x), three quantile models (median/2.5/97.5) for CI, live FRED economic data, Silver Bulletin approval data, PostgreSQL backend, Streamlit dashboard.
- GitHub: keithpotz/Election-Prediction
