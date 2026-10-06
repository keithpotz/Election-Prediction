# Methodology Changelog

Every change to how forecasts are produced gets a dated entry here, *before* the next
forecast built with it is published. Forecast snapshots in `forecasts/2026/` record which
`model_version` produced them.

## 2026.10.06: first published 2026 midterm forecast

**Senate**: all 35 races (33 Class II seats, plus the Florida and Ohio specials).
- Race margin = linear model on:
  - partisan lean: Cook-PVI-style, 75/25 weighting of the last two presidential elections before the race
  - incumbency: +1 D, −1 R, 0 open; appointed incumbents count ½
  - national environment: House popular vote, generic ballot for 2026
  - incumbent strength: how much the incumbent beat their state's lean last time
- Trained on contested regular races 2000–2024 (406 races).
- Backtest, each cycle predicted without seeing it: 88% of winners called correctly, typical miss ±13 pts per race, ±4.6 pts shared nationally.
- Chamber control comes from 20,000 correlated simulations. Each one draws one national error shared by every race, plus an independent per-race error. The national error includes ±3 pts of generic-ballot polling error.
- Holdover seats come from 2022/2024 results and are checked against the actual 47 D / 53 R Senate.
- Open seats, appointments, incumbent party and expert ratings are scraped from Wikipedia and refreshed daily.

**House**: national seats-votes line (Dem seat share vs. House popular-vote margin), trained on 1996–2024. Backtest error is ±9.5 seats, and ±3 pts of generic-ballot polling error is added in simulation.

**Known gaps**:
- No race-level polling yet.
- Mid-decade redistricting since 2024 isn't reflected in the House model.
- Nebraska is modeled as D vs. R even though the main challenger is an independent.
