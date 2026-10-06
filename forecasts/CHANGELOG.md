# Methodology Changelog

Every change to how forecasts are produced gets a dated entry here, *before* the next
forecast built with it is published. Forecast snapshots in `forecasts/2026/` record which
`model_version` produced them.

## 2026.10.06b: polls added; every Senate, governor, and House race (roadmap Phases 1–2)

**Polls (Phase 1)**
- 2026 general-election polls are scraped from each Senate and governor race's Wikipedia page (776 polls, 59 of 71 races). Races with no polls are all safe seats.
- Pollster ratings are computed from 538's archive of 18,000 final-stretch polls with known results (1998–2022). Each pollster is scored against the other pollsters in the same race.
  - Partisan lean (house effect) persists across cycles (r = 0.56).
  - Accuracy mostly doesn't (r = 0.12), so only weak track records are down-weighted.
- Polls paid for by a campaign or party (marked D/R/I) lean about 4 pts toward their sponsor. They get half weight and count as half a poll.
- House-effect and sponsor corrections are applied at half strength. In the backtest they were neutral (6.02 vs 6.03 RMSE). They're kept as a guard because 46% of 2026 polls are sponsored, vs 11% historically.
- Averages weight recency (14-day half-life, 120-day cutoff), sample size, and quality, and cap any single pollster's influence.
- Polling-average backtest on 675 Senate/governor races, out of sample:

| Days out | Typical miss (RMSE) | Correct winner | Shared cycle miss (sd) |
|---|---|---|---|
| 0 | 6.0 pts | 91% | 2.7 pts |
| 28 | 7.2 pts | 89% | 3.6 pts |

**Blend (Phase 2)**
- Each race combines the fundamentals prior and the polling average by inverse-variance weighting.
- Poll uncertainty is fitted from the backtest: sqrt(15.8 + 39/effective polls + 0.20 × days to go). Weight shifts toward polls as they accumulate and election day nears, with no hand-set schedule.
- Senate validation, out of sample: blend RMSE 6.46 vs 7.07 for polls alone and 12.4 for fundamentals alone, at 28 days out. The blend wins at every horizon.
- Governors validation: blend 7.12 vs polls 7.29 at 28 days. Fundamentals alone are weak for governors (79% of winners), so governor forecasts are mostly poll-driven.

**New: governors.** The fundamentals model is trained on 252 polled governor races from the archive (2000–2022): lean, incumbency, national environment, and incumbent strength.

**New: House, all 435 districts.**
- Inputs are Wikipedia's 2026 Cook PVI on the lines actually in use this year (including the 9 states that redrew), plus incumbents and candidates.
- District margin = 1.95 × PVI + 3.7 × incumbency − 0.8, shifted 1:1 by the generic ballot vs 2024. This is calibrated on 2024 results in the 41 states with unchanged lines.
- District error is ±5.7 pts, from applying the model to 2022 (out of year), where it called 94.6% of winners. One-party ballots and unopposed races are settled.
- District polls aren't included yet.

**Joint simulation.** Every simulation draws one national error shared by all Senate, governor, and House races. Each race's exposure depends on how much it relies on polls vs fundamentals. Chamber control is a true joint probability. For Senate control, Osborn (I, Nebraska) is not counted toward the Democratic caucus.

**Known gaps:**
- No House district polls.
- Wikipedia poll tables can mix in three-way or hypothetical matchups (e.g. Idaho).
- Poll margins are D−R of all respondents, while fundamentals are two-party margins (a small scale difference).

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
