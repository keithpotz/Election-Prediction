# Methodology Changelog

Every change to how forecasts are produced gets a dated entry here, *before* the next
forecast built with it is published. Forecast snapshots in `forecasts/2026/` record which
`model_version` produced them.

## 2026.10.13: Poll-table fixes, hand-entered polls, second approval source; tipping point and election night

**Poll tables read correctly in more races.**
- **Nominees.** When a Wikipedia race page has no polling-average table naming the nominees, the scraper now uses the confirmed nominee list. Before, it took the newest table with exactly one Democrat and one Republican. On Alaska's governor page that was a 2025 Peltola–Begich hypothetical, so the race had no usable polls and ran on fundamentals alone.
- **Ranked-choice rounds.** A footnoted round like "3[j]" is now read as round 3, so the final head-to-head round is kept.
- **Split fields.** A first-choice table is skipped when other candidates from the nominees' parties average 10% or more (Alaska governor: two more Republicans at about 21%). So is any row where the two nominees total under 60%. Neither is a head-to-head. This also removes 63 California governor polls taken before the June top-two primary. All were more than 120 days old, so they already counted for nothing.
- **Hand-entered polls.** `src/data/polls_2026/manual_polls.csv` holds polls Wikipedia hasn't listed, each linked to the pollster's own release. They are merged in at load time. Once Wikipedia lists the same poll (same race and pollster, end date within 3 days), the scraped copy is used. The first four are Rasmussen's Michigan, North Carolina and New Hampshire Senate polls and SurveyUSA's Oct 7–11 North Carolina poll for WRAL/WSOC.

**Second approval source.** Wikipedia's approval page skips most months from August 2025 to August 2026. The UC Santa Barbara American Presidency Project's table (academic, no market ties) now fills in the polls Wikipedia doesn't list: 44 polls. Approval feeds the Trends charts and the 2028 tab, not the 2026 forecast.

**New outputs** (no change to any race's odds):
- Each Senate race's and House district's chance of being the tipping point.
- The common paths to a Senate majority.
- The simulations, kept for election-night mode. That mode re-weights them by called races, and only calls in races the forecast had at 3–97% re-weight. Safe-seat calls are counted as seats but don't shift the odds.

**Effect today** (Oct 13): Alaska governor goes from 36% to 69% Democratic (three ranked-choice polls, Kreiss-Tomkins +4 to +10 in the final round). North Carolina Senate goes from 90% to 92%. Senate control is about 53% (was 54%), and House control is 83%.

## 2026.10.08: House incumbent strength; share cards; 2028 projection rebuilt

**House incumbent strength.** The district model now knows how much each running incumbent beat (or trailed) their district's partisan lean last time: their 2024 margin (clipped at ±60) minus 2 × Cook PVI. It is zero for open seats and for states redrawn for 2026.
- Calibrated on 2024 like the rest of the model: margin ≈ 1.97·PVI + 2.77·incumbency + 0.23·strength.
- The in-year district error falls from 4.64 to 4.43 pts.
- Incumbents who habitually outrun their districts (e.g. PA-1, Fitzpatrick: 62% → 48% D) now get credit for it, which closes much of the gap with expert ratings.

**Share cards.** Each Senate and governor race, and each competitive House seat (10–90%), gets a 1200×675 PNG in `forecasts/2026/cards/<date>/`. Cards are made on every daily update and from the dashboard. Each card states a probability, never a call. Each one is dated and versioned and carries no logos. It names its basis and says that no prediction-market data is used.

**2028 presidential projection rebuilt** (the old Prediction/Overview tabs). Three problems in the old model:
- **Missing states.** MEDSL's 2024 county file leaves the vote-method column blank for 8 states (NC, WI, UT, ID, RI, WA, WV, WY), and the old filter silently dropped their 2024 results. They are now included; North Carolina's vote-method rows are summed.
- **Leakage.** Several features (county and state lean, turnout) were averaged over every election, including the one being predicted. Features now use only earlier elections.
- **Honest test.** Leave-one-election-out, 2004–2024, county two-party shares. The XGBoost model lost to "repeat the last election" in 4 of 6 years. Even when given the true national swing, it beat uniform swing on average only unweighted (8.36 vs 8.47 RMSE). It lost when counties were weighted by votes (6.94 vs 6.61), and it was worse in both 2020 and 2024.
- **Result.** The 2028 projection is now the 2024 county results plus one national swing from the economic and approval inputs. Uncertainty is ±(5.2 state, 4.0 national) pts of margin; 5.2 is uniform swing's historical state-level error.
- **Tab fixes.** The Quick Projection said the GOP had held the White House for 2 consecutive terms; it is 1. "Incumbent on the ballot" now defaults to off for 2028. A live generic ballot of 0 no longer reads as missing.

**Effect today** (Oct 8, 26 days out): Senate D control 52% (was 53%); House 80%, median 230 D seats (90% range 208–262); 19 governorships. The 2028 change doesn't touch the 2026 forecast.

## 2026.10.07c: 2024 added to the history; state-poll lean correction; comparisons

**2024 added.**
- FiveThirtyEight's archive ends in 2022. 2024's polls and results are now collected from Wikipedia's race pages (`src/history_2024.py`): 482 Senate polls, 80 governor polls, and 136 national presidential polls from the final 60 days, with certified results.
- 2024 now feeds the pollster ratings, every correction, and the backtest.
- In 2024 the Senate polls overstated Democrats by about 2.5 points, in 17 of 22 races, and national presidential polls by 2.3.
- No list of individual 2024 generic-ballot polls exists, so the national presidential miss stands in for that year in the generic-ballot history. The generic-ballot correction is essentially unchanged: 3.2 pts at four weeks out, 12 of 14 cycles.

**State and district poll lean is now corrected.** This was previously tested and rejected without 2024. With 2024 included, five variants were tested at four weeks out:

| Variant | Margin RMSE | Brier | Log loss |
|---|---|---|---|
| None | 6.72 | 0.0673 | 0.2155 |
| Full | 6.68 | 0.0698 | 0.2200 |
| Half | 6.64 | 0.0677 | 0.2153 |
| Recent-weighted, full | 6.57 | 0.0695 | 0.2192 |
| **Recent-weighted, half (adopted)** | 6.58 | 0.0673 | **0.2144** |

- The adopted version applies half of the historical average lean, weighting recent cycles more. That is 1.8 pts toward Republicans at four weeks out, applied to every Senate, governor, and House district polling average.
- It shrinks a 2024-style miss in the backtest from 4.4 to 3.0 pts.
- Calibration now has all 7 buckets in range at both horizons, up from 6 of 7 on election eve.
- Five variants were tried, so there is some risk of fitting noise; the two half-strength versions agree.

**Effect today:**
- Senate Dem control 66% → 53% (median 52 → 51 seats).
- House 82% → 79%.
- Dem governorships (median) 20 → 19.

**New on the dashboard:**
- A "What if the polls are off?" scenario that shifts every poll a further 3 or 6 pts toward Republicans. With a 2024-size extra miss: Senate D 28%, House 50%.
- A Comparisons tab with other forecasters' race ratings and Senate totals (market-tied ones excluded), and each pollster's lean versus our averages.

**Fix.** Bipartisan polling teams, such as the Fox News poll by Beacon Research (D) and Shaw & Co. (R), are no longer treated as Democratic-sponsored.

## 2026.10.07b: grading plan, expert comparison, independents and top-four races, environment cross-check

**Grading plan, set before the results** (`src/grade_2026.py`):
- Every published snapshot will be graded against official state results entered in `forecasts/2026/results_2026.csv`, with Brier score, log loss, winner accuracy, calibration, and misses.
- Baselines: polls only, fundamentals only, and the expert consensus. Expert ratings are converted to probabilities with a stated scale (Safe ≈ 97%, Likely ≈ 85%, Lean ≈ 70%).
- Snapshots now record each input's own uncertainty so the baselines can be scored.

**Expert consensus excludes market-tied forecasters.**
- The Senate and governor consensus now leaves out Decision Desk HQ, RealClearPolitics, and Silver Bulletin, matching the independence policy. The consensus is a comparison only and is never a model input.
- House district ratings were added from Wikipedia's state pages: 351 districts, from Cook, Sabato's Crystal Ball, Inside Elections, The Economist, and Split Ticket.
- The dashboard lists where the model disagrees most with the experts. Today the model is about 7 House seats more Democratic than the experts' implied total, with the largest gaps at strong individual incumbents such as PA-1 (Fitzpatrick). Adding incumbent-specific strength to the House model, as the Senate model already has, is the leading follow-up.

**Independents and top-four ballots.**
- Where the main challenger is an independent (Osborn in Nebraska, Achilles in Idaho), the fundamentals prior gets double the uncertainty, since it was fit on Democrat-vs-Republican races. These races now lean on their polls.
  - Nebraska: R+5.9 → R+1.4 (Osborn 42%).
  - Idaho: R+9.9 → R+6.9 (Achilles 18%).
- When several same-party candidates are on the general ballot and polls don't settle the matchup, the race page's primary results pick the leader. Alaska governor is now Kreiss-Tomkins vs. Wilson, who led the Republicans in the top-four primary.
- Unpolled races now show the confirmed nominees' names.

**Environment cross-check.**
- The national environment implied by our own race polls is compared with the generic-ballot polls, before lean correction on both sides so the comparison is like for like.
- Today Senate and House polls imply D+8.3 vs. D+8.45 from the generic ballot: consistent.
- Governor polls imply D+13.5 but are shown separately, because the governor structural model is weak.
- A gap over 3 pts raises a data warning.

## 2026.10.07: FiveThirtyEight-style pollster ratings; regional correlation

**Pollster ratings now follow FiveThirtyEight's published method** ("How Our Pollster Ratings Work," March 2023, read from the Internet Archive):
- **Simple plus-minus:** a poll's error versus the error expected from race type, sample size, days to election, and how many firms polled the race.
- **Advanced plus-minus:** blends error relative to other pollsters in the same race (weight *n*) with simple plus-minus (weight 3). Recent cycles count more (0.93 per year).
- **Herding penalty:** applied to pollsters whose results sit closer to the earlier polling average than sampling error allows.
- **Predictive plus-minus:** `(max(−2, APM + herding) × n + prior × 18) / (18 + n)`, with `prior = 0.66 − 0.57 × quality − 0.03 × min(18, n)`. Quality = 1 for AAPOR Transparency Initiative or Roper Center members.
- **Simplifications:** no strength-of-schedule iteration; field crowding enters through the expected-error regression.
- **Grading:**
  - On its own, the new rating did not predict which pollsters would be accurate later any better than the previous same-race-only rating. Results were mixed across 2014, 2016, and 2018 splits.
  - In the forecast backtests it was slightly better at every horizon: poll-average RMSE 6.025 vs 6.031 on election eve and 7.234 vs 7.246 at four weeks.
  - The full model at four weeks improved from 6.81 to 6.79 RMSE and log loss from 0.218 to 0.217. Calibration still has all 7 buckets in range.
  - Adopted on that basis.
- FiveThirtyEight also banned pollsters that bet on markets their polls could affect, which is consistent with this forecast's independence policy.

**Regional correlation.**
- Races in the same region now share an error component in the simulations. Regions: Northeast, Rust Belt, Plains, South, Mountain, Pacific.
- Size: ±1.2 pts at four weeks and ±1.3 on election eve, estimated by decomposing the out-of-sample backtest errors after removing the national miss.
- Each race's total uncertainty is unchanged, since the regional part is carved out of its own error, so per-race probabilities stay calibrated. Only the way races move together changes.

**Effect today:** essentially none at the topline: Senate Dem control 66%, House 82%, governors 20. The new ratings shift a few race averages slightly, and regional correlation widens seat ranges a little.

**Daily refresh.** `run_daily_update.cmd` runs `src/daily_update.py` and logs to `forecasts/2026/daily_update_output.log`. It is meant for a 7:00 AM Windows scheduled task through Nov 4.

## 2026.10.06e: independence policy, with no prediction-market-affiliated data sources

**Policy.** The forecast uses no data from organizations with business ties to prediction or betting markets. As checked on 2026-10-06:

| Source | Tie | Status |
|---|---|---|
| VoteHub | Data partnership with Kalshi (Midterms Hub, July 2026) | Removed as poll source |
| Silver Bulletin | Nate Silver advises Polymarket | Removed for generic ballot and approval |
| Decision Desk HQ | Runs live Polymarket/Kalshi odds displays in its election coverage | Excluded |
| RealClearPolitics | Added Polymarket odds to its 2026 forecast maps | Excluded |

YouGov, which is a pollster rather than a data source, showed no market tie and is kept. It is rated on its track record like every other pollster.

**Sources now:**
- **Senate, governor, and House district polls:** Wikipedia race pages and state House pages, which compile pollsters' own releases.
  - 795 Senate/governor polls.
  - 201 district polls across 85 districts, versus 84 polls across 53 districts from VoteHub.
- **Generic ballot:** average of the non-market-tied aggregators listed on Wikipedia, currently FiftyPlusOne (D+7.9) and NBC News (D+9.0), giving D+8.45. Wikipedia lists no individual 2026 generic-ballot polls.
- **Approval:** our own recency/sample-weighted average of the individual polls on Wikipedia.

**Robustness.**
- Failed page loads are retried, and any that still fail are flagged in the data checks. Previously a failed page silently gave a race zero polls.
- When an official candidate list can't settle a matchup, the race's polls are left out rather than using a matchup that contradicts the list. This currently applies to Alaska governor, where three Republicans are on the general ballot; that race runs on fundamentals alone.

**Accuracy impact.** The backtested accuracy is unchanged, since the backtest never used these sources. Comparing today's inputs, expected per-race poll error went from ±6.56 to ±6.65 pts.

**Effect today:**
- Senate Dem control 64% → 66%.
- House 84% → 82%: more district polls, partly offset by a lower generic-ballot input (D+8.4 vs D+8.8).
- Governors unchanged at a median of 20.

## 2026.10.06d: VoteHub as primary poll source; confirmed nominees; House district polls; data checks

**Poll sources.**
- VoteHub's poll database (votehub.com, CC BY 4.0) is now the primary source. Each poll links to the pollster's own release and carries structured dates, sample, voter screen, and an explicit partisan-sponsor flag.
- Wikipedia race pages fill in polls VoteHub lacks. Duplicates are removed by matching pollster, date, and results; this caught 462 polls that appear in both sources.
- Current file: 597 VoteHub polls plus 330 Wikipedia-only polls.

**Nominees confirmed.**
- Matchups now come from the candidate lists (with party labels) on Wikipedia's Senate and governor overview pages, instead of being inferred from poll tables.
- Where a party has several candidates on the general ballot, as in Alaska's top-four system, the one appearing in recent polls is used.
- Polls of other matchups (hypotheticals, withdrawn candidates) are dropped.
- This corrected Alaska governor, where a stale Peltola vs. Begich matchup had been used, and restored Maine governor's polls ("Bobby" vs. "Robert B." Charles).

**House district polls.** 84 district polls across 50 districts are blended with the district model, the same way Senate and governor races are blended. Most districts have one or two polls, often campaign-sponsored, so polls get 24–43% weight there.

**Data-quality checks** run on every refresh:
- outlier polls
- VoteHub vs. Wikipedia disagreement
- nominee mismatches
- unconfirmed nominees
- stale races
- missing fields

Open warnings are recorded in each snapshot (`data_warnings`).

**Daily update.** `python src/daily_update.py` refreshes everything, runs the checks, publishes the snapshot, and logs the run in `forecasts/2026/update_log.csv`.

**Effect today:** Senate Dem control 66% → 64%. House 87% → 84%, mainly from district polls.

## 2026.10.06c: generic-ballot lean correction; calibration report (roadmap Phase 3)

**Changed:**
- The generic-ballot polling average is converted to an expected national House vote by removing its historical Democratic overstatement. At four weeks out that's 3.0 pts, interpolated by days to go.
  - Generic-ballot polls overstated Democrats in 11 of 13 cycles from 1998 to 2022.
  - The correction improved the out-of-sample backtest at both horizons tested.
- Effect on today's forecast:
  - D+8.8 in the polls becomes an expected House vote of D+5.8.
  - House Dem control 97% → 87%.
  - Senate Dem control 70% → 66%.

**Calibration report (`forecasts/CALIBRATION.md`):** every polled Senate and governor race from 2000 to 2022 was re-forecast 0 and 28 days out, using only other cycles' data.
- The full model beats four baselines on Brier score and log loss at both horizons: polls only, a simple 30-day poll average, fundamentals only, and the last election result.
- At 28 days: Brier 0.068 vs 0.071 polls-only, 0.090 simple average, 0.123 fundamentals, 0.185 last election.
- Calibration: all 7 probability buckets at 28 days land inside their expected range. Where calibration misses, the model is slightly underconfident.

**Tested, not adopted:** correcting state polls' ~1-pt average Democratic lean. It made probabilities worse (Brier 0.068 → 0.071).

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
