# Post-Election Plan: Becoming a Forecaster People Trust

*Written November 6, 2026, after the 2026 midterms. Updated November 10 with the first grading run, and November 16 with 495 of 506 races decided. Work through it top to bottom and check items off as you go.*

The goal: turn one strong cycle into a public, verifiable track record, the same way Nate Silver built his. He's a **forecaster** (he models other people's polls), not a **pollster** (who runs surveys). The Ledger Forecast is the same kind of thing, so this plan is about being a trusted forecaster.

---

## Where things stand (Nov 16)

From `forecasts/GRADES_2026.md`, final forecast (Nov 3), 495 races decided. Preliminary: 11 races are still out (Alaska Senate, AK-AL, CA-23, LA-5 runoff, LA-6, MN-1, NC-3/5/7/10, TN-4).

| | Ledger model | Experts | Polls only | Fundamentals only |
|---|---|---|---|---|
| Winners right | 93% | 94% | 94% | 83% |
| Brier (lower is better) | **0.053** | 0.057 | 0.055 | 0.122 |
| Races scored | 495 | 439 | 62 | 70 |

**What went right**
- Both chambers called for Democrats (68% Senate, 92% House). Democrats won 55 Senate seats and 259 House seats so far.
- Beat the expert consensus overall (0.053 vs. 0.057), in the House (0.051 vs. 0.055) and in the Senate (0.067 vs. 0.073).
- Improved over the campaign: Brier was 0.057 on Oct 6 and 0.063 on Oct 7, when the lean correction was added, then fell steadily to 0.053 on Nov 3.
- The 99%+ calls held: 327 races, 99% right.
- No betting-market data anywhere in the model.

**What went wrong**
- **Nearly every miss went one way.** 35 of 36 misses were seats the model gave to Republicans that Democrats won, including the Mississippi Senate race (86% R) and Kansas governor (60% R). The one exception was ME-2 (56% D, won by the Republican). Democrats beat our forecast margin in 79% of races, by 5.3 points on average, about the same in every office: House +5.3, Senate +5.3, governors +6.0. That's one national error, not noise.
- **Our lean correction made it worse.** In the 28 polled Senate races, raw polling averages missed by D+3.8. After we pulled them 1.8 pts toward Republicans, they missed by D+5.1.
- **Fundamentals hurt polled races.** On the 62 polled Senate and governor races, polls alone (Brier 0.055) beat the full model (0.072). *(Nov 11 check: this came from the lean correction, not the fundamentals weight. See section 4.)*
- **Three "impossible" misses.** FL-6, FL-19 and PA-11 had no polls, and the model gave Republicans 99.6% or more. Democrats won by 11 to 24 points. These three alone add 0.006 to the Brier; without them it's 0.047. The experts had them Safe R too, but their scale stops at 97%, so the miss cost them less.
- **Competitive races were weak.** Where the favorite had under 90%, we went 59 of 84 (70%). On those same races the experts got 67 of 83 (81%), counting 5 pure toss-ups as misses.
- **Overconfident in the middle.** Favorites at 60–70% won 52% of the time; favorites at 90–95% won 81%.
- **Governors: the experts beat us** (0.059 vs. 0.062; 94% vs. 89% of winners right).
- **Ranges too narrow.** 79% of Senate results (27 of 34) and 78% of governor results (28 of 36) fell inside our 90% range, against a target of 90%.
- **House seats at the edge.** 259 Democratic seats so far, at the top of our 90% range (215–259).

---

## 1. Finish the 2026 record

- [ ] Enter the remaining results in `forecasts/2026/results_2026.csv` (495 of 506 done as of Nov 16):
  - Senate: Alaska
  - House: AK-AL, CA-23, LA-6, MN-1, NC-3/5/7/10, TN-4
  - LA-5: wait for the runoff, then replace `RUNOFF` with the winner
  - Alaska uses ranked-choice voting, so its races likely won't be final until the RCV tabulation (usually about two weeks after Election Day)
- [ ] Check the 🏁 Results tab → **Results file checks** shows no problems
- [ ] Once states certify, re-check the close races against official totals. Election-night margins often move a point or two.
- [ ] Re-run `python src/grade_2026.py` after each batch of results. The "Preliminary" banner in `GRADES_2026.md` disappears once every race is in.
- [ ] Add a `forecasts/CHANGELOG.md` entry recording the final grades

## 2. Publish an honest post-mortem

People trust forecasters who explain their misses. Publish this before anyone else grades you. The numbers above won't move much with 20 mostly safe races left, so the draft can start now.

- [ ] Write "How the Ledger Forecast did" for The Ledger (Substack). Use the Results tab and `GRADES_2026.md` for the numbers.
- [ ] Lead with the wins:
  - Both chambers called for Democrats (68% Senate, 92% House)
  - 93% of winners right, Brier 0.053, better than the experts' 0.057
  - Got more accurate every week of the campaign
  - Built without any betting-market data
- [ ] Say these weaknesses yourself, before critics do:
  - **Every miss went the same way.** All 33 were Democratic wins we had going Republican. Democrats beat our margins by about 5 points everywhere.
  - **Our correction backfired.** Pulling polls toward Republicans, based on 2016–2024, pushed Senate averages from D+3.8 off to D+5.1 off.
  - **Easy races pad the 93%.** In competitive races (favorite under 90%) we went 56 of 78 (72%); the experts did better there.
  - **Too sure in unpolled seats.** FL-6, FL-19 and PA-11 at 99.6%+ Republican, all won by Democrats.
  - **Senate ranges too narrow.** 79% inside our 90% ranges.
  - **Silver was more confident on both chambers, and right.** See section 7.
- [ ] Post the teasers from Post Builder → **Results**: report card, forecast vs. result, misses, most accurate pollsters, us vs. the alternatives
- [ ] Link the full write-up from every post

## 3. Make the record verifiable

"Every forecast on the record" is the tagline, so make it provable.

- [ ] Put the daily snapshot files (`forecasts/2026/*.csv`, `*_summary.json`, `topline_log.csv`) in a **public GitHub repo**. Public history gives them timestamps, so no one can claim they were edited after Nov 3. **This is the biggest credibility step.**
- [ ] Decide whether the model code goes public too. Open code lets people check your work, which is how Silver earned trust.
- [ ] From here on, publish each daily snapshot to the repo the day it comes out
- [ ] Answer the open question in the methodology Doc about naming the GitHub repo

## 4. Fix what 2026 taught us (before the next election)

In rough order of impact:

- [x] **Revisit the R-lean correction.** *Done Nov 11 (model 2026.11.11): state-poll correction removed; generic-ballot correction kept until the 2026 House vote is final. See CHANGELOG.* It came from history (2016–2024 polls overstated Democrats) but went the wrong way in 2026, and it's the clearest single cause of the one-sided misses. Test no correction, a smaller one, or one that shrinks toward zero in midterms with an unpopular president.
- [x] **Size the shared national error.** *Tested Nov 11 (model 2026.11.12): on election eve it's the right size in the 2000–2022 backtest, so it wasn't widened. Further out it was too big and is now held flat at its election-eve size. 2026's D+4.5 cycle-wide miss was about 1.3 times the assumed size, which is large but within range. See CHANGELOG.* The misses all moving together says races were more correlated than we allowed. A bigger shared error also helps the Senate 90% coverage (79%) and the overconfident 60–70% and 90–95% buckets.
- [x] **Give House districts their own shared error.** *Done Nov 11 (model 2026.11.12): weighted by poll reliance, ±3.0 for unpolled districts instead of the polls' ±2.7. Small gain in the 2026 replay. See CHANGELOG.*
- [x] **Put a floor on unpolled races.** *Done Nov 11 (model 2026.11.12): capped at 99.5% through a 1% coin-flip chance in the simulation. The fix for these three races is in log loss, not Brier: an upset at 99.5% costs nearly as much Brier as one at 99.9%. See CHANGELOG.* No race with zero polls should go above about 98–99%. An R+14 seat isn't safe in a wave.
- [x] **Lean harder on polls where they exist.** *Checked Nov 11: no change. In the 2000–2022 backtest the current weight is best: log loss 0.2125 at 28 days, vs. 0.2132 with fundamentals' uncertainty 25% wider. The 2026 polled races agree (0.161 vs. 0.162). 2024 alone preferred more poll weight. Polls alone beat the blend in 2026 because the polls-only score used raw poll averages, while the blend had the state-poll lean correction. Without the correction, the blend scores 0.044 on those 59 races vs. 0.045 for polls alone.* Polls alone beat the blended model in the 59 polled races. Check whether the fundamentals weight in polled races is too high.
- [ ] **Look at the national environment.** We used D+5.9. Compare it with the actual House popular vote once it's final. *Provisional, Nov 11: the House vote is about D+11.2 (Wikipedia's AP-based table, with 58 districts still under 95% counted). The generic ballot (D+8.3 on election eve) understated Democrats by about 2.9, the largest understatement in the 15-cycle record (the previous largest was 1.1 in 2008). Adding 2026 lowers the correction from 2.4 to about 2.1 on election eve, and from 3.2 to about 2.8 at four weeks out. The provisional numbers won't support a midterm/unpopular-president exception: generic-ballot polls overstated Democrats by 3.1 on average in past Republican-president midterms (2002, 2006, 2018). **To do once certified:** enter the national House margin as `gb_actual` in the three 2026 rows of `src/data/polling_data/generic_ballot_recent.csv`, then rerun `python src/calibration.py`.*
- [x] **Governors.** *Checked Nov 11: nothing governor-specific to fix. All 3 misses were close races where polls were near even (TX R+1.6, NV D+0.3, GA D+0.7) and the R-lean correction plus fundamentals pulled us to the Republican. The experts called NV and GA for Democrats. Without the state-poll correction (model 2026.11.12), GA flips to a correct call, and the governor Brier goes from 0.054 to 0.043, ahead of the experts' 0.047. Governor fundamentals missed by D+8.9 on average, vs. D+4.5 for governor polls. That's the national environment again, which the item above covers.* Experts beat us here. Look at the 3 governor misses (TX, NV, GA) and whether governor fundamentals need their own tuning.
- [x] **Fix the Results tab's range check.** *Done Nov 10.* `dashboard_results.py` (line 460) tests results against `race_sd` only. The full 90% range also includes `shared_sd` (`sqrt(race_sd² + shared_sd²)`), which gives 79% for the Senate, not 67%. Fix this before quoting the number publicly.
- [ ] **Fix the known poll-data issues:**
  - Idaho campaign internals (the scraper ignores Wikipedia's Sponsor column)
  - The possible copied Change Research poll
- [ ] **Re-run the backtest** with 2026 added. Publish the new calibration in `forecasts/CALIBRATION.md`.
- [ ] Write a CHANGELOG entry for every change, saying what changed and why, **before** the next forecast goes out

## 5. Build the audience

- [ ] Publish on a regular schedule, even in quiet months (weekly or every two weeks)
- [ ] Share the race-level posts. Specific races travel further than toplines.
- [ ] Engage with other forecasters and political journalists. Being cited by reporters is how a blog becomes a source.
- [ ] Make the public dashboard (roadmap Phase 5) shareable

## 6. Keep forecasting between big elections

A record only counts if it keeps growing.

- [ ] **2027 governors:** Kentucky, Louisiana, Mississippi (November 2027). Plus New Jersey and Virginia legislatures, and any specials.
- [ ] **Special elections:** forecast House specials as they come up
- [ ] **2028 presidential:** the big test. The current 2028 model is 2024 county results plus a national swing. Rebuild it on the 2026 polls-first engine, with the section 4 fixes in place first.
- [ ] **Realistic timeline:** one strong cycle gets you noticed. Two or three with a public, verifiable record get you treated as an authority.

---

## 7. Head to head: the Ledger Forecast vs. Nate Silver's FLIPR

FLIPR is Silver's 2026 midterm model (Forecast with Leading Indicators, Polls and (Expert) Ratings), published at Silver Bulletin. It blends polls, fundamentals (fundraising, incumbency) and expert ratings across about 40,000 simulations.

> **Independence note:** Silver Bulletin stays excluded as a **data source** under our independence policy (Silver advises Polymarket). Comparing results against it after the election is benchmarking, not an input, so it's fine. Don't feed any FLIPR numbers back into the model.

### Chamber control (final forecasts)

| | Ledger (Nov 3) | FLIPR (Nov 3) | Result |
|---|---|---|---|
| Senate, Dem control | **67.5%** | **82.4%** | Democrats ✓ |
| House, Dem control | **91.9%** | **96.4%** | Democrats ✓ |
| Senate, Dem seats | 52 median (90% range 47–56) | not public | 54 so far (MS and AK still out) |
| House, Dem seats | 234 median (90% range 215–259) | not public | 257 so far (about 16 still out) |

**Both models called both chambers right. FLIPR was more confident and right, so it wins the chamber comparison.**

| Brier score on chamber control (lower is better) | Ledger | FLIPR |
|---|---|---|
| Senate | 0.106 | 0.031 |
| House | 0.007 | 0.001 |

Be upfront about this in the post-mortem. Our House seat median (234) was 23 seats short of the result so far, and 257 is already near the top of our 90% range. That's the same national error as in the race-level misses. Our national environment was too pessimistic for Democrats, the R-lean correction made it worse, and errors across races were more correlated than we allowed.

### Individual Senate races

The full FLIPR race-by-race forecast is paywalled, so only three race-level numbers are public, all from press coverage of FLIPR's **polling averages** (not its final forecast). The fair comparison is with **our raw polling average**:

| Race | FLIPR poll avg | Our raw poll avg | Our final forecast (win chance) | Result | Closer polling avg |
|---|---|---|---|---|---|
| Iowa | D+1.1 | D+1.8 | D+0.1 (52%) | **D+4.1** | Ledger (2.3 off vs. 3.0) |
| Nebraska | R+1.2 | R+1.7 | R+3.8 (24% Osborn) | **R+6.8** | Ledger (5.1 off vs. 5.6) |
| South Carolina | R+1.1 | R+0.9 | R+3.2 (27% Andrews) | **D+4.4** | Ledger (5.3 off vs. 5.5) |

Our polling averages came slightly closer in all three. Both had South Carolina's winner wrong. In Iowa and South Carolina, our lean correction pushed the *final forecast* further from the result than our raw average. That's the same finding as above.

**Bottom line:** FLIPR beat us on chamber odds. Our race-level polling averages held up as well or slightly better in the few races we can check. A full comparison needs FLIPR's race-by-race final numbers, which you'd get with a Silver Bulletin subscription.

### To-do for the comparison

- [ ] Get FLIPR's final race-level win probabilities (paid subscription, or Silver's own post-election grading post) and compute Brier scores on the same races
- [ ] Add FLIPR seat medians if he publishes them
- [ ] Update this table once Alaska, Mississippi and the remaining House races are in
- [ ] Decide whether to include this comparison in the public post-mortem. It's honest and reads well, since you lost the chamber call but held your own race by race.

### Sources

- [FLIPR methodology (Silver Bulletin)](https://www.natesilver.net/flipr-midterms-model-methodology)
- [Silver Bulletin Senate forecast, updated Nov 3, 2026 (paywalled)](https://www.natesilver.net/p/nate-silver-2026-midterm-election-senate-model)
- [Covers, Nov 3, 2026: FLIPR Senate 82.4%, House 96.4%, and the IA/NE/SC polling margins](https://www.covers.com/politics/nate-silver-vs-prediction-markets-polling-prices-midterms-elections-november-2026)
- [Daily Beast, Nov 2, 2026: FLIPR Senate about 80%](https://www.thedailybeast.com/nate-silver-makes-huge-senate-call-on-eve-of-election)
- [Newsweek: FLIPR Senate 75%, moved to Likely D](https://www.newsweek.com/nate-silver-revises-democrats-chances-winning-senate-12586949)
- [Newsweek: Democratic chances hit new highs](https://www.newsweek.com/democrat-chances-winning-house-senate-hit-new-high-nate-silver-forecast-12545290)

*The FLIPR numbers come from press reports on the dates shown, not Silver Bulletin's own files. Ledger numbers come from `forecasts/2026/2026-11-03_*`, `results_2026.csv` and `GRADES_2026.md` as of Nov 10.*
