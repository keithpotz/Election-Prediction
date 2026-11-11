# How the Ledger Forecast Works

*Model version 2026.11.12. Every change to the method is dated and logged publicly before the next forecast that uses it.*

The Ledger Forecast is The Ledger's model of the 2026 midterms. The name is the promise: every forecast goes on the record, dated and versioned, and gets graded against the results. Every Senate race, every governor's race and all 435 House districts on the November 3 ballot get a forecast. Each one is a probability, not a prediction of who wins: when we say a candidate has a 70% chance, we expect them to lose about three times in ten. This piece explains where those numbers come from, what we leave out on purpose, and how we'll know afterward whether we got it right.

## The short version

1. **Polls come first.** We average every public poll in each race, correcting for each pollster's track record and partisan lean.
2. **Fundamentals fill the gaps.** A model of each state's partisanship, incumbency and the national mood gives a baseline estimate. It matters most where polls are scarce.
3. **The two are blended by how accurate each has been.** Well-polled races lean almost entirely on polls. Unpolled races lean on fundamentals.
4. **Then we run the election 20,000 times.** Each run includes a nationwide polling miss that moves every race together, because that's how polls actually fail.
5. **We test all of it on past elections** and will grade it publicly against the official results.

## Where the data comes from, and what we leave out

**Polls.** Our race polls come from Wikipedia's race pages, which compile the pollsters' own releases. When a pollster has released a poll Wikipedia hasn't listed yet, we enter it by hand from the pollster's own release and link to it. When Wikipedia catches up, its copy replaces ours.

**Presidential approval.** We average the individual polls ourselves. They come from Wikipedia and the UC Santa Barbara American Presidency Project.

**No prediction-market data.** We don't use any data from an organization with a business tie to a prediction or betting market. That rules out several well-known aggregators:
- Decision Desk HQ, which runs live Polymarket and Kalshi odds in its coverage.
- RealClearPolitics, which added Polymarket odds to its 2026 maps.
- Silver Bulletin, since Nate Silver advises Polymarket.
- VoteHub, which has a data partnership with Kalshi.

Forecasts and markets increasingly feed off each other. We want the Ledger Forecast to be an independent read of the polls, not an echo of the betting line. We check every new source against this rule before using it.

**Expert ratings aren't inputs.** We show Cook, Sabato's Crystal Ball, Inside Elections and others alongside our numbers for comparison, but they never feed into the model. Comparing against them only means something if we haven't copied them.

**No hand edits.** We don't adjust individual races by feel. If we think the model is wrong somewhere, we change the method for every race and log the change.

## Step 1: Polling averages

Each race's polling average weights every poll by:
- **Recency.** A poll's weight halves every 14 days, and polls more than 120 days old drop out.
- **Sample size.** Bigger samples count more, with diminishing returns.
- **Pollster accuracy.** Pollsters with better track records count more (see the next section).
- **One pollster, one voice.** A firm that polls a race every week shares its weight across those polls, so it can't dominate the average by volume.

Then we make two corrections:
- **Pollster lean.** Some pollsters consistently land to the left or right of the others in the same races. We subtract half of each pollster's historical lean. In testing, the half and full corrections performed about the same, and half is the safer choice.
- **Campaign and party polls.** Polls paid for by a campaign or party count half. They're also corrected for the typical lean of sponsored polls, which favor their sponsor.

A few practical rules:
- **Likely voters first.** When a pollster reports several versions of one poll, we keep the likely-voter version.
- **Ranked-choice states.** In Alaska and Maine we use the final ranked-choice round, which is the actual head-to-head.
- **Head-to-heads only.** A poll that doesn't measure the two nominees against each other (with other candidates still splitting the vote) isn't used. Neither is a poll of a hypothetical matchup.

## Pollster ratings

We rate pollsters on about 18,000 polls with known results, taken from the final weeks of elections between 1998 and 2024. The source is FiveThirtyEight's public archive, plus 2024 polls we collected ourselves.

**How the ratings work:**
- **Same-race comparison.** Each poll is compared with **other pollsters' polls of the same race**. That way nobody is punished for a year when every pollster missed (2016, 2020), or rewarded for only polling easy races.
- **Two numbers per pollster:** how much bigger or smaller its errors were than its peers', and which way it leaned.
- **Shrinkage.** Ratings for pollsters with small track records are pulled toward average, so three lucky polls can't put a firm at the top of the list.
- **Transparency credit.** Following FiveThirtyEight's published method, pollsters in the AAPOR Transparency Initiative or the Roper Center archive get a small credit, and pollsters that herd toward the existing average get a penalty.

## Step 2: Fundamentals

Fundamentals answer the question: what would we expect in this race if there were no polls at all? The model uses four things:
- **The state's partisan lean.**
- **Incumbency.**
- **How strong this incumbent has been before:** how much they beat or trailed their state's lean last time.
- **The national environment** (next section).

It's trained on every Senate and governor race since 2000. Earlier races came from a much less polarized era and predict today's races badly.

On its own, fundamentals miss by about 13–14 points on average, which is why it's a backstop and not the main engine. It still matters: in races with few or no polls it's most of what we have.

**Independent candidates.** The model was trained on Democrat-versus-Republican races. Where the main challenger is an independent (Dan Osborn in Nebraska, Todd Achilles in Idaho), we double its uncertainty so the race's own polls carry more weight.

## The national environment

The national mood comes from the generic congressional ballot ("Which party would you vote for in your House race?").

- **Source.** We average the published generic-ballot averages that have no market ties, currently FiftyPlusOne and NBC News.
- **Lean correction.** Generic-ballot polls have historically overstated Democrats: in 12 of the last 14 cycles at four weeks out, by an average of about 3 points. We remove that historical lean before using the number. Testing on past elections showed the correction improves the forecast.
- **Cross-check.** We compare the generic ballot with what our own Senate and House polls imply about the national mood. If the two disagree by more than 3 points, we flag it.

## Step 3: Blending polls and fundamentals

Each race now has two estimates, each with a known track record. We combine them the standard way: each gets weight in proportion to its accuracy (inverse-variance weighting). No schedule is set by hand. The balance shifts on its own:
- **Lots of recent polls:** polls get most of the weight.
- **One poll from July:** fundamentals still matter.
- **No polls:** fundamentals carry the whole race.

**No state poll lean correction.** State and district polls have also leaned Democratic on average, but less consistently than the generic ballot, and in 2026 they leaned the other way. During the 2026 campaign we shifted every race's polling average 1.8 points toward Republicans to remove half of that historical lean. That shift never improved our probabilities in testing on past elections, and in 2026 it made the misses worse, so it was removed after the election (model 2026.11.11). The risk of a one-sided polling miss is now carried entirely by the shared national error in the simulations (Step 4), which can move every race in either direction.

## The House

All 435 districts are forecast on this year's maps, including the nine states that redrew for 2026. Each district's estimate combines four things:
- its partisan lean (Cook PVI);
- whether an incumbent is running;
- how much that incumbent has outrun or trailed the district before;
- the national environment.

**How we tested it.** We fit the district model on 2024 and applied it to 2022 districts whose lines hadn't changed. It called 94.6% of winners correctly, with a typical miss of about 5.7 points.

**District polls.** Where a district has its own polls, they're blended in exactly like Senate polls. Most districts have only one or two, often paid for by a campaign, so they get modest weight.

## Step 4: 20,000 simulated elections

Polling errors aren't independent coin flips. When polls miss, they usually miss in the same direction almost everywhere, as in 2016 and 2020. So each of our 20,000 simulated elections draws:

1. **One national error** that moves every race at once. For a well-polled race it's about ±2.7 points, and the same size whether the election is tomorrow or two months away. Polls alone miss by more further out, but once they're blended with fundamentals, the forecast's nationwide miss has stayed about the same size at every point we tested, from election eve to six weeks out. Senate and governor races that lean on fundamentals get a larger national error, about ±5.5 points, because the fundamentals' nationwide misses have been larger. Unpolled House districts get about ±3.0 points, which is how far our estimate of the national House vote can be off.
2. **A regional error** shared by nearby states in six regions: the Northeast, the Rust Belt, the Plains, the South, the Mountain West and the Pacific.
3. **Each race's own error**, sized by how accurate that race's estimate has historically been.

The regional piece changes how races move together without making any single race more or less certain.

Chamber control is simply how often each party wins the majority across the 20,000 runs. Democrats need 51 Senate seats; Republicans need 50, because Vice President Vance breaks ties. A House majority is 218.

The simulations also tell us:
- **The seat range.** The 90% range for each chamber's seat count.
- **The tipping point.** Which race delivers the deciding seat, found by lining up every race from most to least Democratic in each simulation.
- **The paths to a majority.** The most common combinations of wins that get each party there.

**Why the chamber odds are less certain than the race odds.** If each close race were a separate coin flip, the luck would even out across races. A shared polling miss doesn't even out: it can flip several close races in the same direction at once.

## How good is it?

We re-ran the forecast for every polled Senate and governor race from 2000 to 2024, as of four weeks before each election. That's 553 races. For each election year, the model was built only from the **other** years' data, so it never saw the answer.

**Four weeks out, the full model:**
- called 90% of winners;
- missed the final margin by 6.7 points on average (root-mean-square);
- scored 0.067 on the Brier score, which measures probability quality; lower is better and a coin flip scores 0.25.

**Every simpler method we compared it with did worse.** Brier scores at the same horizon:
- polls only: 0.070
- a simple 30-day poll average: 0.089
- fundamentals only: 0.123
- repeating the last election's result: 0.182

**Calibration.** Do the probabilities mean what they say? We grouped forecasts by the chance they gave the favorite. In all seven groups, favorites won about as often as forecast, within the range random chance allows. Where the model is off, it's slightly *under*confident: favorites win a little more often than we say. That's the safer direction to be wrong in.

**A caveat about scale.** Races within the same election year share their polling miss, so this is really about a dozen independent tests, not 553. That's especially true for chamber control, which no forecaster can calibrate from just a dozen elections.

## Known limits

- **Wider than the experts in unpolled races.** A race with no polls can show something like 70–75% where the experts say "Safe." That's deliberate: our backtests show fundamentals alone really are that uncertain. We'd rather be honest about it than borrow confidence from the experts.
- **Independents and unusual races.** Three-way races and independent challengers fit the model less well. We widen the uncertainty instead of pretending to know more.
- **District shapes.** The House map is shown by state, because accurate shapes for the redrawn 2026 districts weren't available. The forecasts themselves are district by district.
- **We can only use public polls.** Campaigns and parties see private polling we don't.

## Accountability

- **Every day's forecast is saved,** dated and stamped with its model version, so nobody (including us) can quietly rewrite what we said.
- **Every method change is logged** in a public changelog, with what it did to the forecast, before we publish a forecast that uses it.
- **We'll grade ourselves after November 3.** We set the plan in advance: every published forecast will be scored against official state results. We'll report the Brier score, log loss, winners called, calibration and every miss. We'll score the expert consensus, a polls-only version and a fundamentals-only version the same way, so you can see whether the full model earned its complexity.

We'd rather publish a forecast that's honest about what it doesn't know than one that sounds certain. When we're wrong, you'll hear it from us first.
