"""
National "fundamentals" adjustment for the presidential model.

gdp_growth/unemployment/inflation/consumer_confidence/incumbent_on_ballot/
consecutive_terms/prior_midterm_house_margin are national, election-year-constant
values. Across 2000-2024 there are exactly 7 historical observations of each —
nowhere near enough for src/modeling.py's per-county XGBoost model to learn a
reliable relationship from (confirmed empirically: including them barely moved
predictions and let two spurious features, party_OTHER/party_REPUBLICAN, dominate
~70% of the model's decision weight). So instead of feeding these into the tree
model, they're applied here as a national swing on top of it — the same mechanism
app.py already uses for `approval` (a hand-picked `(approval-50)*0.15` adjustment),
just extended to the rest of the "national conditions" inputs.

The coefficients below are HAND-CALIBRATED, not fit by regression. With only 7
historical presidential elections, a regression against 7 variables would have
(much) more parameters than data points — any "fit" would be overfitting dressed
up as data-driven precision. `national_incumbent_margin_by_year()` below computes
the real historical incumbent-party national margin alongside each cycle's actual
fundamentals, for eyeballing direction/magnitude sanity — not for literal fitting.
Sanity-check table (computed 2026-08-21):

    year  inc_party   margin   gdp   unemp  infl   conf    iob   ct   pmhm
    2000  DEMOCRAT      0.51  6.52    3.9  3.45  105.8  False   2  -2.59
    2004  REPUBLICAN    2.47  6.39    5.5  3.19   91.7   True   1  -6.61
    2008  REPUBLICAN   -7.54  2.30    6.5  3.73   57.6  False   2   7.35
    2012  DEMOCRAT      4.02  4.29    7.8  2.16   82.6   True   1  -7.19
    2016  DEMOCRAT      2.33  2.67    4.9  1.69   87.2  False   2  -7.17
    2020  REPUBLICAN   -4.54 -0.06    6.9  1.23   81.8   True   1   7.45
    2024  DEMOCRAT     -1.50  5.12    4.1  2.58   70.5  False   1  -3.26

Directionally consistent with the coefficients below: 2008's crash-adjacent economy
and 2020's pandemic economy were the two worst incumbent performances; open-seat
years (iob=False: 2000, 2016, 2024) all underperformed a "normal" incumbent run.
"""
import os
import pandas as pd

from feature_engineering import INCUMBENT_PARTY
from data_cleaning import load_and_merge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/polling_data')

# Baseline = roughly the historical average for that indicator; coefficient = pt
# swing in the incumbent party's national margin per unit above/below baseline.
GDP_BASELINE, GDP_COEF               = 2.5, 0.8    # per pt GDP growth above baseline
UNEMPLOYMENT_BASELINE, UNEMP_COEF    = 5.5, -0.6   # per pt unemployment above baseline
INFLATION_BASELINE, INFLATION_COEF   = 2.5, -0.5   # per pt YoY CPI above baseline
CONFIDENCE_BASELINE, CONFIDENCE_COEF = 75.0, 0.08  # per pt UMCSENT above baseline
OPEN_SEAT_PENALTY   = -2.5   # incumbent president not actually on the ballot
TERM_FATIGUE_COEF   = -1.5   # per consecutive term beyond the first ("time for change")
MIDTERM_CARRYOVER   = 0.25   # fraction of the prior midterm's House margin that carries over
MAX_ADJUSTMENT       = 15.0  # clip the combined swing to a plausible range

def fundamentals_adjustment(gdp_growth, unemployment, inflation, consumer_confidence,
                             incumbent_on_ballot, consecutive_terms,
                             prior_midterm_house_margin, incumbent_party):
    """Pt swing in the incumbent party's vote share from national conditions."""
    adj = 0.0
    adj += (gdp_growth - GDP_BASELINE) * GDP_COEF
    adj += (unemployment - UNEMPLOYMENT_BASELINE) * UNEMP_COEF
    adj += (inflation - INFLATION_BASELINE) * INFLATION_COEF
    adj += (consumer_confidence - CONFIDENCE_BASELINE) * CONFIDENCE_COEF
    if not incumbent_on_ballot:
        adj += OPEN_SEAT_PENALTY
    adj += max(consecutive_terms - 1, 0) * TERM_FATIGUE_COEF
    # prior_midterm_house_margin is always Dem-minus-Republican; flip its sign when
    # the incumbent party is Republican so this term always means "swing toward
    # the incumbent," not "swing toward Democrats."
    signed_pmhm = prior_midterm_house_margin if incumbent_party == 'DEMOCRAT' else -prior_midterm_house_margin
    adj += signed_pmhm * MIDTERM_CARRYOVER
    return max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adj))


def national_incumbent_margin_by_year():
    """
    Actual historical national two-party incumbent-party margin per presidential
    cycle (2000-2024) — for sanity-checking the hand-picked coefficients above,
    not for fitting them.
    """
    votes = load_and_merge(
        os.path.join(DATA_DIR, 'complete_data.csv'),
        os.path.join(DATA_DIR, '2024president.csv'),
    )
    votes['party'] = votes['party_simplified'].str.upper()
    votes = votes[votes['party'].isin(['DEMOCRAT', 'REPUBLICAN'])]

    margins = {}
    for year, inc_party in INCUMBENT_PARTY.items():
        if year not in [2000, 2004, 2008, 2012, 2016, 2020, 2024]:
            continue
        chal_party = 'REPUBLICAN' if inc_party == 'DEMOCRAT' else 'DEMOCRAT'
        agg = votes[votes['year'] == year].groupby('party')['votes'].sum()
        inc_v, chal_v = agg.get(inc_party, 0), agg.get(chal_party, 0)
        if inc_v + chal_v > 0:
            margins[year] = round((inc_v - chal_v) / (inc_v + chal_v) * 100, 2)
    return margins
