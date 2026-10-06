import os
import sys
import math
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from race_status_2026 import _parse_pvi, _parse_last_election_party, _rating_to_number
import midterms as M


# ── Scraper parsers ──────────────────────────────────────────────────────────
def test_parse_pvi():
    assert _parse_pvi('R+15') == -15
    assert _parse_pvi('D+6') == 6
    assert _parse_pvi('EVEN') == 0
    assert math.isnan(_parse_pvi('garbage'))

def test_parse_last_election_party():
    assert _parse_last_election_party('60.10% R') == 'REPUBLICAN'
    assert _parse_last_election_party('62.58% R (2024 sp.)[r]') == 'REPUBLICAN'
    assert _parse_last_election_party('50.62% D') == 'DEMOCRAT'
    assert _parse_last_election_party('Appointed (2025)[q]') is None

def test_rating_to_number():
    assert _rating_to_number('Lean D (flip)') == 7
    assert _rating_to_number('Solid R') == -30
    assert _rating_to_number('Tossup') == 0


# ── Senate data handling (uses the real source files) ────────────────────────
@pytest.fixture(scope='module')
def races():
    return M.senate_races_table()

def _winner(races, year, state):
    return races[(races['year'] == year) & (races['state'] == state)]['winner'].iloc[0]

def test_runoff_decides_winner(races):
    # Perdue led the Nov 2020 round; Ossoff won the Jan 2021 runoff.
    assert _winner(races, 2020, 'GEORGIA') == 'DEMOCRAT'

def test_party_detailed_fallback(races):
    # Duckworth is labeled OTHER in party_simplified but DEMOCRATIC in party_detailed.
    assert _winner(races, 2022, 'ILLINOIS') == 'DEMOCRAT'

def test_party_corrections(races):
    assert _winner(races, 2020, 'WYOMING') == 'REPUBLICAN'

def test_fusion_votes_fold_into_major_party():
    df = M.load_senate_races()
    murphy = df[(df['year'] == 2024) & (df['state'] == 'CONNECTICUT') & (df['candidate'] == 'CHRISTOPHER S. MURPHY')]
    assert len(murphy) == 2 and (murphy['party'] == 'DEMOCRAT').all()

def test_seat_counts_match_current_senate(races):
    status = pd.DataFrame({'is_special': [True, True]}, index=['FLORIDA', 'OHIO'])
    up, holdovers = M.senate_seats_2026(races, status)
    assert len(up) == 35
    assert sum(holdovers.values()) == 65

def test_pvi_style_lean_uses_prior_elections_only():
    lean = {2020: {'X': 10.0}, 2016: {'X': 2.0}, 2024: {'X': -50.0}}
    # A 2024 race must not see the 2024 presidential result.
    assert M.pvi_style_lean(2024, 'X', lean) == pytest.approx(0.75 * 10 + 0.25 * 2)
    assert M.pvi_style_lean(2026, 'X', lean) == pytest.approx(0.75 * -50 + 0.25 * 10)

def test_incumbent_strength():
    assert M.incumbent_strength(20.0, 5.0, 0) == 0.0          # open seat
    assert M.incumbent_strength(20.0, 5.0, 1) == 15.0
    assert M.incumbent_strength(100.0, 5.0, 1) == 55.0        # uncontested clipped to 60


# ── Forecast behavior ────────────────────────────────────────────────────────
def test_house_forecast_moves_with_generic_ballot():
    model = M.build_house_swing_model()
    lo, hi = M.predict_house_2026(-5, model_and_std=model), M.predict_house_2026(5, model_and_std=model)
    assert lo['dem_seats'] < hi['dem_seats']
    assert lo['dem_control_prob'] < hi['dem_control_prob']
