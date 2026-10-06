import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from polls_2026 import _end_date, _sample, _pct, _cand_name, _pick_sides, _party_columns, _sanitize
from pollster_ratings import match_pollster, sponsor_party, _tokens
import polls_average as PA


# ── Wikipedia poll-table parsing ─────────────────────────────────────────────
@pytest.mark.parametrize('text,expected', [
    ('October 1–4, 2026', '2026-10-04'),
    ('September 29 – October 1, 2026', '2026-10-01'),
    ('June 5, 2026[3]', '2026-06-05'),
    ('September 22−27, 2026', '2026-09-27'),                 # Unicode minus sign
    ('December 27, 2024 − January 4, 2025', '2025-01-04'),    # spans a year boundary
])
def test_end_date(text, expected):
    assert _end_date(text) == pd.Timestamp(expected)

def test_sample_and_pct():
    assert _sample('1,505 (RV)') == (1505.0, 'RV')
    assert _sample('417 (LV)') == (417.0, 'LV')
    assert _pct('51%') == 51
    assert pd.isna(_pct('—'))

def test_candidate_names_keep_full_name():
    # Alaska 2026 has two Republicans named Dan Sullivan.
    assert _cand_name('Dan S. Sullivan (R)') != _cand_name('Dan J. Sullivan (R)')
    assert _cand_name('Susan Collins (R)[x]') == 'susan collins'

def test_pick_sides():
    cols = _party_columns(['Poll source', 'Susan Collins (R)', 'Troy Jackson (D)', 'Other'])
    assert _pick_sides(cols) == ('Troy Jackson (D)', 'Susan Collins (R)', 'DEMOCRAT')
    # Independent stands in when there's no Democrat (Nebraska)
    cols = _party_columns(['Dan Osborn (I)', 'Pete Ricketts (R)'])
    assert _pick_sides(cols)[2] == 'INDEPENDENT'
    # Minnesota's DFL counts as Democratic
    cols = _party_columns(['Peggy Flanagan (DFL)', 'Michele Tafoya (R)'])
    assert _pick_sides(cols)[2] == 'DEMOCRAT'

def test_sanitize_malformed_colspan():
    assert _sanitize('<td colspan="2"" x>') == '<td colspan="2" x>'


# ── Pollster matching ────────────────────────────────────────────────────────
@pytest.fixture
def ratings():
    return pd.DataFrame([
        {'pollster': 'Marist College', 'aliases': 'Marist College', 'n_polls': 250, 'house_effect': 0.5, 'plus_minus': -0.3},
        {'pollster': 'YouGov', 'aliases': 'YouGov|YouGov/CBS News', 'n_polls': 800, 'house_effect': 0.2, 'plus_minus': -0.1},
        {'pollster': 'Emerson College', 'aliases': 'Emerson College', 'n_polls': 300, 'house_effect': -1.3, 'plus_minus': -0.7},
    ])

def test_match_pollster(ratings):
    assert match_pollster('Marist University[349]', ratings)['pollster'] == 'Marist College'
    assert match_pollster('CBS News/YouGov[352]', ratings)['pollster'] == 'YouGov'
    assert match_pollster('Emerson', ratings)['pollster'] == 'Emerson College'
    assert match_pollster('Data for Progress (D)', ratings) is None

def test_sponsor_party():
    assert sponsor_party('Tavern Research (D)[200]') == 'DEMOCRAT'
    assert sponsor_party('Trafalgar Group (R)[350]') == 'REPUBLICAN'
    assert sponsor_party('Marist University[349]') is None


# ── Averaging ────────────────────────────────────────────────────────────────
def _poll(pollster, days_ago, margin, n=800, qw=1.0, corr=0.0):
    return {'pollster': pollster, 'end_date': pd.Timestamp('2026-10-06') - pd.Timedelta(days=days_ago),
            'sample_size': n, 'margin': margin, 'adjusted_margin': margin - corr, 'quality_weight': qw}

def test_recent_polls_count_more():
    polls = pd.DataFrame([_poll('A', 1, 10), _poll('B', 60, -10)])
    assert PA.race_average(polls, '2026-10-06')['poll_avg'] > 5

def test_old_polls_dropped():
    polls = pd.DataFrame([_poll('A', PA.MAX_POLL_AGE_DAYS + 5, 10)])
    assert PA.race_average(polls, '2026-10-06') is None

def test_one_pollster_cannot_flood_average():
    flood = [_poll('A', 2, 10) for _ in range(9)]
    polls = pd.DataFrame(flood + [_poll('B', 2, -10)])
    # 9 polls from A vs 1 from B: A gets sqrt(9)=3x B's weight, not 9x
    assert PA.race_average(polls, '2026-10-06')['poll_avg'] == pytest.approx(5.0, abs=0.01)

def test_poll_error_shrinks_with_more_polls():
    m = {'floor': 10.0, 'thin': 60.0}
    assert PA.poll_error_sd(1, m) > PA.poll_error_sd(4, m) > PA.poll_error_sd(16, m) > 10 ** 0.5
