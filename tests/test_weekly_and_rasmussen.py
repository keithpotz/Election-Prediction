"""What-moved attribution, and reading Rasmussen's write-ups."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import weekly as W
import rasmussen_check as R


def _snap(rows, version='v1'):
    cols = ['state', 'poll_weight', 'poll_avg', 'poll_avg_corrected', 'fund_pred', 'forecast_margin', 'n_polls',
            'latest_poll', 'dem_side_win_prob', 'dem_side_party', 'dem_candidate', 'rep_candidate', 'incumbent_party']
    return {'senate': pd.DataFrame(rows, columns=cols), 'summary': {'model_version': version}}


def _row(state, w, raw, corrected, fund, n, latest, p):
    return [state, w, raw, corrected, fund, w * (corrected or 0.0) + (1 - w) * fund, n, latest, p,
            'DEMOCRAT', 'Ann Dee', 'Bob Arr', 'REPUBLICAN']


def test_change_splits_into_polls_environment_and_method():
    old = _snap([_row('A', 0.8, 2.0, 1.0, -4.0, 5, '2026-10-01', 0.55)])
    # New poll moved the raw average up 2; the environment moved fundamentals up 1;
    # the lean correction grew from 1.0 to 1.5 (a method change)
    new = _snap([_row('A', 0.8, 4.0, 2.5, -3.0, 6, '2026-10-08', 0.62)])
    m = W.statewide_moves(old, new, 'senate').iloc[0]
    total = m['margin_after'] - m['margin_before']
    assert abs(m['from_polls'] - 0.8 * 2.0) < 1e-9
    assert abs(m['from_environment'] - 0.2 * 1.0) < 1e-9
    assert abs(m['from_method'] - 0.8 * -0.5) < 1e-9
    assert abs(m['from_polls'] + m['from_environment'] + m['from_method'] - total) < 1e-9


def test_first_poll_in_an_unpolled_race_counts_as_polls():
    old = _snap([_row('A', 0.0, None, None, -4.0, 0, None, 0.3)])
    new = _snap([_row('A', 0.5, 2.0, 1.0, -4.0, 1, '2026-10-08', 0.45)])
    m = W.statewide_moves(old, new, 'senate').iloc[0]
    assert abs(m['from_polls'] - 2.5) < 1e-9 and abs(m['from_method']) < 1e-9


def test_reason_names_new_polls_and_model_updates():
    r = {'from_polls': 2.0, 'from_environment': 0.1, 'from_method': 0.1, 'margin_before': 0.0, 'margin_after': 2.2,
         'n_polls_before': 3, 'n_polls_after': 4, 'dem_candidate': 'Ann Dee', 'rep_candidate': 'Bob Arr',
         'new_polls': [{'pollster': 'Pollco', 'end_date': pd.Timestamp('2026-10-07'), 'margin': 3.0}]}
    assert W.reason(r, method_changed=False) == "1 new poll: Pollco, Oct 7: Dee +3"
    r2 = dict(r, from_polls=0.2, from_method=2.0, new_polls=[])
    assert W.reason(r2, method_changed=True) == "Mostly the model update"


def test_method_changes_lists_versions_between_snapshots(monkeypatch):
    monkeypatch.setattr(W, 'changelog_versions', lambda: [('a', 'one'), ('b', 'two'), ('c', 'three')])
    assert W.method_changes('a', 'c') == [('b', 'two'), ('c', 'three')]
    assert W.method_changes('c', 'c') == []


PAGE = ("Ohio Senate: Democrat Brown Has 3-Point Lead Friday, September 25, 2026 Former Democratic Sen. Sherrod "
        "Brown narrowly leads. The latest Rasmussen Reports national telephone and online survey finds that, if the "
        "election were held today, 46% of Likely Ohio voters would vote for Brown, while 43% would vote for Husted. "
        "Three percent (3%) support Libertarian candidate William B. Redpath. The survey of 1,115 Ohio Likely Voters "
        "was conducted on September 30-October 1, 2026 by Rasmussen Reports.")


def test_reads_numbers_dates_and_sample(monkeypatch):
    monkeypatch.setattr(R, 'page_text', lambda url: PAGE)
    nominees = {('senate', 'OHIO'): {'dem_candidate': 'Sherrod Brown', 'rep_candidate': 'Jon Husted',
                                     'dem_side_party': 'DEMOCRAT'}}
    row, problems = R.read_poll('senate', 'OHIO', 'u', nominees)
    assert problems == []
    assert (row['dem_pct'], row['rep_pct'], row['sample_size'], row['population']) == (46, 43, 1115, 'LV')
    assert row['end_date'] == pd.Timestamp('2026-10-01') and row['start_date'] == pd.Timestamp('2026-09-30')
    assert row['published'] == pd.Timestamp('2026-09-25')


def test_flags_a_page_about_other_candidates(monkeypatch):
    monkeypatch.setattr(R, 'page_text', lambda url: PAGE)
    nominees = {('senate', 'OHIO'): {'dem_candidate': 'Sherrod Brown', 'rep_candidate': 'Someone Else',
                                     'dem_side_party': 'DEMOCRAT'}}
    row, problems = R.read_poll('senate', 'OHIO', 'u', nominees)
    assert row['rep_pct'] is None and problems
