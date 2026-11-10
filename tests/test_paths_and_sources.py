"""Tipping point, paths, election-night re-weighting, manual polls, ranked-choice parsing,
and the second approval source."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import simulations as S
from polls_2026 import _rcv_round, _split_field, merge_manual
from national_polls import combine_approval


def test_tipping_point_is_the_needth_most_democratic_race():
    # Two simulations, three races. Need 2 Dem wins.
    m = np.array([[5.0, 1.0, -3.0],     # order: race0, race1, race2 -> 2nd is race1
                  [-1.0, 4.0, 2.0]])    # order: race1, race2, race0 -> 2nd is race2
    tp = S.tipping_point(m, [True, True, True], 2)
    assert list(tp) == [0.0, 0.5, 0.5]


def test_tipping_point_skips_independent_races():
    m = np.array([[5.0, 9.0, 1.0]])
    tp = S.tipping_point(m, [True, False, True], 2)   # race1 is an independent: doesn't count
    assert list(tp) == [0.0, 0.0, 1.0]


def test_paths_report_share_of_winning_simulations():
    wins = np.array([[1, 1], [1, 0], [1, 1], [0, 0]], dtype=bool)
    control = wins.all(axis=1) | (wins[:, 0] & ~wins[:, 1])
    p = S.paths_to_control(wins, held=[True, False], control=control, labels=['A', 'B'], tossup=0.2)
    assert p['control'] == 0.75
    assert sum(c['share_of_wins'] for c in p['combos']) == 1.0


def _sims(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    nat = rng.normal(0, 1, (n, 1))
    sen = (nat * 3 + rng.normal(0, 2, (n, 3))).astype(np.float32)
    return {'senate': {'states': ['A', 'B', 'C'], 'dem_side_party': ['DEMOCRAT'] * 3, 'margin': sen},
            'governor': {'states': ['A'], 'dem_side_party': ['DEMOCRAT'], 'margin': sen[:, :1]},
            'house': {'keys': [('A', 1)], 'dem_win': (sen[:, :1] > 0)},
            'senate_holdovers': {'DEMOCRAT': 49, 'REPUBLICAN': 48}}


def test_election_night_moves_correlated_races():
    sims = _sims()
    before = S.election_night(sims, [])
    after = S.election_night(sims, [('senate', 0, True)])
    assert abs(after['race_probs']['senate'][0] - 1.0) < 1e-9            # the called race is fixed
    assert after['race_probs']['senate'][1] > before['race_probs']['senate'][1]   # shared error moves the rest
    assert after['senate']['dem_control'] > before['senate']['dem_control']


def test_safe_seat_calls_dont_reweight():
    sims = _sims()
    sims['senate']['margin'][:, 2] = 50.0                       # a race Dems win in every simulation
    w, ess = S.call_weights(sims, [('senate', 2, True)])
    assert np.allclose(w, 1 / len(w))


def test_rcv_round_parsing():
    assert _rcv_round('3[j]') == 3 and _rcv_round('1') == 1
    assert np.isnan(_rcv_round('BA'))


def test_split_field_table_is_not_a_head_to_head():
    t = pd.DataFrame({'Poll source': ['X'], 'Wilson (R)': ['20%'], 'Kreiss-Tomkins (D)': ['40%'],
                      'Bronson (R)': ['14%'], 'Taylor (R)': ['7%']})
    assert _split_field(t, 'Kreiss-Tomkins (D)', 'Wilson (R)')
    small = t.assign(**{'Bronson (R)': ['3%'], 'Taylor (R)': ['2%']})
    assert not _split_field(small, 'Kreiss-Tomkins (D)', 'Wilson (R)')


def _poll(**kw):
    base = dict(office='senate', state='OHIO', district=np.nan, pollster='Rasmussen Reports (R)',
                end_date=pd.Timestamp('2026-10-01'), dem_candidate='Sherrod Brown', rep_candidate='Jon Husted',
                dem_side_party='DEMOCRAT', dem_pct=48.0, rep_pct=45.0, margin=3.0, source='wikipedia')
    base.update(kw)
    return base


def test_manual_poll_dropped_once_wikipedia_has_it(tmp_path, monkeypatch):
    scraped = pd.DataFrame([_poll()])
    dup = pd.DataFrame([_poll(pollster='Rasmussen Reports', end_date=pd.Timestamp('2026-10-03'), source='manual')])
    new = pd.DataFrame([_poll(end_date=pd.Timestamp('2026-10-20'), source='manual')])
    import polls_2026
    monkeypatch.setattr(polls_2026, 'POLLS_DIR', str(tmp_path))      # no nominees file: skip the name check
    assert len(merge_manual(scraped, dup)) == 1
    assert len(merge_manual(scraped, new)) == 2


def test_ucsb_polls_fill_gaps_without_duplicating():
    wiki = pd.DataFrame({'pollster': ['Reuters/Ipsos'], 'end_date': [pd.Timestamp('2026-10-05')],
                         'approve': [32.0], 'disapprove': [65.0]})
    ucsb = pd.DataFrame({'pollster': ['Ipsos', 'Gallup'],
                         'end_date': [pd.Timestamp('2026-10-04'), pd.Timestamp('2025-11-25')],
                         'approve': [32.0, 36.0], 'disapprove': [65.0, 60.0]})
    out = combine_approval(wiki, ucsb)
    assert len(out) == 2 and set(out['source']) == {'wikipedia', 'ucsb'}
