import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import names as VH
from polls_2026 import resolve_nominees
import national_polls as NP


# ── Name matching ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('a,b', [
    ('Dan Sullvian', 'Dan S. Sullivan'),        # typo in a poll listing
    ('Dan Sullivan', 'Dan S. Sullivan'),        # middle initial
    ('Bobby Charles', 'Robert B. Charles'),     # nickname
    ('Troy Jackson', 'Troy Jackson'),
])
def test_same_person(a, b):
    assert VH._name_score(a, b) >= 0.8

def test_different_people():
    assert VH._name_score('Susan Collins', 'Troy Jackson') < 0.8
    assert VH._name_score('Graham Platner', 'Troy Jackson') < 0.8

def test_match_prefers_exact_when_surnames_collide():
    answers = [{'choice': 'Dan J. Sullivan'}, {'choice': 'Dan S. Sullivan'}, {'choice': 'Mary Peltola'}]
    assert VH._match(answers, 'Dan S. Sullivan') == 1

# ── Nominee resolution ───────────────────────────────────────────────────────
def test_resolve_lone_major_party_candidates():
    cands = [('Troy Jackson', 'DEMOCRAT'), ('Susan Collins', 'REPUBLICAN'), ('Some Green', 'GREEN')]
    assert resolve_nominees(cands, []) == ('Troy Jackson', 'Susan Collins', 'DEMOCRAT')

def test_resolve_independent_when_no_democrat():
    cands = [('Dan Osborn', 'INDEPENDENT'), ('Other Guy', 'INDEPENDENT'), ('Pete Ricketts', 'REPUBLICAN')]
    assert resolve_nominees(cands, ['Dan Osborn', 'Pete Ricketts'] * 3) == ('Dan Osborn', 'Pete Ricketts', 'INDEPENDENT')

def test_resolve_multiple_republicans_by_poll_presence():
    cands = [('Jonathan Kreiss-Tomkins', 'DEMOCRAT'), ('Dave Bronson', 'REPUBLICAN'), ('Bernadette Wilson', 'REPUBLICAN')]
    names = ['Jonathan Kreiss-Tomkins', 'Bernadette Wilson'] * 3 + ['Dave Bronson']
    assert resolve_nominees(cands, names)[1] == 'Bernadette Wilson'

def test_resolve_ambiguous_uses_fallback_only_if_on_the_list():
    cands = [('Jonathan Kreiss-Tomkins', 'DEMOCRAT'), ('Dave Bronson', 'REPUBLICAN'), ('Bernadette Wilson', 'REPUBLICAN')]
    ok = ('Jonathan Kreiss-Tomkins', 'Bernadette Wilson', 'DEMOCRAT')
    assert resolve_nominees(cands, [], fallback=ok) == ok
    # Stale hypothetical (Peltola is running for Senate, Begich isn't on this ballot)
    assert resolve_nominees(cands, [], fallback=('Mary Peltola', 'Nick Begich', 'DEMOCRAT')) is None


# ── Independence policy and national inputs ──────────────────────────────────
@pytest.mark.parametrize('name,tied', [
    ('Decision Desk HQ', True), ('RealClearPolitics', True), ('Real Clear Politics', True),
    ('Silver Bulletin', True), ('FiftyPlusOne', False), ('NBC News', False),
])
def test_market_tied_aggregators(name, tied):
    assert NP._is_market_tied(name) is tied

def test_approval_dates_take_year_from_section():
    assert NP._approval_date('Sep 29 – Oct 2', 2026, 10) == pd.Timestamp('2026-10-02')
    assert NP._approval_date('October 1–4', 2026, 10) == pd.Timestamp('2026-10-04')
    assert NP._approval_date('Dec 30 – Dec 31', 2026, 1) == pd.Timestamp('2025-12-31')

def test_approval_average_weights_recent_polls():
    polls = pd.DataFrame({'pollster': ['A', 'B'], 'end_date': pd.to_datetime(['2026-10-05', '2026-09-10']),
                          'sample_size': [1000, 1000], 'approve': [30.0, 40.0]})
    assert NP.approval_average(polls, '2026-10-05')['approve'] < 35


def test_bipartisan_poll_is_not_sponsored():
    from pollster_ratings import sponsor_party
    assert sponsor_party('Beacon Research (D)/ Shaw & Co. Research (R)') is None
    assert sponsor_party('Trafalgar Group (R)[3]') == 'REPUBLICAN'

def test_our_rating_words():
    from comparisons import our_rating
    assert our_rating(0.52) == ('Tossup', 0)
    assert our_rating(0.85) == ('Likely D', 15)
    assert our_rating(0.10) == ('Likely R', -15)
    assert our_rating(0.97, 'INDEPENDENT') == ('Safe I', 30)
