import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from blend import blend, shared_sd, shared_poll_miss_sd
import house_districts as H


# ── Blend ────────────────────────────────────────────────────────────────────
def test_blend_without_polls_is_fundamentals():
    assert blend(-10.0, 13.0) == (-10.0, 13.0, 0.0)
    assert blend(-10.0, 13.0, float('nan'), 5.0)[2] == 0.0

def test_blend_weights_by_precision():
    mean, sd, w = blend(0.0, 10.0, 10.0, 10.0)        # equal precision -> halfway
    assert mean == pytest.approx(5.0) and w == pytest.approx(0.5)
    assert sd == pytest.approx(10 / np.sqrt(2))
    _, _, w_tight = blend(0.0, 13.0, 5.0, 4.0)
    _, _, w_loose = blend(0.0, 13.0, 5.0, 9.0)
    assert w_tight > w_loose                            # better polls get more weight

def test_blended_sd_never_exceeds_either_input():
    _, sd, _ = blend(3.0, 12.0, -2.0, 7.0)
    assert sd < 7.0

def test_shared_error_grows_with_days_and_tracks_reliance():
    assert shared_poll_miss_sd(28) > shared_poll_miss_sd(0)
    assert shared_sd(1.0, 5.0, 0) == pytest.approx(shared_poll_miss_sd(0))
    assert shared_sd(0.0, 5.0, 0) == pytest.approx(5.0)


# ── House district parsing ───────────────────────────────────────────────────
def test_parse_location():
    assert H._parse_location('Alabama 2') == ('ALABAMA', 2)
    assert H._parse_location('Alaska at-large') == ('ALASKA', 0)
    assert H._parse_location('New York 10[3]') == ('NEW YORK', 10)

def test_parse_pvi():
    assert H._parse_pvi('R+17') == -17
    assert H._parse_pvi('D+4') == 4
    assert H._parse_pvi('EVEN') == 0

def test_parse_candidates():
    c = H._candidates('▌Jerry Carl (Republican) ▌Clyde Jones Jr. (Democratic)')
    assert ('Jerry Carl', 'REPUBLICAN') in c and ('Clyde Jones Jr.', 'DEMOCRAT') in c
    assert ('Angie Craig', 'DEMOCRAT') in H._candidates('▌Angie Craig (DFL) ▌Tyler Kistner (Republican)')


# ── House simulation logic (synthetic districts, no network) ─────────────────
class _FlatModel:
    def predict(self, X):
        return 2.0 * X['pvi'].values + 4.0 * X['incumbency'].values

def _districts(n_d, n_r):
    rows = [{'pvi': 10.0, 'incumbent_party': 'DEMOCRAT'}] * n_d + [{'pvi': -10.0, 'incumbent_party': 'REPUBLICAN'}] * n_r
    df = pd.DataFrame(rows)
    df['incumbent_running'] = True
    df['already_decided'] = False
    df['has_dem'] = True
    df['has_rep'] = True
    return df

def test_house_forecast_uncontested_and_swing(monkeypatch):
    monkeypatch.setattr(H, 'national_house_margin_by_year', lambda: pd.Series({2024: 0.0}))
    d = _districts(3, 2)
    d.loc[4, 'has_dem'] = False                         # a one-party Republican ballot
    out = H.forecast_house(0.0, d, (_FlatModel(), {'district_sd': 1.0}), n_sims=2000)
    assert out['districts'].loc[4, 'dem_win_prob'] == 0.0
    assert out['dem_seats_median'] == 3
    wave = H.forecast_house(40.0, d, (_FlatModel(), {'district_sd': 1.0}), n_sims=2000)
    assert wave['dem_seats_median'] == 4                # the uncontested seat can't flip
