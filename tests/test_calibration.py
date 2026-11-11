import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import calibration as C


def test_score_rewards_better_probabilities():
    df = pd.DataFrame({'actual': [5.0, -3.0, 8.0, -1.0]})
    for m, _ in C.METHODS:
        df[f'{m}_margin'] = [1.0, -1.0, 1.0, -1.0]
        df[f'{m}_p'] = 0.5
    df['model_p'] = [0.9, 0.1, 0.9, 0.2]          # confident and right
    s = C.score(df).set_index('method')
    model = s.loc['Full model (polls + fundamentals)']
    coin = s.loc['Fundamentals only']
    assert model['brier'] < coin['brier'] and model['log_loss'] < coin['log_loss']

def test_score_uses_only_races_every_method_covers():
    df = pd.DataFrame({'actual': [5.0, -3.0]})
    for m, _ in C.METHODS:
        df[f'{m}_margin'] = [1.0, -1.0]
        df[f'{m}_p'] = [0.7, 0.3]
    df.loc[1, 'last_p'] = np.nan
    assert C.score(df)['races'].iloc[0] == 1

def test_calibration_table_counts_favorite_wins():
    # 10 races at 75% for the Dem; Dems win 7 -> favorite won 70%
    df = pd.DataFrame({'model_p': [0.75] * 10, 'actual': [1] * 7 + [-1] * 3})
    t = C.calibration_table(df)
    row = t[t['bucket'] == '70-80%'].iloc[0]
    assert row['races'] == 10 and row['actual'] == pytest.approx(0.7)
    assert row['expected_low'] < 0.75 < row['expected_high']

def test_calibration_table_handles_republican_favorites():
    df = pd.DataFrame({'model_p': [0.1] * 4, 'actual': [-5, -5, -5, 5]})   # R favored at 90%
    row = C.calibration_table(df).iloc[0]
    assert row['bucket'] == '90-95%' and row['actual'] == pytest.approx(0.75)

def test_generic_ballot_bias_interpolates(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'OUT_DIR', str(tmp_path))
    rows = [{'days_before': 0, 'mean_dem_overstatement': 2.0}, {'days_before': 28, 'mean_dem_overstatement': 3.0}]
    (tmp_path / 'calibration_report.json').write_text(json.dumps({'generic_ballot_bias': rows}))
    assert C.generic_ballot_bias(14) == pytest.approx(2.5)
    assert C.generic_ballot_bias(60) == pytest.approx(3.0)     # held flat beyond the last horizon


def test_state_poll_bias_reads_report(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'OUT_DIR', str(tmp_path))
    assert C.state_poll_bias(28) == 0.0                         # no report yet
    rows = [{'days_before': 0, 'applied_correction': 1.0}, {'days_before': 28, 'applied_correction': 2.0}]
    (tmp_path / 'calibration_report.json').write_text(json.dumps({'state_poll_bias': rows}))
    assert C.state_poll_bias(14) == pytest.approx(1.5)

def test_state_poll_bias_estimates_weight_recent_cycles():
    frames = {28: pd.DataFrame({'cycle': [2018, 2018, 2024, 2024], 'error': [0.0, 0.0, 4.0, 4.0]})}
    row = C.state_poll_bias_estimates(frames, as_of_cycle=2026)[0]
    assert 2.0 < row['weighted_dem_overstatement'] < 4.0        # recent 2024 miss counts more than 2018
    assert row['applied_correction'] == pytest.approx(C.STATE_POLL_BIAS_SCALE * row['weighted_dem_overstatement'])

def test_state_poll_correction_is_off():
    # Removed in 2026.11.11; see the comment on STATE_POLL_BIAS_SCALE.
    assert C.STATE_POLL_BIAS_SCALE == 0.0
