"""
Shareable race cards (1200x675 PNG, the size social platforms preview best).

Each card shows the race, the favored candidate and win chance, the projected margin
with its likely range, what the forecast rests on, and the as-of date and model version.

Kept safe to share:
  - a probability, never a call ("favored", plus "a forecast, not a call")
  - dated and versioned, so an old card can't pass for a current one
  - text only: no logos, photos, or anything that implies an affiliation
  - sourcing stated, including that no prediction-market data is used

    python src/share_cards.py     # cards for every Senate and governor race + competitive House seats
"""
import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from blend import shared_poll_miss_sd
from states import STATE_ABBREV

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(BASE_DIR, '..', 'forecasts', '2026', 'cards')
from brand import MODEL_NAME as BRAND

BG, INK, INK_2, INK_3, RULE = '#ffffff', '#111111', '#55534f', '#8a8984', '#e4e3df'
PARTY_COLOR = {'DEMOCRAT': '#2a78d6', 'REPUBLICAN': '#e34948', 'INDEPENDENT': '#7b6fd6'}
PARTY_LETTER = {'DEMOCRAT': 'D', 'REPUBLICAN': 'R', 'INDEPENDENT': 'I'}
RANGE_Z = 1.2816    # 80% interval

def _rating(p_fav):
    for cut, word in ((0.55, 'Toss-up'), (0.65, 'Tilt'), (0.775, 'Lean'), (0.91, 'Likely'), (1.01, 'Safe')):
        if p_fav < cut:
            return word

def _fmt_margin(m, d_letter='D'):
    if abs(m) < 0.05:
        return 'Even'
    return f"{d_letter}+{m:.1f}" if m > 0 else f"R+{-m:.1f}"

def card_facts(race, office, days_to_election):
    """Normalizes a Senate/governor row or a House district row into what a card shows."""
    if office == 'house':
        p_d = float(race['dem_win_prob'])
        margin = float(race['predicted_margin'])
        d_party = 'DEMOCRAT'
        d_name, r_name = race.get('dem_candidate'), race.get('rep_candidate')
        title = f"{STATE_ABBREV[race['state']]}-{'AL' if int(race['district']) == 0 else int(race['district'])}"
        sd = float(np.hypot(race.get('race_sd', 5.7), shared_poll_miss_sd(days_to_election)))
        n_polls = int(race.get('n_polls', 0) or 0)
        basis = (f"{n_polls} district poll{'s' if n_polls != 1 else ''} + district partisan lean and incumbency"
                 if n_polls else "District partisan lean, incumbency and the national environment (no district polls)")
        label = f"U.S. House · {race['state'].title()}'s {'at-large district' if int(race['district']) == 0 else _ordinal(int(race['district'])) + ' district'}"
        uncontested = bool(race.get('uncontested', False))
    else:
        p_d = float(race['dem_side_win_prob'])
        margin = float(race['forecast_margin'])
        d_party = race.get('dem_side_party') or 'DEMOCRAT'
        d_name, r_name = race.get('dem_candidate'), race.get('rep_candidate')
        title = race['state'].title()
        sd = float(np.hypot(race['race_sd'], race['shared_sd']))
        n_polls = int(race.get('n_polls', 0) or 0)
        basis = (f"{n_polls} poll{'s' if n_polls != 1 else ''} (latest {pd.Timestamp(race['latest_poll']):%b} {pd.Timestamp(race['latest_poll']).day}) "
                 f"blended with fundamentals" if n_polls and pd.notna(race.get('latest_poll')) else
                 "State partisan lean, incumbency and the national environment (no polls yet)")
        special = ' (special)' if bool(race.get('special', False)) else ''
        label = f"{'U.S. Senate' if office == 'senate' else 'Governor'}{special} · {title}"
        uncontested = False
    d_name = d_name if isinstance(d_name, str) and d_name else ('Democratic candidate' if d_party == 'DEMOCRAT' else 'Independent candidate')
    r_name = r_name if isinstance(r_name, str) and r_name else 'Republican candidate'
    return {'p_d': p_d, 'margin': margin, 'sd': sd, 'd_party': d_party, 'd_name': d_name, 'r_name': r_name,
            'title': title, 'label': label, 'basis': basis, 'uncontested': uncontested}

def _ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

def make_card(race, office, as_of, model_version, days_to_election):
    """Returns the card as PNG bytes."""
    f = card_facts(race, office, days_to_election)
    p_d, p_r = f['p_d'], 1 - f['p_d']
    d_col, r_col = PARTY_COLOR[f['d_party']], PARTY_COLOR['REPUBLICAN']
    d_letter = PARTY_LETTER[f['d_party']]
    fav_is_d = p_d >= 0.5
    p_fav = max(p_d, p_r)
    word = _rating(p_fav)
    fav_name, fav_col = (f['d_name'], d_col) if fav_is_d else (f['r_name'], r_col)

    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1200); ax.set_ylim(0, 675); ax.axis('off')
    font = {'family': 'DejaVu Sans'}

    # Header
    ax.text(60, 615, f['label'].upper(), fontsize=17, color=INK_2, fontweight='bold', **font)
    ax.text(1140, 615, f"Forecast as of {as_of:%b} {as_of.day}, {as_of.year}", fontsize=15, color=INK_3, ha='right', **font)
    ax.plot([60, 1140], [592, 592], color=RULE, lw=2)

    # Headline
    if f['uncontested']:
        headline = f"{fav_name} is unopposed by a major party"
    elif word == 'Toss-up':
        headline = "Toss-up"
    else:
        headline = f"{fav_name} favored"
    ax.text(60, 515, headline, fontsize=40, color=INK, fontweight='bold', **font)
    sub = ("" if f['uncontested'] else
           (f"{word} {'Democratic' if fav_is_d and d_letter == 'D' else 'independent' if fav_is_d else 'Republican'}"
            if word != 'Toss-up' else "Neither candidate is favored"))
    ax.text(60, 468, sub, fontsize=20, color=fav_col if word != 'Toss-up' else INK_2, **font)

    # Win-chance bar: Dem side left, Rep right
    x0, x1, y, h = 60, 1140, 338, 54
    split = x0 + (x1 - x0) * p_d
    ax.add_patch(FancyBboxPatch((x0, y), max(split - x0 - 3, 0.1), h, boxstyle='round,pad=0,rounding_size=8', fc=d_col, ec='none'))
    ax.add_patch(FancyBboxPatch((split + 3, y), max(x1 - split - 3, 0.1), h, boxstyle='round,pad=0,rounding_size=8', fc=r_col, ec='none'))
    ax.text(x0, y - 38, f"{f['d_name']} ({d_letter})", fontsize=19, color=INK, **font)
    ax.text(x0, y + h + 14, f"{p_d:.0%}", fontsize=30, color=d_col, fontweight='bold', **font)
    ax.text(x1, y - 38, f"{f['r_name']} (R)", fontsize=19, color=INK, ha='right', **font)
    ax.text(x1, y + h + 14, f"{p_r:.0%}", fontsize=30, color=r_col, fontweight='bold', ha='right', **font)
    ax.text(600, y + h + 22, "chance of winning", fontsize=15, color=INK_3, ha='center', **font)

    # Margin and basis
    if not f['uncontested']:
        lo, hi = f['margin'] - RANGE_Z * f['sd'], f['margin'] + RANGE_Z * f['sd']
        ax.text(60, 230, f"Projected margin: {_fmt_margin(f['margin'], d_letter)}", fontsize=21, color=INK, fontweight='bold', **font)
        ax.text(60, 196, f"Likely range (80%): {_fmt_margin(lo, d_letter)} to {_fmt_margin(hi, d_letter)}", fontsize=17, color=INK_2, **font)
    ax.text(60, 150, f"Based on: {f['basis']}", fontsize=15, color=INK_2, **font)

    # Footer
    ax.plot([60, 1140], [112, 112], color=RULE, lw=2)
    ax.text(60, 78, "A forecast, not a call: probabilities from 20,000 simulated elections.", fontsize=14, color=INK_2, **font)
    ax.text(60, 50, "Polls compiled from pollsters' own releases. No prediction-market data.", fontsize=14, color=INK_3, **font)
    ax.text(1140, 78, BRAND, fontsize=16, color=INK, fontweight='bold', ha='right', **font)
    ax.text(1140, 50, f"model {model_version}", fontsize=13, color=INK_3, ha='right', **font)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG)
    plt.close(fig)
    return buf.getvalue()

def card_filename(race, office):
    if office == 'house':
        return f"house_{STATE_ABBREV[race['state']]}-{int(race['district']):02d}.png"
    return f"{office}_{race['state'].lower().replace(' ', '-')}.png"

def generate_all(forecast, model_version, out_dir=None, house='competitive'):
    """Cards for every Senate and governor race, and House seats (competitive ones by default)."""
    as_of = pd.Timestamp(forecast['as_of'])
    days = forecast['days_to_election']
    out_dir = out_dir or os.path.join(CARDS_DIR, f"{as_of:%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for office in ('senate', 'governor'):
        for _, r in forecast[office].iterrows():
            with open(os.path.join(out_dir, card_filename(r, office)), 'wb') as fh:
                fh.write(make_card(r, office, as_of, model_version, days))
            n += 1
    h = forecast['house']
    if house == 'competitive':
        h = h[(h['dem_win_prob'] > 0.1) & (h['dem_win_prob'] < 0.9)]
    for _, r in h.iterrows():
        with open(os.path.join(out_dir, card_filename(r, 'house')), 'wb') as fh:
            fh.write(make_card(r, 'house', as_of, model_version, days))
        n += 1
    return out_dir, n

if __name__ == '__main__':
    from data_sources import fetch_generic_ballot
    from forecast_2026 import build_forecast
    from publish_forecast import MODEL_VERSION
    gb, _ = fetch_generic_ballot()
    out, n = generate_all(build_forecast(gb), MODEL_VERSION)
    print(f"Wrote {n} cards to {os.path.abspath(out)}")
