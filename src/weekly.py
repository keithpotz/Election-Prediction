"""
What moved: compare two published forecast snapshots (forecasts/2026/<date>_*.csv) and
explain each race's change.

Each statewide race's forecast is a blend, margin = w * polls + (1 - w) * fundamentals,
so a change splits into:
  polls        new polls, or old ones fading with age (holding fundamentals at the old value)
  environment  the fundamentals estimate moving, mostly the national environment
  method       whatever is left: a model change between the two snapshots, plus rounding

    python src/weekly.py                       # last 7 days
    python src/weekly.py --since 2026-10-06

writes forecasts/substack/what_moved_<date>.md (paste into Substack) and a matching .png.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from brand import MODEL_NAME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE_DIR, '..', 'forecasts', '2026')
CHANGELOG = os.path.join(BASE_DIR, '..', 'forecasts', 'CHANGELOG.md')
OFFICE_LABEL = {'senate': 'Senate', 'governor': 'Governor', 'house': 'House'}
PARTY_LETTER = {'DEMOCRAT': 'D', 'REPUBLICAN': 'R', 'INDEPENDENT': 'I'}
MIN_MOVE = 0.03          # win-chance moves smaller than this aren't news
METHOD_SHARE = 0.5       # call a move a method change when the leftover is at least half of it

def snapshot_dates():
    return sorted({os.path.basename(p)[:10] for p in glob.glob(os.path.join(SNAP_DIR, '*_summary.json'))})

def load_snapshot(day):
    with open(os.path.join(SNAP_DIR, f'{day}_summary.json')) as fh:
        summary = json.load(fh)
    snap = {'date': pd.Timestamp(day), 'summary': summary}
    for office, name in (('senate', 'senate_races'), ('governor', 'governor_races'), ('house', 'house_districts')):
        snap[office] = pd.read_csv(os.path.join(SNAP_DIR, f'{day}_{name}.csv'))
    return snap

def default_pair(days=7):
    """The latest snapshot and the earliest one within `days` days of it."""
    dates = snapshot_dates()
    if len(dates) < 2:
        return None
    last = pd.Timestamp(dates[-1])
    first = next(d for d in dates if pd.Timestamp(d) >= last - pd.Timedelta(days=days))
    return (first if first != dates[-1] else dates[-2]), dates[-1]

def _last(name):
    tokens = [t for t in str(name).split() if t.lower().strip(',.') not in {'jr', 'sr', 'ii', 'iii', 'iv'}]
    return tokens[-1].strip(',') if tokens else str(name)

def _poll_avg(df):
    return df['poll_avg_corrected'] if 'poll_avg_corrected' in df else df['poll_avg']

def _new_polls(annotated, office, state, after):
    """Polls in this race that ended after the old snapshot's latest poll."""
    if annotated is None:
        return []
    a = annotated[(annotated['office'] == office) & (annotated['state'] == state)]
    if pd.notna(after):
        a = a[a['end_date'] > pd.Timestamp(after)]
    return [{'pollster': r['pollster'], 'end_date': r['end_date'], 'margin': r['margin'],
             'dem_side_party': r['dem_side_party']} for _, r in a.sort_values('end_date').iterrows()]

def statewide_moves(old, new, office, annotated=None):
    """One row per race in both snapshots: old and new win chance and margin, and the split
    of the margin change into polls, environment and method."""
    a = old[office].set_index('state')
    b = new[office].set_index('state')
    rows = []
    for state in b.index.intersection(a.index):
        r0, r1 = a.loc[state], b.loc[state]
        w0, w1 = float(r0['poll_weight']), float(r1['poll_weight'])
        f0, f1 = float(r0['fund_pred']), float(r1['fund_pred'])
        m0, m1 = float(r0['forecast_margin']), float(r1['forecast_margin'])
        # Polls effect: the new blend minus the old one, holding fundamentals at the old value
        # and reading the old raw average through today's lean correction, so a change in
        # that correction counts as method, not polls.
        raw0, raw1 = r0['poll_avg'], r1['poll_avg']
        c1 = _poll_avg(b).get(state)
        shift = (c1 - raw1) if pd.notna(c1) and pd.notna(raw1) else 0.0
        c0 = raw0 + shift if pd.notna(raw0) else np.nan
        blend = lambda w, c, f: w * (c if pd.notna(c) else 0.0) + (1 - w) * f
        polls = blend(w1, c1, f0) - blend(w0, c0, f0)
        env = (1 - w1) * (f1 - f0)
        n0, n1 = int(r0['n_polls']), int(r1['n_polls'])
        rows.append({
            'office': office, 'state': state,
            'dem_side_party': r1['dem_side_party'], 'dem_candidate': r1['dem_candidate'],
            'rep_candidate': r1['rep_candidate'], 'incumbent_party': r1['incumbent_party'],
            'p_before': float(r0['dem_side_win_prob']), 'p_after': float(r1['dem_side_win_prob']),
            'margin_before': m0, 'margin_after': m1,
            'poll_avg_before': raw0, 'poll_avg_after': raw1, 'n_polls_before': n0, 'n_polls_after': n1,
            'from_polls': polls, 'from_environment': env, 'from_method': (m1 - m0) - polls - env,
            'new_polls': _new_polls(annotated, office, state, r0.get('latest_poll')),
        })
    m = pd.DataFrame(rows)
    m['change'] = m['p_after'] - m['p_before']
    return m

def house_moves(old, new):
    key = ['state', 'district']
    a, b = old['house'].set_index(key), new['house'].set_index(key)
    idx = b.index.intersection(a.index)
    col = lambda df, c: df.loc[idx, c] if c in df else pd.Series(np.nan, index=idx)  # older snapshots lack some columns
    m = pd.DataFrame({
        'p_before': a.loc[idx, 'dem_win_prob'], 'p_after': b.loc[idx, 'dem_win_prob'],
        'margin_before': a.loc[idx, 'predicted_margin'], 'margin_after': b.loc[idx, 'predicted_margin'],
        'n_polls_before': col(a, 'n_polls'), 'n_polls_after': col(b, 'n_polls'),
        'dem_candidate': col(b, 'dem_candidate'), 'rep_candidate': col(b, 'rep_candidate'),
        'incumbent_party': col(b, 'incumbent_party'),
    }).reset_index()
    m['change'] = m['p_after'] - m['p_before']
    return m

def reason(r, method_changed):
    """A short, plain explanation of one race's move, from its biggest component."""
    parts = {'polls': r['from_polls'], 'environment': r['from_environment'], 'method': r['from_method']}
    total = r['margin_after'] - r['margin_before']
    if method_changed and abs(parts['method']) >= METHOD_SHARE * max(abs(total), 0.5):
        return "Mostly the model update"
    if abs(parts['environment']) > abs(parts['polls']):
        return "The national environment moved" + ("" if r['n_polls_after'] else " (no polls in this race)")
    new = r['new_polls']
    if new:
        n = len(new)
        return f"{n} new poll{'s' if n > 1 else ''}: " + "; ".join(_poll_phrase(p, r) for p in new[-3:]) \
            + ("; ..." if n > 3 else "")
    return "Older polls counting for less as they age"

def _poll_phrase(p, r):
    if abs(p['margin']) < 0.5:
        lead = "tied"
    elif p['margin'] > 0:
        lead = f"{_last(r['dem_candidate'])} +{p['margin']:.0f}"
    else:
        lead = f"{_last(r['rep_candidate'])} +{-p['margin']:.0f}"
    return f"{p['pollster']}, {p['end_date']:%b} {p['end_date'].day}: {lead}"

def changelog_versions():
    """Model versions in the changelog, oldest first, with each heading's title."""
    if not os.path.exists(CHANGELOG):
        return []
    with open(CHANGELOG, encoding='utf-8') as fh:
        heads = re.findall(r'^## (\S+?): (.+)$', fh.read(), flags=re.M)
    return heads[::-1]

def method_changes(v0, v1):
    """Changelog entries after v0, up to and including v1."""
    versions = changelog_versions()
    names = [v for v, _ in versions]
    if v0 == v1 or v0 not in names or v1 not in names:
        return []
    i0, i1 = names.index(v0), names.index(v1)
    return versions[i0 + 1:i1 + 1]

def what_moved(day0, day1, annotated=None, n=8):
    old, new = load_snapshot(day0), load_snapshot(day1)
    s0, s1 = old['summary'], new['summary']
    v0, v1 = s0['model_version'], s1['model_version']
    method_changed = v0 != v1
    races = pd.concat([statewide_moves(old, new, o, annotated) for o in ('senate', 'governor')], ignore_index=True)
    races['reason'] = [reason(r, method_changed) for _, r in races.iterrows()]
    big = races[races['change'].abs() >= MIN_MOVE]
    big = big.loc[big['change'].abs().sort_values(ascending=False).index].head(n)
    house = house_moves(old, new)
    house = house[house['change'].abs() >= 0.05]
    house = house.loc[house['change'].abs().sort_values(ascending=False).index].head(5)
    top = {
        'senate': (s0['senate']['dem_control_prob'], s1['senate']['dem_control_prob']),
        'house': (s0['house']['dem_control_prob'], s1['house']['dem_control_prob']),
        'senate_seats': (s0['senate']['dem_seats_median'], s1['senate']['dem_seats_median']),
        'house_seats': (s0['house']['dem_seats_median'], s1['house']['dem_seats_median']),
        'governors': (s0['governor']['dem_governorships_median'], s1['governor']['dem_governorships_median']),
        'generic_ballot': (s0['generic_ballot_dem_margin'], s1['generic_ballot_dem_margin']),
        'environment': (s0['national_environment_used'], s1['national_environment_used']),
    }
    return {'start': old['date'], 'end': new['date'], 'versions': (v0, v1),
            'method_changes': method_changes(v0, v1), 'topline': top,
            'movers': big.reset_index(drop=True), 'house': house.reset_index(drop=True), 'all': races}

# ── Post text ────────────────────────────────────────────────────────────────
def _d(ts):
    return f"{ts:%b} {ts.day}"

def _pts(a, b):
    """'up 3 points' / 'down 3 points' / 'unchanged' for a probability move."""
    d = round((b - a) * 100)
    return "unchanged" if d == 0 else f"{'up' if d > 0 else 'down'} {abs(d)} point{'s' if abs(d) != 1 else ''}"

def _side(r):
    party = PARTY_LETTER.get(r['dem_side_party'], 'D')
    return f"{_last(r['dem_candidate'])} ({party})"

def race_line(r):
    office = OFFICE_LABEL[r['office']]
    return (f"**{str(r['state']).title()} {office}:** {_side(r)} {r['p_before']:.0%} → {r['p_after']:.0%}. "
            f"{r['reason']}.")

def house_line(r):
    seat = f"{str(r['state']).title()} at-large" if int(r['district']) == 0 else f"{str(r['state']).title()}-{int(r['district'])}"
    return (f"**{seat}:** Democrat {_last(r['dem_candidate'])} "
            f"{r['p_before']:.0%} → {r['p_after']:.0%}")

def post_markdown(w):
    t = w['topline']
    span = f"{_d(w['start'])} to {_d(w['end'])}"
    sen0, sen1 = t['senate']; h0, h1 = t['house']
    lines = [f"# What moved: {span}", "",
             f"*{MODEL_NAME}, model {w['versions'][1]}. Every number here is from a forecast published on that date.*", "",
             "## The big picture", "",
             f"- **Senate:** Democrats' chance of control is {sen1:.0%}, {_pts(sen0, sen1)} since {_d(w['start'])} "
             f"(median {t['senate_seats'][1]} seats, was {t['senate_seats'][0]}).",
             f"- **House:** {h1:.0%}, {_pts(h0, h1)} (median {t['house_seats'][1]} Democratic seats, was {t['house_seats'][0]}).",
             f"- **Governors:** Democrats are projected to win a median of {t['governors'][1]} of 36 "
             f"(was {t['governors'][0]}).",
             f"- **Generic ballot:** D+{t['generic_ballot'][1]:.1f} (was D+{t['generic_ballot'][0]:.1f}). After the "
             f"historical lean correction, the forecast reads that as a national environment of D+{t['environment'][1]:.1f} "
             f"(was D+{t['environment'][0]:.1f}).", ""]
    if w['method_changes']:
        lines += ["## Changes to the model", "",
                  "Some of this week's movement comes from changes to the method, not new information. Each one is "
                  "logged in the public changelog:", ""]
        lines += [f"- **{v}:** {title}" for v, title in w['method_changes']] + [""]
    lines += ["## The races that moved", ""]
    if len(w['movers']):
        lines += [f"- {race_line(r)}" for _, r in w['movers'].iterrows()] + [""]
    else:
        lines += [f"No Senate or governor race moved by {MIN_MOVE:.0%} or more.", ""]
    if len(w['house']):
        lines += ["**House districts that moved most:**", ""]
        lines += [f"- {house_line(r)}" for _, r in w['house'].iterrows()] + [""]
    lines += ["Win chances are for the candidate named. \"New polls\" are polls that finished fielding after the "
              "earlier forecast's latest poll in that race."]
    return '\n'.join(lines)

def movers_png(w, dark=True, n=8):
    """The movers as a share image: one row per race, old → new win chance with an arrow."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DEM, REP, IND = '#2a78d6', '#e34948', '#7b6fd6'
    bg, ink, ink_2, rule = ('#15202b', '#ffffff', '#8b98a5', '#2a3a48') if dark else ('#ffffff', '#111111', '#55534f', '#e2e1dc')
    m = w['movers'].head(n)
    row_h, top, bottom = 0.7, 1.75, 0.8
    h_in = top + row_h * max(len(m), 1) + bottom
    fig = plt.figure(figsize=(12, h_in), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(h_in, 0); ax.axis('off')
    ax.text(0.5, 0.55, MODEL_NAME.upper(), fontsize=13, color=ink_2, fontweight='bold', va='center')
    ax.text(0.5, 1.1, f"What moved, {_d(w['start'])} – {_d(w['end'])}", fontsize=28, color=ink, fontweight='bold', va='center')
    x0, x1 = 6.2, 11.3          # the 0-100% scale
    ax.text(x0, 1.55, "0%", fontsize=11, color=ink_2, ha='center'); ax.text(x1, 1.55, "100%", fontsize=11, color=ink_2, ha='center')
    ax.text((x0 + x1) / 2, 1.55, "50%", fontsize=11, color=ink_2, ha='center')
    xs = lambda p: x0 + (x1 - x0) * p
    for i, (_, r) in enumerate(m.iterrows()):
        y = top + row_h * (i + 0.5)
        ax.plot([x0, x1], [y, y], color=rule, lw=1, zorder=0)
        party_color = IND if r['dem_side_party'] == 'INDEPENDENT' else DEM
        color = party_color if r['change'] > 0 else REP
        ax.text(0.5, y - 0.1, f"{str(r['state']).title()} {OFFICE_LABEL[r['office']]}", fontsize=19, color=ink, va='center')
        ax.text(0.5, y + 0.2, f"{_side(r)} win chance", fontsize=11, color=ink_2, va='center')
        ax.text(5.0, y, f"{r['p_before']:.0%} → {r['p_after']:.0%}", fontsize=17, color=ink, va='center', ha='center')
        ax.annotate('', xy=(xs(r['p_after']), y), xytext=(xs(r['p_before']), y),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=3, mutation_scale=20))
        ax.plot([xs(r['p_before'])], [y], 'o', color=ink_2, ms=8)
    ax.plot([xs(0.5)] * 2, [top - 0.1, top + row_h * max(len(m), 1)], color=ink_2, lw=1, ls=(0, (2, 3)), zorder=0)
    foot = f"{MODEL_NAME} · forecasts of {_d(w['start'])} and {_d(w['end'])}, {w['end'].year} · model {w['versions'][1]}"
    ax.text(0.5, h_in - 0.4, foot, fontsize=12, color=ink_2, va='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=bg)
    plt.close(fig)
    return buf.getvalue()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', help='earlier snapshot date (YYYY-MM-DD); default 7 days before the latest')
    ap.add_argument('--until', help='later snapshot date; default the latest')
    args = ap.parse_args()
    dates = snapshot_dates()
    d1 = args.until or dates[-1]
    d0 = args.since or default_pair()[0]
    from forecast_2026 import load_inputs
    w = what_moved(d0, d1, load_inputs()['annotated_polls'])
    out = os.path.join(BASE_DIR, '..', 'forecasts', 'substack', f'what_moved_{d1}')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.md', 'w', encoding='utf-8') as fh:
        fh.write(post_markdown(w) + '\n')
    with open(out + '.png', 'wb') as fh:
        fh.write(movers_png(w))
    print(f"Wrote {os.path.abspath(out)}.md and .png ({d0} to {d1})")
