"""
Social posts in the "race list" style: one line per race with a party-colored dot, the
leader's last name and lead, and "(Flip)" when the leader's party doesn't hold the seat.

    Senate Polling — Rasmussen Reports
    🔵 Iowa - Turek +4 (Flip)
    🟡 Alaska - Tied

Three sources, all built from data the forecast already uses:
  pollster   one pollster's latest poll in each race (last `days` days)
  averages   our polling averages (before the forecast's lean correction)
  forecast   our forecast margin, with the win probability

post_text() gives emoji text to paste; post_png() draws the same list as an image, since
emoji render differently across platforms and some strip them.
"""
import io
import re

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from states import STATE_ABBREV

DOT = {'DEMOCRAT': '🔵', 'REPUBLICAN': '🔴', 'INDEPENDENT': '🟣', None: '🟡'}
DOT_COLOR = {'DEMOCRAT': '#2a78d6', 'REPUBLICAN': '#e34948', 'INDEPENDENT': '#7b6fd6', None: '#eda100'}
# Below this lead a race reads "Tied" (half the last shown digit, so nothing rounds to +0)
TIE_MARGIN = {0: 0.5, 1: 0.05}
SUFFIXES = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv'}
OFFICE_LABEL = {'senate': 'Senate', 'governor': 'Governor', 'house': 'House'}

def house_label(state, district):
    """'PA-7', or 'WY-AL' for an at-large seat (district 0)."""
    d = int(district) if pd.notna(district) else 0
    return f"{STATE_ABBREV.get(state, state.title())}-{'AL' if d == 0 else d}"

def race_table(forecast, office):
    """The office's forecast table with a display label, the party holding the seat, and the
    forecast margin and win probability under one set of names. House keeps only contested
    districts that aren't already decided."""
    f = forecast[office].copy()
    if office == 'house':
        live = ~f['uncontested'].fillna(False).astype(bool) & ~f['already_decided'].fillna(False).astype(bool)
        f = f[live]
        return f.assign(label=[house_label(s, d) for s, d in zip(f['state'], f['district'])], held=f['held_by'],
                        margin=f['predicted_margin'], p=f['dem_win_prob'], dem_side_party='DEMOCRAT')
    return f.assign(label=f['state'].str.title(), held=f['incumbent_party'],
                    margin=f['forecast_margin'], p=f['dem_side_win_prob'])

def _poll_label(office, r):
    return house_label(r['state'], r['district']) if office == 'house' else r['state'].title()

def last_name(full):
    tokens = [t for t in str(full).split() if t.lower().strip(',') not in SUFFIXES]
    return tokens[-1].strip(',') if tokens else str(full)

def pollster_key(name):
    """Pollster name without a sponsor tag, so 'Rasmussen Reports (R)' and 'Rasmussen
    Reports' count as one pollster."""
    return re.sub(r'\s*\((?:D|R|I)\)\s*$', '', str(name)).strip()

def _row(label, dem_margin, dem_name, rep_name, dem_side_party, incumbent_party, extra=None, decimals=0):
    """One line: who leads, by how much, and whether that flips the seat. label is the race
    as shown ('Iowa', 'PA-7')."""
    if abs(dem_margin) < TIE_MARGIN[decimals]:
        party, name = None, None
    elif dem_margin > 0:
        party, name = dem_side_party, last_name(dem_name)
    else:
        party, name = 'REPUBLICAN', last_name(rep_name)
    return {'state': label, 'party': party, 'name': name, 'lead': abs(dem_margin),
            'dem_margin': dem_margin, 'flip': party is not None and isinstance(incumbent_party, str)
            and party != incumbent_party, 'extra': extra, 'decimals': decimals}

def pollster_choices(annotated, office, days=60, min_races=2):
    """Pollsters with polls in at least `min_races` races of this office recently, busiest first."""
    a = annotated[annotated['office'] == office]
    a = a[a['end_date'] > a['end_date'].max() - pd.Timedelta(days=days)]
    a = a.assign(key=a['pollster'].map(pollster_key), race=[_poll_label(office, r) for _, r in a.iterrows()])
    counts = a.groupby('key')['race'].nunique()
    return counts[counts >= min_races].sort_values(ascending=False).index.tolist()

def rows_from_pollster(annotated, forecast, office, pollster, days=60):
    """Each race's most recent poll from this pollster in the last `days` days."""
    a = annotated[(annotated['office'] == office) & (annotated['pollster'].map(pollster_key) == pollster)]
    a = a[a['end_date'] > annotated['end_date'].max() - pd.Timedelta(days=days)]
    a = a.assign(race=[_poll_label(office, r) for _, r in a.iterrows()])
    latest = a.sort_values('end_date').groupby('race').tail(1)
    held = race_table(forecast, office).set_index('label')['held']
    rows = [_row(r['race'], r['margin'], r['dem_candidate'], r['rep_candidate'], r['dem_side_party'],
                 held.get(r['race'])) for _, r in latest.iterrows()]
    dates = (latest['end_date'].min(), latest['end_date'].max()) if len(latest) else (None, None)
    return sort_rows(rows), dates

def rows_from_averages(forecast, office):
    f = race_table(forecast, office).dropna(subset=['poll_avg'])
    return sort_rows([_row(r['label'], r['poll_avg'], r['dem_candidate'], r['rep_candidate'], r['dem_side_party'],
                           r['held'], extra=f"{int(r['n_polls'])} poll{'' if int(r['n_polls']) == 1 else 's'}", decimals=1) for _, r in f.iterrows()])

def rows_from_forecast(forecast, office):
    f = race_table(forecast, office)
    rows = []
    for _, r in f.iterrows():
        p = float(r['p'])
        row = _row(r['label'], r['margin'], r['dem_candidate'], r['rep_candidate'], r['dem_side_party'],
                   r['held'], decimals=1)
        row['extra'] = f"{max(p, 1 - p):.0%}"
        rows.append(row)
    return sort_rows(rows)

def sort_rows(rows):
    """Democratic-side leaders first, then ties, then Republicans; flips lead each group,
    then by size of lead (the order these lists usually take)."""
    group = lambda r: 2 if r['party'] == 'REPUBLICAN' else (1 if r['party'] is None else 0)
    return sorted(rows, key=lambda r: (group(r), not r['flip'], -r['lead']))

def _line(r, show_flip=True, show_extra=False):
    if r['party'] is None:
        body = f"{r['state']} - Tied"
    else:
        body = f"{r['state']} - {r['name']} +{r['lead']:.{r['decimals']}f}"
    if show_extra and r['extra']:
        body += f" · {r['extra']}"
    if show_flip and r['flip']:
        body += " (Flip)"
    return body

def post_text(title, rows, footer=None, show_flip=True, show_extra=False):
    lines = [title, ''] + [f"{DOT[r['party']]} {_line(r, show_flip, show_extra)}" for r in rows]
    if footer:
        lines += ['', footer]
    return '\n'.join(lines)

def post_png(title, rows, footer=None, show_flip=True, show_extra=False, dark=True):
    """The list as an image: 1200 px wide, height grows with the number of races."""
    bg, ink, ink_2 = ('#15202b', '#ffffff', '#8b98a5') if dark else ('#ffffff', '#111111', '#55534f')
    line_h, top, bottom = 0.62, 1.3, (0.9 if footer else 0.4)
    h_in = top + line_h * len(rows) + bottom
    fig = plt.figure(figsize=(12, h_in), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(h_in, 0); ax.axis('off')
    ax.text(0.5, 0.75, title, fontsize=30, color=ink, va='center', fontweight='bold')
    for i, r in enumerate(rows):
        y = top + line_h * (i + 0.5)
        ax.add_patch(Circle((0.72, y), 0.19, color=DOT_COLOR[r['party']]))
        ax.text(1.1, y, _line(r, show_flip, show_extra), fontsize=24, color=ink, va='center')
    if footer:
        ax.text(0.5, h_in - 0.45, footer, fontsize=15, color=ink_2, va='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=bg)
    plt.close(fig)
    return buf.getvalue()
