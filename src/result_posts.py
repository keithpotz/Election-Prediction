"""
Social posts about how the forecast did, built from the Results page's data
(dashboard_results.build). Same look as posts.py: emoji text to paste, plus a PNG.

    report card       chamber calls, winners right, Brier score against the experts
    forecast vs result one line per race: winner, margin, our chance for them, ✓ or ✗
    misses            races where our favorite lost
    polling error     how far each race's polls missed, and which way
    pollsters         the most accurate pollsters in the final weeks
    alternatives      our forecast against polls alone, fundamentals alone and the experts
"""
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

import grade_2026 as GR
import posts as P

OK_EMOJI, MISS_EMOJI = '✅', '❌'
OK_MARK, MISS_MARK = '✓', '✗'
METHOD_ORDER = ['Our forecast', 'Experts', 'Polls only', 'Fundamentals only']


def fmt_margin(m):
    return f"D+{m:.1f}" if m > 0 else (f"R+{-m:.1f}" if m < 0 else "Even")


def race_name(r):
    return P.house_label(r['state'], r['district']) if r['office'] == 'house' else r['state'].title()


def winner_name(r):
    return P.last_name(r['rep_name'] if r['winner_party'] == 'REPUBLICAN' else r['dem_name'])


def favorite_won(r):
    return bool((r['model_p'] >= 0.5) == r['won'])


def _line(party, body, ok=None):
    return {'party': party, 'body': body, 'ok': ok}


# ── Race lists ────────────────────────────────────────────────────────────────
def forecast_vs_result(g, office, which='competitive'):
    """One line per race: the winner and margin, (Flip), and our chance for the winner."""
    d = g[g['office'] == office]
    if which == 'competitive':
        d = d[d['model_p'].between(0.1, 0.9)]
    elif which == 'flips':
        d = d[d['flip']]
    elif which == 'misses':
        d = d[(d['model_p'] >= 0.5) != d['won']]
    d = d.assign(_grp=np.where(d['winner_party'] == 'REPUBLICAN', 1, 0)).sort_values(['_grp', 'p_winner'])
    lines = []
    for _, r in d.iterrows():
        margin = f" +{abs(r['actual']):.1f}" if pd.notna(r['actual']) else ' won'
        flip = ' (Flip)' if r['flip'] else ''
        lines.append(_line(r['winner_party'], f"{race_name(r)} - {winner_name(r)}{margin}{flip} · we had "
                                              f"{r['p_winner']:.0%}", favorite_won(r)))
    return lines


def misses(g, office=None):
    d = g if office is None else g[g['office'] == office]
    d = d[(d['model_p'] >= 0.5) != d['won']].copy()
    d['conf'] = np.maximum(d['model_p'], 1 - d['model_p'])
    lines = []
    for _, r in d.sort_values('conf', ascending=False).iterrows():
        loser = P.last_name(r['dem_name'] if r['winner_party'] == 'REPUBLICAN' else r['rep_name'])
        where = race_name(r) if office else (race_name(r) if r['office'] == 'house'
                                              else f"{r['state'].title()} {r['office'].title()}")
        margin = f" +{abs(r['actual']):.1f}" if pd.notna(r['actual']) else ''
        lines.append(_line(r['winner_party'], f"{where} - {winner_name(r)}{margin} · we gave {loser} {r['conf']:.0%}",
                           False))
    return lines


def polling_error(g, office):
    """Each race's result minus its polling average. Blue: Democrats beat their polls."""
    d = g[(g['office'] == office) & g['poll_avg'].notna() & g['actual'].notna()].copy()
    d['miss'] = d['actual'] - d['poll_avg']
    lines = []
    for _, r in d.sort_values('miss', ascending=False).iterrows():
        side = 'Dems' if r['miss'] > 0 else 'GOP'
        lines.append(_line('DEMOCRAT' if r['miss'] > 0 else 'REPUBLICAN',
                           f"{race_name(r)} - {side} beat polls by {abs(r['miss']):.1f} (polls {fmt_margin(r['poll_avg'])}, "
                           f"result {fmt_margin(r['actual'])})"))
    avg = float(d['miss'].mean()) if len(d) else 0.0
    return lines, avg, len(d)


def pollster_lines(m, top=10, min_polls=2):
    t = m.groupby('pollster').agg(polls=('error', 'size'), avg_abs=('error', lambda e: e.abs().mean()),
                                  bias=('error', 'mean')).reset_index()
    t = t[t['polls'] >= min_polls].sort_values('avg_abs').head(top)
    lines = []
    for i, r in enumerate(t.itertuples(), 1):
        party = 'DEMOCRAT' if r.bias > 0.5 else ('REPUBLICAN' if r.bias < -0.5 else None)
        lean = 'no lean' if party is None else f"leaned {fmt_margin(r.bias)}"
        lines.append(_line(party, f"{i}. {r.pollster} - off by {r.avg_abs:.1f} on avg · {lean} · {r.polls} polls"))
    return lines


def alternatives(scores, final_date, office='all'):
    s = scores[(scores['snapshot'] == final_date) & (scores['office'] == office)].set_index('method')
    lines = []
    for method in METHOD_ORDER:
        if method in s.index:
            r = s.loc[method]
            lines.append(_line('DEMOCRAT' if method == 'Our forecast' else None,
                               f"{method} - Brier {r['brier']:.3f} · {r['winners_right']:.0%} right · "
                               f"{int(r['races'])} races"))
    return lines


# ── Report card ───────────────────────────────────────────────────────────────
def report_card(data):
    """Tiles (big value, label, note) for the headline scorecard."""
    g, res, s = data['final'], data['res'], data['summary']
    ch = GR.chamber_outcomes(res)
    tiles = []
    for key, name, p in (('senate', 'Senate', s['senate']['dem_control_prob']),
                         ('house', 'House', s['house']['dem_control_prob'])):
        seats = ch.get(f'{key}_dem_seats')
        ctrl = ch.get(f'{key}_dem_control')
        if ctrl is None:
            tiles.append(('Not decided', name, f"{seats} Dem seats so far · we gave Dems {p:.0%}"))
        else:
            right = (p >= 0.5) == ctrl
            tiles.append((f"{'Dems' if ctrl else 'GOP'} {OK_MARK if right else MISS_MARK}", name,
                          f"we gave Dems {p:.0%} · {seats} Dem seats"))
    right = (g['model_p'] >= 0.5) == g['won']
    comp = g[g['model_p'].between(0.1, 0.9)]
    comp_right = (comp['model_p'] >= 0.5) == comp['won']
    tiles.append((f"{int(right.sum())}/{len(g)}", "Winners called", f"{right.mean():.1%} right"))
    tiles.append((f"{int(comp_right.sum())}/{len(comp)}", "Competitive races",
                  "races we had under 90%" if len(comp) else ""))
    fin = data['scores'][(data['scores']['snapshot'] == data['final_date']) & (data['scores']['office'] == 'all')]
    fin = fin.set_index('method')
    ours = fin['brier'].get('Our forecast', np.nan)
    exp = fin['brier'].get('Experts', np.nan)
    tiles.append((f"{ours:.3f}", "Brier score", f"experts {exp:.3f} · lower is better" if pd.notna(exp)
                  else "lower is better"))
    gov = res[res['office'] == 'governor']
    tiles.append((f"{int((gov['winner_party'] == 'DEMOCRAT').sum())} D", "Governors",
                  f"forecast {s['governor']['dem_governorships_median']}, range "
                  f"{s['governor']['dem_governorships_90pct'][0]}–{s['governor']['dem_governorships_90pct'][1]}"))
    return tiles


def report_text(title, tiles, footer=None, extra=None):
    lines = [title, '']
    for value, label, note in tiles:
        v = value.replace(OK_MARK, OK_EMOJI).replace(MISS_MARK, MISS_EMOJI)
        lines.append(f"{label}: {v}" + (f" ({note})" if note else ''))
    if extra:
        lines += [''] + extra
    if footer:
        lines += ['', footer]
    return '\n'.join(lines)


def _colors(dark):
    return ('#15202b', '#ffffff', '#8b98a5', '#1e2c3a') if dark else ('#ffffff', '#111111', '#55534f', '#f2f1ec')


def report_png(title, tiles, footer=None, dark=True, subtitle=None):
    bg, ink, ink_2, tile_bg = _colors(dark)
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    tile_w, tile_h, gap, left = 3.55, 2.0, 0.25, 0.5
    top = 1.75 if subtitle else 1.3
    h_in = top + rows * (tile_h + gap) + (0.9 if footer else 0.4)
    fig = plt.figure(figsize=(12, h_in), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(h_in, 0); ax.axis('off')
    ax.text(0.5, 0.75, title, fontsize=30, color=ink, va='center', fontweight='bold')
    if subtitle:
        ax.text(0.5, 1.3, subtitle, fontsize=17, color=ink_2, va='center')
    for i, (value, label, note) in enumerate(tiles):
        x = left + (i % cols) * (tile_w + gap)
        y = top + (i // cols) * (tile_h + gap)
        ax.add_patch(FancyBboxPatch((x, y), tile_w, tile_h, boxstyle='round,pad=0,rounding_size=0.15',
                                    facecolor=tile_bg, edgecolor='none'))
        color = ink
        if OK_MARK in value:
            color = '#1baf7a'
        elif MISS_MARK in value:
            color = P.DOT_COLOR['REPUBLICAN']
        ax.text(x + 0.25, y + 0.45, label.upper(), fontsize=13, color=ink_2, va='center', fontweight='bold')
        ax.text(x + 0.25, y + 1.05, value, fontsize=34, color=color, va='center', fontweight='bold')
        ax.text(x + 0.25, y + 1.62, note, fontsize=12.5, color=ink_2, va='center')
    if footer:
        ax.text(0.5, h_in - 0.45, footer, fontsize=15, color=ink_2, va='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


# ── Text and image for line posts ─────────────────────────────────────────────
def lines_text(title, lines, footer=None, header=None):
    out = [title, '']
    if header:
        out += [header, '']
    for ln in lines:
        mark = '' if ln['ok'] is None else (f" {OK_EMOJI}" if ln['ok'] else f" {MISS_EMOJI}")
        out.append(f"{P.DOT[ln['party']]} {ln['body']}{mark}")
    if footer:
        out += ['', footer]
    return '\n'.join(out)


def lines_png(title, lines, footer=None, dark=True, header=None):
    """1200 px wide, height grows with the number of lines (like posts.post_png)."""
    bg, ink, ink_2, _ = _colors(dark)
    line_h = 0.62
    top = 1.3 + (0.6 if header else 0)
    h_in = top + line_h * len(lines) + (0.9 if footer else 0.4)
    fig = plt.figure(figsize=(12, h_in), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(h_in, 0); ax.axis('off')
    ax.text(0.5, 0.75, title, fontsize=30, color=ink, va='center', fontweight='bold')
    if header:
        ax.text(0.5, 1.4, header, fontsize=18, color=ink_2, va='center')
    size = 24 if max((len(ln['body']) for ln in lines), default=0) < 52 else 19
    for i, ln in enumerate(lines):
        y = top + line_h * (i + 0.5)
        ax.add_patch(Circle((0.72, y), 0.19, color=P.DOT_COLOR[ln['party']]))
        t = ax.text(1.1, y, ln['body'], fontsize=size, color=ink, va='center')
        if ln['ok'] is not None:
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transData.inverted())
            ax.text(bb.x1 + 0.15, y, OK_MARK if ln['ok'] else MISS_MARK, fontsize=size, va='center', fontweight='bold',
                    color='#1baf7a' if ln['ok'] else P.DOT_COLOR['REPUBLICAN'])
    if footer:
        ax.text(0.5, h_in - 0.45, footer, fontsize=15, color=ink_2, va='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=bg)
    plt.close(fig)
    return buf.getvalue()
