"""
Draft of a full-forecast Substack post: the Senate, House and governors, the races to
watch, and how to read the numbers. Every number comes from the forecast; the prose
around them is a template to edit before posting.

Run it right after the daily update, so the post matches a published snapshot:

    python src/daily_update.py
    python src/launch_post.py

Writes forecasts/substack/forecast_<date>/: post.md plus senate.png, house.png and
path_to_51.png to upload where the post marks them.
"""
import json
import os

import numpy as np

from brand import MODEL_NAME, PUBLICATION, TAGLINE
from forecast_2026 import build_forecast, load_inputs
from publish_forecast import MODEL_VERSION, OUT_DIR
import weekly as W

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _title(state):
    return ' '.join(w if w.isupper() and len(w) <= 2 else w.capitalize() for w in str(state).split())

def _last(name):
    return W._last(name)

def _pct(p):
    return f"{p:.0%}"

def _margin(m, dem_name, rep_name):
    if abs(m) < 0.5:
        return "even"
    return f"{_last(dem_name)} +{m:.1f}" if m > 0 else f"{_last(rep_name)} +{-m:.1f}"

def _list(items):
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + (',' if len(items) > 2 else '') + ' and ' + items[-1]

def latest_snapshot():
    dates = W.snapshot_dates()
    if not dates:
        raise SystemExit("No published forecast yet. Run python src/daily_update.py first.")
    with open(os.path.join(OUT_DIR, f'{dates[-1]}_summary.json')) as fh:
        return dates[-1], json.load(fh)

def senate_section(f):
    s = f['summary']['senate']
    sen = f['senate']
    held_d = s['holdovers']['DEMOCRAT'] + int((sen['incumbent_party'] == 'DEMOCRAT').sum())
    gain_needed = 51 - held_d
    lo, hi = s['dem_seats_90pct']
    pickups = sen[(sen['incumbent_party'] == 'REPUBLICAN') & (sen['dem_side_party'] == 'DEMOCRAT')
                  & (sen['dem_side_win_prob'] > 0.5)].sort_values('dem_side_win_prob', ascending=False)
    at_risk = sen[(sen['incumbent_party'] == 'DEMOCRAT') & (sen['dem_side_win_prob'] < 0.8)].sort_values('dem_side_win_prob')
    watch = sen.sort_values('tipping_point_prob', ascending=False).head(6)
    lines = [
        "## The Senate", "",
        "*[Image: senate.png]*", "",
        f"Democrats have a **{_pct(s['dem_control_prob'])} chance** of winning the Senate, and Republicans "
        f"{_pct(s['rep_control_prob'])}. Democrats hold {held_d} seats going in, so they need a net gain of "
        f"{gain_needed} to reach 51. Republicans need only 50, since Vice President Vance breaks ties. Our median "
        f"outcome is {s['dem_seats_median']} Democratic seats, and the 90% range runs from {lo} to {hi}.", "",
    ]
    if len(pickups):
        lines += [f"Democrats currently lead in {len(pickups)} Republican-held seats: "
                  + _list(f"{_title(r['state'])} ({_pct(r['dem_side_win_prob'])})" for _, r in pickups.iterrows()) + ". "
                  + ("None of their own seats is rated worse than a 20% risk."
                     if at_risk.empty else
                     "Their most vulnerable seats are " + _list(f"{_title(r['state'])} ({_pct(r['dem_side_win_prob'])})"
                                                               for _, r in at_risk.iterrows()) + "."), ""]
    ind = sen[sen['dem_side_party'] == 'INDEPENDENT']
    if len(ind):
        lines += ["Independents " + _list(f"{r['dem_candidate']} in {_title(r['state'])} ({_pct(r['dem_side_win_prob'])})"
                                          for _, r in ind.iterrows())
                  + " don't count toward either party's total"
                  + (", though a win would still cost Republicans a seat." if (ind['incumbent_party'] == 'REPUBLICAN').all()
                     else "."), ""]
    lines += [
        "*[Image: path_to_51.png]*", "",
        "**The races that decide it.** Line up all 35 races from most to least Democratic, and the race that "
        "delivers the 51st seat is the tipping point. No single race decides it most of the time, which is what a "
        "close Senate looks like:", ""]
    for _, r in watch.iterrows():
        lines.append(f"- **{_title(r['state'])}:** {r['dem_candidate']} vs. {r['rep_candidate']}. "
                     f"{_last(r['dem_candidate'])} {_pct(r['dem_side_win_prob'])}; forecast "
                     f"{_margin(r['forecast_margin'], r['dem_candidate'], r['rep_candidate'])} from "
                     f"{int(r['n_polls'])} poll{'s' if int(r['n_polls']) != 1 else ''}. Tipping point in "
                     f"{_pct(r['tipping_point_prob'])} of simulations.")
    paths = s.get('paths', {}).get('DEMOCRAT', {})
    from states import STATE_ABBREV
    name_of = {v: _title(k) for k, v in STATE_ABBREV.items()}
    swing = [r for r in paths.get('races', []) if 0.3 <= r['p'] <= 0.7 and not r['held']]
    if swing:
        label = lambda r: name_of.get(r['race'].split()[0], r['race']) + (" (an independent)" if '(I)' in r['race'] else "")
        lines += ["", "**The Democratic path.** In the simulations where Democrats win the Senate, they hold nearly "
                  "all of their own seats and then pick up enough of these Republican-held seats: "
                  + _list(label(r) for r in swing) + ". Each one goes their way only "
                  f"{min(r['p'] for r in swing):.0%} to {max(r['p'] for r in swing):.0%} of the time. Because one "
                  "polling miss tends to move all of them together, the Senate can break clearly in either direction."]
    return lines + [""]

def house_section(f):
    s = f['summary']['house']
    h = f['house']
    lo, hi = s['dem_seats_90pct']
    toss = h[(h['dem_win_prob'] > 0.35) & (h['dem_win_prob'] < 0.65)]
    tp = h.sort_values('tipping_point_prob', ascending=False).head(5)
    seat = lambda r: f"{_title(r['state'])}-{int(r['district'])}" if int(r['district']) else f"{_title(r['state'])} at-large"
    return [
        "## The House", "",
        "*[Image: house.png]*", "",
        f"Democrats have an **{_pct(s['dem_control_prob'])} chance** of winning the House. Our median is "
        f"{s['dem_seats_median']} Democratic seats, with a 90% range of {lo} to {hi}; 218 is a majority. "
        f"{len(toss)} districts are true toss-ups (35–65%).", "",
        "Most districts have few or no polls, so the House leans on each district's partisan lean, its "
        "incumbent's track record and the national environment. The districts most likely to deliver the 218th "
        "seat: " + _list(f"{seat(r)} ({r['dem_candidate']}, {_pct(r['dem_win_prob'])})" for _, r in tp.iterrows())
        + ".", ""]

def governor_section(f):
    s = f['summary']['governor']
    g = f['governor']
    lo, hi = s['dem_governorships_90pct']
    close = g[(g['dem_side_win_prob'] > 0.25) & (g['dem_side_win_prob'] < 0.75)].sort_values('dem_side_win_prob')
    flips_d = g[(g['incumbent_party'] == 'REPUBLICAN') & (g['dem_side_win_prob'] > 0.5)]
    flips_r = g[(g['incumbent_party'] == 'DEMOCRAT') & (g['dem_side_win_prob'] < 0.5)]
    lines = [
        "## Governors", "",
        f"36 governorships are up. Democrats are projected to win a median of **{s['dem_governorships_median']}** "
        f"(90% range {lo} to {hi}).", ""]
    if len(flips_d) or len(flips_r):
        parts = []
        if len(flips_d):
            parts.append("Democrats lead in Republican-held " + _list(_title(x) for x in flips_d['state']))
        if len(flips_r):
            parts.append("Republicans lead in Democratic-held " + _list(_title(x) for x in flips_r['state']))
        lines += ["; ".join(parts) + ".", ""]
    if len(close):
        lines += ["The closest races:", ""]
        lines += [f"- **{_title(r['state'])}:** {r['dem_candidate']} vs. {r['rep_candidate']}, "
                  f"{_last(r['dem_candidate'])} {_pct(r['dem_side_win_prob'])} "
                  f"({_margin(r['forecast_margin'], r['dem_candidate'], r['rep_candidate'])}"
                  + (", no polls yet" if int(r['n_polls']) == 0 else f", {int(r['n_polls'])} polls") + ")"
                  for _, r in close.iterrows()]
        lines.append("")
    return lines

def post_markdown(f, published, summary):
    sen, house, gov = f['summary']['senate'], f['summary']['house'], f['summary']['governor']
    day = W.pd.Timestamp(published)
    p = np.concatenate([f['senate']['dem_side_win_prob'].values, f['governor']['dem_side_win_prob'].values])
    upsets = float(np.minimum(p, 1 - p).sum())
    when = f"{day:%B} {day.day}, {day.year}"
    lines = [
        f"# {MODEL_NAME}: the 2026 midterms, {f['days_to_election']} days out", "",
        f"*{TAGLINE}*", "",
        f"Today we're publishing {MODEL_NAME}: a probability for every Senate race, every governor's race and all "
        f"435 House districts on the November 3 ballot, updated every morning until Election Day. Here's where it "
        f"stands as of {when}:", "",
        f"- **Senate:** Democrats {_pct(sen['dem_control_prob'])}, Republicans {_pct(sen['rep_control_prob'])}.",
        f"- **House:** Democrats {_pct(house['dem_control_prob'])}.",
        f"- **Governors:** Democrats win a median of {gov['dem_governorships_median']} of 36.", "",
        f"The national environment favors Democrats. Generic-ballot polls average D+{summary['generic_ballot_dem_margin']:.1f}. "
        f"Those polls have overstated Democrats in most recent cycles, so the forecast reads that as about "
        f"D+{summary['national_environment_used']:.1f}. That's enough to make the House likely. The Senate is closer, "
        "because Democrats' path runs through Republican-held seats, most of them in states Trump carried in 2024.", "",
    ]
    lines += senate_section(f) + house_section(f) + governor_section(f)
    lines += [
        "## How to read these numbers", "",
        "A 70% chance isn't a prediction that a candidate wins. It means that if we ran this election ten times, "
        f"they'd lose about three. Across all {len(p)} Senate and governor races, the forecast expects its favorites to lose "
        f"about {upsets:.0f}. If none lose, the forecast was too cautious; if many more lose, it was too sure.", "",
        "Polls carry most of the weight wherever there are enough of them. We correct each pollster for its track "
        "record and lean, and we use no prediction-market data. Each forecast then simulates the election 20,000 "
        "times, with one shared polling miss in each run, because polls tend to miss in the same direction "
        "everywhere at once. The full method is in [How the Ledger Forecast Works](LINK).", "",
        "## What's next", "",
        "- **Every morning:** the forecast updates with new polls. Each day's version is saved and dated.",
        "- **Every week:** a short \"What moved\" post with the races that changed and why.",
        "- **After November 3:** we grade every forecast against the official results, misses included.", "",
        "---", "",
        f"*{MODEL_NAME}, model {MODEL_VERSION}, published {when}. A forecast, not a call. {PUBLICATION}.*",
    ]
    return '\n'.join(lines)

def path_to_51_png(f, dark=True):
    """Every Senate race in a row, most Democratic first, shaded by win chance, with the
    line at the 51st Democratic seat."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch
    from states import STATE_ABBREV
    bg, ink, ink_2 = ('#15202b', '#ffffff', '#8b98a5') if dark else ('#ffffff', '#111111', '#55534f')
    cmap = LinearSegmentedColormap.from_list('p', ['#e34948', '#c9c8c2', '#2a78d6'])
    sen = f['senate'].sort_values('dem_side_win_prob', ascending=False).reset_index(drop=True)
    s = f['summary']['senate']
    need = 51 - s['holdovers']['DEMOCRAT']
    dem_rows = np.flatnonzero(sen['dem_side_party'].values == 'DEMOCRAT')
    line_at = int(dem_rows[need - 1]) + 1 if 0 < need <= len(dem_rows) else None
    n = len(sen)
    fig = plt.figure(figsize=(12, 4.2), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200); ax.set_ylim(0, 420); ax.axis('off')
    ax.text(40, 370, MODEL_NAME.upper(), fontsize=13, color=ink_2, fontweight='bold')
    ax.text(40, 320, "The path to 51", fontsize=28, color=ink, fontweight='bold')
    ax.text(40, 285, f"All {n} Senate races, most Democratic on the left, shaded by win chance. "
                     f"Democrats need {need} of them for 51 seats.", fontsize=13, color=ink_2)
    x0, w, gap, y0, h = 40, (1120 - (n - 1) * 3) / n, 3, 150, 90
    for i, r in sen.iterrows():
        p = float(r['dem_side_win_prob'])
        x = x0 + i * (w + gap)
        ind = r['dem_side_party'] == 'INDEPENDENT'
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle='round,pad=0,rounding_size=3', fc=cmap(p), lw=0,
                                    ec=(1, 1, 1, 0.45) if ind else 'none', hatch='//' if ind else None))
        ax.text(x + w / 2, y0 + h / 2, STATE_ABBREV[r['state']] + ('*' if ind else ''), fontsize=10, ha='center',
                va='center', color='#111111' if 0.25 < p < 0.75 else 'white', rotation=90 if w < 30 else 0)
    if line_at:
        lx = x0 + line_at * (w + gap) - gap / 2
        ax.plot([lx, lx], [y0 - 18, y0 + h + 18], color=ink, lw=2, ls=(0, (2, 2)))
        ax.text(lx, y0 + h + 26, "51st Democratic seat", fontsize=12, color=ink, ha='center')
    ax.text(40, 105, "Safer Democratic", fontsize=12, color=ink_2)
    ax.text(1160, 105, "Safer Republican", fontsize=12, color=ink_2, ha='right')
    foot = (f"{MODEL_NAME} · model {MODEL_VERSION} · as of {W.pd.Timestamp(f['as_of']):%b %d, %Y} · "
            f"Democrats {s['dem_control_prob']:.0%} to win the Senate"
            + (" · * independent, not counted toward 51" if (sen['dem_side_party'] == 'INDEPENDENT').any() else ""))
    ax.text(40, 40, foot, fontsize=11, color=ink_2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=bg)
    plt.close(fig)
    return buf.getvalue()

def main():
    published, summary = latest_snapshot()
    if summary['model_version'] != MODEL_VERSION:
        print(f"Note: the latest published forecast ({published}) is model {summary['model_version']}, but the code is "
              f"{MODEL_VERSION}. Run python src/daily_update.py first so the post matches a published snapshot.")
    f = build_forecast(summary['generic_ballot_dem_margin'], inputs=load_inputs())
    out = os.path.join(BASE_DIR, '..', 'forecasts', 'substack', f'forecast_{published}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'post.md'), 'w', encoding='utf-8') as fh:
        fh.write(post_markdown(f, published, summary) + '\n')
    with open(os.path.join(out, 'path_to_51.png'), 'wb') as fh:
        fh.write(path_to_51_png(f))
    import dashboard_2026 as dash
    for office in ('senate', 'house'):
        try:
            with open(os.path.join(out, f'{office}.png'), 'wb') as fh:
                fh.write(dash.share_image(office, f, MODEL_VERSION))
        except Exception as e:     # kaleido needs a local Chrome
            print(f"Couldn't draw {office}.png ({e}); build it from the {office.title()} tab instead.")
    print(f"Wrote {os.path.abspath(out)}")

if __name__ == '__main__':
    main()
