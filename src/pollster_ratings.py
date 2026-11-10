"""
Pollster ratings computed from historical polls with known results (roadmap Phase 1).

Source: FiveThirtyEight's public pollster-ratings archive (raw_polls.csv), which has
every poll in the final ~3 weeks of US general elections 1998-2022 alongside the
actual result. 538's org was shut down in 2025, but the GitHub archive is still public
and is the best free record of polls vs. outcomes. Their own published ratings are kept
only as a cross-check; the ratings used by the forecast are computed here.

Two numbers per pollster, both measured against *other pollsters in the same race*.
That way a pollster isn't punished for a year when everyone missed (2016, 2020) and
isn't rewarded for only polling easy races:

  plus_minus    How much bigger (+) or smaller (-) the pollster's absolute error was
                than its peers' in the same race. Negative is better.
  house_effect  How far the pollster's margin sat to the Democratic (+) or
                Republican (-) side of its peers in the same race. Used to correct
                that pollster's 2026 polls before averaging.

Both are shrunk toward zero by SHRINKAGE_POLLS phantom average polls, so a pollster
with 3 polls can't earn an extreme rating by luck.

Run as a script to rebuild the ratings and print the out-of-sample grading:
    python src/pollster_ratings.py
"""
import os
import re
import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'polling_data')
ARCHIVE_CSV = os.path.join(DATA_DIR, '538_raw_polls.csv')
ARCHIVE_538_RATINGS_CSV = os.path.join(DATA_DIR, '538_pollster_ratings.csv')
RATINGS_CSV = os.path.join(DATA_DIR, 'pollster_ratings.csv')

ARCHIVE_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv"
ARCHIVE_538_RATINGS_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/pollster-ratings-combined.csv"

GENERAL_ELECTION_TYPES = ['Pres-G', 'Sen-G', 'Gov-G', 'House-G', 'House-G-US']
SHRINKAGE_POLLS = 15

def _download(url, path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)

HISTORY_2024_CSV = os.path.join(DATA_DIR, 'history_2024.csv')

def load_archive(refresh=False, include_2024=True):
    """
    Historical polls with results: FiveThirtyEight's archive (1998-2022, cached locally,
    frozen) plus 2024 from Wikipedia's race pages (history_2024.py), in the same format.
    """
    if refresh or not os.path.exists(ARCHIVE_CSV):
        _download(ARCHIVE_URL, ARCHIVE_CSV)
    if refresh or not os.path.exists(ARCHIVE_538_RATINGS_CSV):
        _download(ARCHIVE_538_RATINGS_URL, ARCHIVE_538_RATINGS_CSV)
    archive = pd.read_csv(ARCHIVE_CSV, low_memory=False)
    if include_2024 and os.path.exists(HISTORY_2024_CSV):
        archive = pd.concat([archive, pd.read_csv(HISTORY_2024_CSV, low_memory=False)], ignore_index=True)
    return archive

HERDING_CACHE = os.path.join(DATA_DIR, '538_herding_cache.csv')

def add_herding(polls):
    """
    Per-poll herding distances (see compute_ratings). They depend only on other polls
    in the same race, and the archive is frozen, so they're computed once and cached.
    """
    out = polls.copy()
    key = ['poll_id', 'question_id']
    cache = pd.read_csv(HERDING_CACHE) if os.path.exists(HERDING_CACHE) else pd.DataFrame(columns=key + ['dist', 'dist_floor'])
    done = set(zip(cache['poll_id'], cache['question_id']))
    todo_races = out.loc[[(p, q) not in done for p, q in zip(out['poll_id'], out['question_id'])], 'race'].unique()
    if len(todo_races):                       # only races with polls not yet in the cache
        sub = out[out['race'].isin(todo_races)].copy()
        sub['dist'], sub['dist_floor'] = _herding(sub)
        cache = pd.concat([cache[~cache.set_index(key).index.isin(sub.set_index(key).index)],
                           sub[key + ['dist', 'dist_floor']]], ignore_index=True)
        cache.to_csv(HERDING_CACHE, index=False)
    merged = out.merge(cache, on=key, how='left')
    merged.index = out.index
    return merged

def prepare_polls(raw):
    """
    General-election D-vs-R polls with Dem-positive poll and actual margins.
    Pollsters are keyed by 538's pollster_rating_id, which already merges name variants.
    """
    df = raw[raw['type_simple'].isin(GENERAL_ELECTION_TYPES)].copy()
    d_first = (df['cand1_party'] == 'DEM') & (df['cand2_party'] == 'REP')
    r_first = (df['cand1_party'] == 'REP') & (df['cand2_party'] == 'DEM')
    df = df[d_first | r_first].copy()
    sign = np.where(d_first[d_first | r_first], 1.0, -1.0)
    df['poll_margin']   = df['margin_poll'] * sign
    df['actual_margin'] = df['margin_actual'] * sign
    df['error']     = df['poll_margin'] - df['actual_margin']   # + = poll too Democratic
    df['abs_error'] = df['error'].abs()
    df = df.dropna(subset=['error', 'pollster_rating_id'])
    df['pollster_rating_id'] = df['pollster_rating_id'].astype(int)
    return df

def _relative_to_peers(df):
    """
    For each poll, compare against the average of *other pollsters'* polls in the same
    race (leave-own-pollster-out, so a pollster isn't graded against itself). Polls in
    races with only one pollster have no peers and are dropped.
    """
    by_race = df.groupby('race').agg(sum_err=('error', 'sum'), sum_abs=('abs_error', 'sum'), n=('error', 'size'))
    by_rp = df.groupby(['race', 'pollster_rating_id']).agg(
        sum_err=('error', 'sum'), sum_abs=('abs_error', 'sum'), n=('error', 'size'))
    peers = by_rp.join(by_race, rsuffix='_race')
    peers['peer_n'] = peers['n_race'] - peers['n']
    peers = peers[peers['peer_n'] > 0]
    peers['peer_err'] = (peers['sum_err_race'] - peers['sum_err']) / peers['peer_n']
    peers['peer_abs'] = (peers['sum_abs_race'] - peers['sum_abs']) / peers['peer_n']

    out = df.join(peers[['peer_err', 'peer_abs']], on=['race', 'pollster_rating_id'], how='inner')
    out['plus_minus']     = out['abs_error'] - out['peer_abs']
    out['relative_error'] = out['error'] - out['peer_err']
    return out

# ── FiveThirtyEight-style Predictive Plus-Minus ────────────────────────────────
# Follows "How Our Pollster Ratings Work" (FiveThirtyEight, March 2023):
#   simple plus-minus    error vs. the error expected from race type, sample size,
#                        days to election, and how crowded the field was (regression)
#   relative plus-minus  error vs. other pollsters in the same race
#   advanced plus-minus  relative weighted by n (other firms in the race) + simple
#                        weighted by 3; recent cycles weigh more (0.93 per year)
#   herding penalty      half the shortfall of a pollster's distance from the prior
#                        polling average below what sampling error alone would produce
#   predictive plus-minus = (max(-2, APM + herding) * n_disc + prior * 18) / (18 + n_disc)
#                        prior = 0.66 - 0.57 * quality + min(18, n_disc) * -0.03,
#                        quality = 1 for AAPOR Transparency Initiative / Roper members
# Simplifications: no strength-of-schedule iteration; the field-crowding adjustment
# enters through the expected-error regression rather than as a separate step.
RECENCY_DECAY = 0.93
PRIOR_POLLS = 18

def _expected_error(df):
    """Regression of absolute error on race type, sampling error, sqrt(days), and the
    number of other firms polling the race. Returns predicted absolute error per poll."""
    n = df['samplesize'].fillna(600).clip(lower=100)
    X = pd.get_dummies(df['type_simple'], prefix='t', drop_first=True, dtype=float)
    X['sampling'] = 1 / np.sqrt(n)
    X['days'] = np.sqrt(df['time_to_election'].clip(lower=0))
    X['crowd'] = np.log1p(df['other_firms'])
    X.insert(0, 'const', 1.0)
    coef, *_ = np.linalg.lstsq(X.values, df['abs_error'].values, rcond=None)
    return X.values @ coef

def _herding(df):
    """Per poll: distance from the average of earlier (3+ days) nonpartisan polls of the
    same race by other firms, and the distance sampling error alone would produce."""
    dist = pd.Series(np.nan, index=df.index)
    floor = pd.Series(np.nan, index=df.index)
    df = df.assign(_date=pd.to_datetime(df['polldate']))
    for _, g in df.groupby('race'):
        if g['pollster_rating_id'].nunique() < 2:
            continue
        g = g.sort_values('_date')
        for i, r in g.iterrows():
            prev = g[(g['_date'] <= r['_date'] - pd.Timedelta(days=3)) &
                     (g['pollster_rating_id'] != r['pollster_rating_id']) & g['partisan'].isna()]
            if prev.empty:
                continue
            prev = prev.groupby('pollster_rating_id').tail(1)          # latest per firm
            dist[i] = abs(r['poll_margin'] - prev['poll_margin'].mean())
            n_self = r['samplesize'] if pd.notna(r['samplesize']) else 600
            sd_poll = 100 / np.sqrt(max(n_self, 100))
            sd_avg = 100 / np.sqrt(prev['samplesize'].fillna(600).clip(lower=100).sum())
            floor[i] = np.hypot(sd_poll, sd_avg) * np.sqrt(2 / np.pi)
    return dist, floor

def compute_ratings(polls):
    """One row per pollster: Predictive Plus-Minus (stored as plus_minus, lower is
    better), shrunk house effect, and the intermediate 538-style components."""
    df = polls.copy()
    firms = df.groupby('race')['pollster_rating_id'].nunique()
    df['other_firms'] = df['race'].map(firms) - 1
    df['simple_pm'] = df['abs_error'] - _expected_error(df)

    rel = _relative_to_peers(df)
    df['relative_pm'] = rel['plus_minus'].reindex(df.index)
    df['relative_error'] = rel['relative_error'].reindex(df.index)
    n = df['other_firms']
    df['advanced_pm'] = np.where(df['relative_pm'].notna(),
                                 (n * df['relative_pm'].fillna(0) + 3 * df['simple_pm']) / (n + 3),
                                 df['simple_pm'])
    df['w'] = RECENCY_DECAY ** (df['cycle'].max() - df['cycle'])
    if 'dist' not in df:      # herding distances only depend on the same race, so callers
        df['dist'], df['dist_floor'] = _herding(df)   # rating many subsets can precompute them

    names = df.groupby('pollster_rating_id')['pollster'].agg(lambda s: s.value_counts().index[0])
    aliases = df.groupby('pollster_rating_id')['pollster'].agg(lambda s: '|'.join(sorted(set(s))))
    rows = {}
    for pid, g in df.groupby('pollster_rating_id'):
        w = g['w']
        n_disc = float(w.sum())
        apm = float(np.average(g['advanced_pm'], weights=w))
        h = g.dropna(subset=['dist'])
        herd = 0.5 * max(0.0, float(h['dist_floor'].mean() - h['dist'].mean())) if len(h) >= 3 else 0.0
        quality = 1.0 if g['aapor_roper'].fillna(False).astype(bool).any() else 0.0
        prior = 0.66 - 0.57 * quality - 0.03 * min(PRIOR_POLLS, n_disc)
        ppm = (max(-2.0, apm + herd) * n_disc + prior * PRIOR_POLLS) / (PRIOR_POLLS + n_disc)
        hr = g.dropna(subset=['relative_error'])
        rows[pid] = {
            'pollster': names[pid], 'aliases': aliases[pid],
            'n_polls': int(len(g)), 'n_races': int(g['race'].nunique()),
            'first_cycle': int(g['cycle'].min()), 'last_cycle': int(g['cycle'].max()),
            'mean_abs_error': float(g['abs_error'].mean()),
            'simple_plus_minus': float(np.average(g['simple_pm'], weights=w)),
            'advanced_plus_minus': apm, 'herding_penalty': herd, 'aapor_roper': bool(quality),
            'plus_minus': ppm,
            'house_effect': float(hr['relative_error'].sum() / (len(hr) + SHRINKAGE_POLLS)) if len(hr) else 0.0,
        }
    return pd.DataFrame.from_dict(rows, orient='index').rename_axis('pollster_rating_id').sort_values('plus_minus')

def compute_ratings_peer_only(polls):
    """Previous method (kept for comparison): plus-minus vs. same-race peers, shrunk to 0."""
    rel = _relative_to_peers(polls)
    names = polls.groupby('pollster_rating_id')['pollster'].agg(lambda s: s.value_counts().index[0])
    aliases = polls.groupby('pollster_rating_id')['pollster'].agg(lambda s: '|'.join(sorted(set(s))))
    g = rel.groupby('pollster_rating_id')
    ratings = pd.DataFrame({
        'pollster':        names,
        'aliases':         aliases,
        'n_polls':         g.size(),
        'n_races':         g['race'].nunique(),
        'first_cycle':     g['cycle'].min(),
        'last_cycle':      g['cycle'].max(),
        'mean_abs_error':  g['abs_error'].mean(),
        'plus_minus':      g['plus_minus'].sum() / (g.size() + SHRINKAGE_POLLS),
        'house_effect':    g['relative_error'].sum() / (g.size() + SHRINKAGE_POLLS),
    }).dropna(subset=['n_polls'])
    ratings['n_polls'] = ratings['n_polls'].astype(int)
    return ratings.sort_values('plus_minus')

def partisan_sponsor_effects(polls):
    """
    Average house effect of polls sponsored by a campaign or party, by sponsor party.
    Applied to 2026 polls Wikipedia marks (D)/(R) when the pollster itself has no
    track record.
    """
    rel = _relative_to_peers(polls)
    out = {}
    for party, key in [('DEM', 'DEMOCRAT'), ('REP', 'REPUBLICAN')]:
        sub = rel[rel['partisan'] == party]['relative_error']
        out[key] = float(sub.sum() / (len(sub) + SHRINKAGE_POLLS)) if len(sub) else 0.0
    return out

def grade_ratings(polls, split_cycle=2018, rate=None):
    """
    Out-of-sample check that the ratings mean something: rate pollsters using cycles up
    to `split_cycle`, then see whether those ratings predict who did well afterwards.
    """
    train = polls[polls['cycle'] <= split_cycle]
    test_rel = _relative_to_peers(polls[polls['cycle'] > split_cycle])
    train_r = (rate or compute_ratings)(train)

    test_by = test_rel.groupby('pollster_rating_id').agg(
        n=('plus_minus', 'size'), test_pm=('plus_minus', 'mean'), test_he=('relative_error', 'mean'))
    both = test_by.join(train_r[['plus_minus', 'house_effect', 'n_polls']], how='inner')
    both = both[(both['n'] >= 5) & (both['n_polls'] >= 10)]

    terciles = pd.qcut(both['plus_minus'], 3, labels=['best third', 'middle third', 'worst third'])
    tercile_err = test_rel.join(terciles.rename('tier'), on='pollster_rating_id').dropna(subset=['tier']) \
                          .groupby('tier', observed=True)['abs_error'].mean()
    w = both['n']
    def wcorr(a, b):
        a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
        return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))
    return {
        'split_cycle':            split_cycle,
        'n_pollsters_graded':     int(len(both)),
        'plus_minus_persistence': wcorr(both['plus_minus'], both['test_pm']),
        'house_effect_persistence': wcorr(both['house_effect'], both['test_he']),
        'test_abs_error_by_tier': tercile_err.round(2).to_dict(),
    }

def build_ratings():
    polls = prepare_polls(load_archive())
    ratings = compute_ratings(polls)
    ref = pd.read_csv(ARCHIVE_538_RATINGS_CSV)[['pollster_rating_id', 'numeric_grade', 'POLLSCORE']]
    ratings = ratings.join(ref.set_index('pollster_rating_id').rename(
        columns={'numeric_grade': 'grade_538', 'POLLSCORE': 'pollscore_538'}))
    ratings.round(3).to_csv(RATINGS_CSV, index_label='pollster_rating_id')
    return ratings, polls

def load_ratings():
    if not os.path.exists(RATINGS_CSV):
        build_ratings()
    return pd.read_csv(RATINGS_CSV)


# ── Matching 2026 pollster names (Wikipedia spelling) to rated pollsters ───────
_STOPWORDS = {'the', 'of', 'and', 'inc', 'llc', 'polling', 'poll', 'polls', 'research', 'group',
              'strategies', 'analytics', 'associates', 'survey', 'center', 'institute', 'company',
              'co', 'news', 'university', 'college', 'for', 'public', 'opinion', 'insights'}

def _tokens(name):
    name = re.sub(r'\[[^\]]*\]', '', str(name))          # footnotes like [37]
    name = re.sub(r'\((D|R|I|L)\)', '', name)            # sponsor party marker
    words = re.findall(r'[a-z0-9]+', name.lower())
    return {w for w in words if w not in _STOPWORDS}

def match_pollster(name, ratings):
    """
    Best-matching rated pollster for a free-text name, or None. Joint polls like
    'CBS News/YouGov' are split on '/' and the best-rated-history part wins. A part must
    share every distinctive word with a rated alias (or vice versa) to count, so
    'Emerson' matches 'Emerson College' but 'Data for Progress' won't match 'Progress Poll'.
    """
    best, best_n = None, -1
    parts = [p for p in re.split(r'[/&]| and ', str(name)) if _tokens(p)]
    for part in parts:
        pt = _tokens(part)
        for _, r in ratings.iterrows():
            for alias in str(r['aliases']).split('|'):
                at = _tokens(alias)
                if at and (pt <= at or at <= pt) and r['n_polls'] > best_n:
                    best, best_n = r, r['n_polls']
    return best

def sponsor_party(name):
    """Sponsor marker Wikipedia puts after a pollster: (D), (R), or (I) for a poll paid
    for by an independent candidate's campaign."""
    marks = set(re.findall(r'\((D|R|I)\)', str(name)))
    if len(marks) != 1:
        return None      # unmarked, or a bipartisan team like 'Beacon Research (D)/Shaw & Co. (R)'
    return {'D': 'DEMOCRAT', 'R': 'REPUBLICAN', 'I': 'INDEPENDENT'}[marks.pop()]


if __name__ == '__main__':
    ratings, polls = build_ratings()
    print(f"Rated {len(ratings)} pollsters from {len(polls)} polls "
          f"({polls['cycle'].min()}-{polls['cycle'].max()}).\n")
    cols = ['pollster', 'n_polls', 'mean_abs_error', 'plus_minus', 'house_effect', 'grade_538']
    big = ratings[ratings['n_polls'] >= 40]
    print("Most accurate (min 40 polls):\n", big.head(12)[cols].round(2).to_string(index=False))
    print("\nLeast accurate (min 40 polls):\n", big.tail(8)[cols].round(2).to_string(index=False))
    print("\nPartisan-sponsor house effects:", partisan_sponsor_effects(polls))
    print("\nOut-of-sample grading (538-style):", grade_ratings(polls))
    print("Out-of-sample grading (previous peer-only):", grade_ratings(polls, rate=compute_ratings_peer_only))
    print("\nAgreement with 538's own ratings (corr of plus_minus vs POLLSCORE, 40+ polls):",
          round(big[['plus_minus', 'pollscore_538']].dropna().corr().iloc[0, 1], 2))
