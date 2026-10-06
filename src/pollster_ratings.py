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

def load_archive(refresh=False):
    """Historical polls, cached locally (the archive is frozen, so one download is enough)."""
    if refresh or not os.path.exists(ARCHIVE_CSV):
        _download(ARCHIVE_URL, ARCHIVE_CSV)
    if refresh or not os.path.exists(ARCHIVE_538_RATINGS_CSV):
        _download(ARCHIVE_538_RATINGS_URL, ARCHIVE_538_RATINGS_CSV)
    return pd.read_csv(ARCHIVE_CSV, low_memory=False)

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

def compute_ratings(polls):
    """One row per pollster: shrunk plus_minus and house_effect, plus raw stats."""
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

def grade_ratings(polls, split_cycle=2018):
    """
    Out-of-sample check that the ratings mean something: rate pollsters using cycles up
    to `split_cycle`, then see whether those ratings predict who did well afterwards.
    """
    train = polls[polls['cycle'] <= split_cycle]
    test_rel = _relative_to_peers(polls[polls['cycle'] > split_cycle])
    train_r = compute_ratings(train)

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
    m = re.search(r'\((D|R|I)\)', str(name))
    return {'D': 'DEMOCRAT', 'R': 'REPUBLICAN', 'I': 'INDEPENDENT'}[m.group(1)] if m else None


if __name__ == '__main__':
    ratings, polls = build_ratings()
    print(f"Rated {len(ratings)} pollsters from {len(polls)} polls "
          f"({polls['cycle'].min()}-{polls['cycle'].max()}).\n")
    cols = ['pollster', 'n_polls', 'mean_abs_error', 'plus_minus', 'house_effect', 'grade_538']
    big = ratings[ratings['n_polls'] >= 40]
    print("Most accurate (min 40 polls):\n", big.head(12)[cols].round(2).to_string(index=False))
    print("\nLeast accurate (min 40 polls):\n", big.tail(8)[cols].round(2).to_string(index=False))
    print("\nPartisan-sponsor house effects:", partisan_sponsor_effects(polls))
    print("\nOut-of-sample grading:", grade_ratings(polls))
    print("\nAgreement with 538's own ratings (corr of plus_minus vs POLLSCORE, 40+ polls):",
          round(big[['plus_minus', 'pollscore_538']].dropna().corr().iloc[0, 1], 2))
