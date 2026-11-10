"""Candidate-name matching that tolerates typos, middle initials, and nicknames."""
import re
from difflib import SequenceMatcher

def _norm(name):
    return ' '.join(re.findall(r'[a-z]+', str(name).lower()))

def _name_score(a, b):
    """Similarity of two candidate names: full-name ratio, boosted when last names match
    closely and first initials agree ('Dan S. Sullivan' vs 'Dan Sullivan')."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    full = SequenceMatcher(None, a, b).ratio()
    la, lb = a.split()[-1], b.split()[-1]
    last = SequenceMatcher(None, la, lb).ratio()
    if last >= 0.85 and a[0] == b[0]:
        return max(full, 0.9)
    if la == lb and len(la) >= 3:
        # Same surname, different first name: usually a nickname ('Bobby' vs 'Robert B.').
        # Scored just above the match threshold, so a closer full-name match still wins
        # when two candidates share a surname (Alaska's two Dan Sullivans).
        return max(full, 0.82)
    return full

def _match(answers, nominee, taken=None):
    best, best_score = None, 0.0
    for i, a in enumerate(answers):
        if i == taken:
            continue
        s = _name_score(a['choice'], nominee)
        if s > best_score:
            best, best_score = i, s
    return best if best_score >= 0.8 else None
