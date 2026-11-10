"""
Independence policy: no data from organizations with business ties to prediction or
betting markets. That covers poll sources, national averages, and the expert ratings
shown for comparison. As checked 2026-10-06:
  Decision Desk HQ (DDHQ)   runs live Polymarket and Kalshi odds displays in its coverage
  RealClearPolitics (RCP)   added Polymarket odds to its 2026 forecast maps (July 2026)
  Silver Bulletin           Nate Silver is a Polymarket adviser (since 2024)
  VoteHub                   data partnership with Kalshi (Midterms Hub, July 2026)
Check any new source for such ties before using it, and add it here if tied.
"""

MARKET_TIED = ('decision desk', 'ddhq', 'realclear', 'real clear', 'rcp', 'silver bulletin', 'silver',
               'votehub', 'polymarket', 'kalshi')

def is_market_tied(name):
    n = str(name).lower()
    return any(k in n for k in MARKET_TIED)
