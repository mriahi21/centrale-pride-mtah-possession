"""Predeclared plausibility gates for a generated label array. NOT label-based, NOT tuned: generic football
sanity limits fixed before looking at any real output. A candidate that fails a gate is not submitted without
investigation. Passing proves nothing about accuracy; it only catches degenerate outputs.
"""
import numpy as np

import poss  # noqa: F401
from evaluation import possession_metrics as pm

GATES = {
    "dead_share_range": (0.05, 0.50),        # share of D after kickoff, per half
    "possessions_per_half": (80, 500),
    "mean_spell_s": (5.0, 60.0),
    "max_team_share": 0.85,                  # one team can't own >85% of live frames in a half
    "max_flicker_frac": 0.05,                # possessions shorter than 1 s
    "max_x_on_graded": 0,
}


def check(codes, blocks, kickoffs, fps=25):
    """Return (ok, [messages], per_half_stats). Stats are aggregates only."""
    msgs, stats = [], []
    for half, a, b in blocks:
        c = codes[a:b + 1]
        k = max(0, kickoffs.get(half, a) - a)
        post = c[k:]
        x = int((c == pm.X).sum())
        if x > GATES["max_x_on_graded"]:
            msgs.append("half {}: {} X on graded frames".format(half, x))
        share_d = float(np.mean(post == pm.D)) if len(post) else 0.0
        spells = pm.possession_spells(c, [(0, len(c))])
        n_alive = int(np.sum((c == pm.H) | (c == pm.A)))
        mean_s = n_alive / fps / max(len(spells), 1)
        home = int(np.sum(c == pm.H))
        team_share = max(home, n_alive - home) / float(max(n_alive, 1))
        flick = np.mean([(e - s) < fps for s, e, _ in spells]) if spells else 1.0
        lo, hi = GATES["dead_share_range"]
        if not (lo <= share_d <= hi):
            msgs.append("half {}: dead share {:.0%} outside [{:.0%},{:.0%}]".format(half, share_d, lo, hi))
        lo, hi = GATES["possessions_per_half"]
        if not (lo <= len(spells) <= hi):
            msgs.append("half {}: {} possessions outside [{},{}]".format(half, len(spells), lo, hi))
        lo, hi = GATES["mean_spell_s"]
        if not (lo <= mean_s <= hi):
            msgs.append("half {}: mean spell {:.1f}s outside [{},{}]".format(half, mean_s, lo, hi))
        if team_share > GATES["max_team_share"]:
            msgs.append("half {}: one team owns {:.0%} of live frames".format(half, team_share))
        if flick > GATES["max_flicker_frac"]:
            msgs.append("half {}: {:.0%} of possessions shorter than 1 s".format(half, flick))
        stats.append({"half": half, "dead_share": round(share_d, 3), "possessions": len(spells),
                      "mean_spell_s": round(mean_s, 1), "team_share": round(team_share, 2)})
    return (not msgs), msgs, stats
