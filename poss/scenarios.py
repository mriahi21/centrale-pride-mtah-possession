"""Feature-level synthetic scenarios (no match data) for regression-testing stoppage/restart logic.

Each scenario is a list of segments; `build` returns a half-feature dict `ft` (same keys the real
feature extractor produces) and the true dead mask. The cases deliberately go beyond the full-match
simulator's assumptions: stationary live balls, false out-of-bounds, occlusions, short live stretches
between stoppages, wrong tags, quiet-but-live build-up.
"""
import numpy as np
import pandas as pd

import poss  # noqa: F401
from poss import features

L, W = features.PITCH_L, features.PITCH_W


def build(segments, seed=0):
    """segments: list of dicts {kind: 'live'|'dead', n: frames, + options}. Returns (ft, dead_truth, info)."""
    rng = np.random.default_rng(seed)
    n = sum(s["n"] for s in segments)
    ball = np.zeros((n, 2))
    conf = np.ones(n, dtype=np.float32)
    oob = np.zeros(n, dtype=bool)
    act = np.zeros(n)
    dH = np.full(n, 30.0)
    dA = np.full(n, 30.0)
    dead = np.zeros(n, dtype=bool)
    tag = np.zeros(n)
    t = 0
    pos = np.array([L / 2, W / 2])
    for s in segments:
        m = s["n"]
        sl = slice(t, t + m)
        if s["kind"] == "live":
            k = 0
            while k < m:
                blk = min(50, m - k)
                carry = True if s.get("carry_only") else rng.random() < 0.5
                speed = rng.uniform(2, 4) if carry else rng.uniform(9, 13)
                ang = rng.uniform(0, 2 * np.pi)
                step = speed / 25.0 * np.array([np.cos(ang), np.sin(ang)])
                for j in range(blk):
                    nxt = pos + step
                    if not (5 < nxt[0] < L - 5):
                        step[0] *= -1
                    if not (5 < nxt[1] < W - 5):
                        step[1] *= -1
                    pos = pos + step
                    ball[t + k + j] = pos
                near, far = rng.uniform(0.6, 1.4), rng.uniform(6, 12)
                team_home = rng.random() < 0.5
                dH[t + k:t + k + blk] = near if (carry and team_home) else (far if carry else 7.0)
                dA[t + k:t + k + blk] = near if (carry and not team_home) else (far if carry else 8.0)
                k += blk
            act[sl] = s.get("activity", 2.5)
        else:
            spot = pos.copy()
            if s.get("oob"):
                spot = np.array([min(max(pos[0], 10), L - 10), -0.9])
                oob[sl] = True
            ball[sl] = spot
            pos = np.array([min(max(spot[0], 5), L - 5), min(max(spot[1], 5), W - 5)])
            act[sl] = s.get("activity", 0.7)
            dH[sl] = rng.uniform(4, 10)
            dA[sl] = rng.uniform(4, 10)
            dead[sl] = True
            if s.get("tag", True):
                tag[sl] = 1.0
        # modifiers (offsets are relative to segment start)
        for off, ln in s.get("still_live", []):
            a = t + off
            ball[a:a + ln] = ball[a]
            dH[a:a + ln], dA[a:a + ln] = 0.8, 9.0
        for off, ln in s.get("occlude", []):
            a = t + off
            conf[a:a + ln] = 0.0
            ball[a:a + ln] = np.nan
            dH[a:a + ln] = dA[a:a + ln] = np.inf
        for off, ln in s.get("false_oob", []):
            a = t + off
            oob[a:a + ln] = True
        for off, ln in s.get("false_tag", []):
            tag[t + off:t + off + ln] = 1.0
        t += m
    act = pd.Series(act).rolling(51, center=True, min_periods=1).median().to_numpy() + rng.normal(0, 0.08, n)
    ft = {"n": n, "ball": ball, "ball_conf": conf, "ball_obs": conf >= 0.99, "oob": oob,
          "ball_speed": features.ball_speed(ball), "activity": act.astype(np.float32),
          "dH": dH.astype(np.float32), "dA": dA.astype(np.float32)}
    ft["ball_speed"] = np.where(conf >= 0.99, ft["ball_speed"], np.nan).astype(np.float32)
    return ft, dead, tag


def runs(mask):
    a = np.concatenate(([0], mask.astype(np.int8), [0]))
    e = np.diff(a)
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def stoppage_metrics(pred, truth):
    """Frame F1 for D, false splits (predicted dead runs lying fully inside live play), missed
    stoppages (true runs >=50 frames with <50% overlap) and mean absolute restart-boundary error."""
    tp = int((pred & truth).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(truth.sum()), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    false_splits = sum(1 for a, b in runs(pred) if not truth[a:b].any())
    missed = sum(1 for a, b in runs(truth) if (b - a) >= 50 and pred[a:b].mean() < 0.5)
    errs = []
    for a, b in runs(truth):
        pr = [(x, y) for x, y in runs(pred) if x < b and y > a]
        if pr:
            errs.append(abs(pr[-1][1] - b))
    return {"d_f1": round(f1, 3), "false_splits": false_splits, "missed": missed,
            "restart_err": float(np.mean(errs)) if errs else float("nan")}


def scenario(name, seed=0):
    L_ = lambda n, **k: dict(kind="live", n=n, **k)   # noqa: E731
    D_ = lambda n, **k: dict(kind="dead", n=n, **k)   # noqa: E731
    base = [L_(400), D_(150, oob=True), L_(600), D_(200), L_(500), D_(150, oob=True), L_(400)]
    if name == "baseline":
        segs = base
    elif name == "stationary_live_ball":      # dribbler pauses the ball 2.4 s while everyone is active
        segs = [L_(400), D_(150, oob=True), L_(900, still_live=[(300, 60)]), D_(200), L_(400)]
    elif name == "false_oob":                  # glitchy out-of-bounds detections inside live play
        segs = [L_(400), D_(150, oob=True), L_(900, false_oob=[(200, 6), (500, 8)]), D_(200), L_(400)]
    elif name == "short_occlusion":            # ball lost for 1-2 s in live play
        segs = [L_(400), D_(150, oob=True), L_(900, occlude=[(100, 30), (400, 50)]), D_(200), L_(400)]
    elif name == "long_occlusion_live":        # ball lost for 6 s during active play
        segs = [L_(400), D_(150, oob=True), L_(900, occlude=[(200, 150)]), D_(200), L_(400)]
    elif name == "short_live_between_stops":   # genuine 2 s live stretch between two stoppages
        segs = [L_(400), D_(200, oob=True), L_(50), D_(200, oob=True), L_(400)]
    elif name == "wrong_tag_in_live_play":     # a set-piece tag where nothing stopped
        segs = [L_(400), D_(150, oob=True), L_(900, false_tag=[(200, 120)]), D_(200), L_(400)]
    elif name == "quiet_but_live_buildup":     # players slow, ball moving with control
        segs = [L_(400), D_(150, oob=True), L_(900, activity=0.8), D_(200), L_(400)]
    elif name == "slow_buildup_carry_only":    # slow build-up, ball carried (no fast passes), everyone walking
        segs = [L_(400), D_(150, oob=True), L_(900, activity=0.8, carry_only=True), D_(200), L_(400)]
    elif name == "occluded_quiet_live":        # slow live play while the ball is untracked for 12 s
        segs = [L_(400), D_(150, oob=True), L_(900, activity=0.8, carry_only=True, occlude=[(200, 300)]), D_(200), L_(400)]
    elif name == "stationary_quiet_live":      # keep-ball: slow, ball nearly still under a player's foot
        segs = [L_(400), D_(150, oob=True), L_(900, activity=0.8, carry_only=True, still_live=[(150, 70), (500, 70)]), D_(200), L_(400)]
    elif name == "dead_inside_pitch":          # free-kick style stoppage: ball inside the lines, tag present
        segs = [L_(400), D_(200, oob=False), L_(600), D_(200, oob=False), L_(400)]
    elif name == "dead_inside_pitch_no_tag":   # same, but the tagger missed it
        segs = [L_(400), D_(200, oob=False, tag=False), L_(600), D_(200, oob=False, tag=False), L_(400)]
    elif name == "stoppage_ball_occluded":     # real stoppage, ball not tracked (tag present)
        segs = [L_(400), D_(300, oob=False, occlude=[(0, 300)]), L_(500)]
    else:
        raise KeyError(name)
    return build(segs, seed)


SCENARIOS = ["baseline", "stationary_live_ball", "false_oob", "short_occlusion", "long_occlusion_live",
             "short_live_between_stops", "wrong_tag_in_live_play", "quiet_but_live_buildup",
             "slow_buildup_carry_only", "occluded_quiet_live", "stationary_quiet_live", "dead_inside_pitch", "dead_inside_pitch_no_tag", "stoppage_ball_occluded"]
