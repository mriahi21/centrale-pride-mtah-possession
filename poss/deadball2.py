"""v2 stoppage / restart detection (profile "v2"). The frozen v1 module (deadball.py) is untouched.

Differences from v1, all motivated by false splits being the costliest error for spellF1:
  * CORROBORATION: weak cues (stationary ball, quiet players, set-piece tag) can never enter D alone;
    two weak cues at full strength, or one strong cue (reliable out-of-bounds), are needed.
  * RELIABILITY: every cue is weighted by the fraction of *observed* (not interpolated) ball frames
    behind it and by track continuity, so occlusions and glitches cannot create or destroy a stoppage.
  * LIVE EVIDENCE: sustained fast ball or controlled dribbling subtracts from the score and preserves
    genuine short live stretches between two stoppages (v1's clean_runs filled them in blindly).
  * BOUNDARIES: a stoppage starts at the first reliable out-of-bounds frame and ends when the ball
    actually moves away from its resting spot (restart), not where a smoothed score crosses a threshold.
A missing ball is never evidence of a stoppage.
"""
import numpy as np
import pandas as pd

import poss  # noqa: F401
from poss.model import alive_segments

W_OOB, W_STILL, W_QUIET, W_TAG, W_LIVE = 2.0, 1.0, 1.0, 0.8, 2.5
W_WHISTLE, W_VQUIET = 0.85, 0.85      # v3 extras: weak, corroborating cues only (each below ENTER on its own)
ENTER, EXIT = 1.65, 1.0
MIN_CORE = 40
MIN_DEAD_STRONG, MIN_DEAD_WEAK = 40, 60
MAX_GAP_MERGE = 25
LIVE_KEEP = 0.35
STILL_SPEED, FAST_SPEED, MOVED_M = 0.6, 6.0, 0.8


def _roll_known(x, known, win, min_known, full_known=None):
    """Mean of x over known frames in a centred window, scaled by reliability (known count)."""
    s = pd.Series(np.where(known, x, np.nan).astype(float))
    m = s.rolling(win, center=True, min_periods=min_known).mean().to_numpy()
    cnt = pd.Series(known.astype(float)).rolling(win, center=True, min_periods=1).sum().to_numpy()
    rel = np.clip(cnt / float(full_known or win * 0.5), 0.0, 1.0)
    return np.nan_to_num(m, nan=0.0) * rel


def activity_separation(log_act, iters=40):
    """Unsupervised informativeness of the 'players are quiet' cue in THIS half: separation of a 2-component
    mixture on log activity (0 if too little data). Large = clearly bimodal (live vs stopped)."""
    x = log_act[np.isfinite(log_act)]
    if len(x) < 5000:
        return 0.0
    xs = x[:: max(1, len(x) // 40000)]
    mu = np.array([np.percentile(xs, 25), np.percentile(xs, 75)])
    sd = np.array([xs.std() / 2 + 1e-3] * 2)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        pdf = np.stack([w[k] / sd[k] * np.exp(-0.5 * ((xs - mu[k]) / sd[k]) ** 2) for k in range(2)])
        r = pdf / (pdf.sum(0, keepdims=True) + 1e-12)
        nk = r.sum(1) + 1e-9
        w = nk / nk.sum()
        mu = (r * xs).sum(1) / nk
        sd = np.sqrt((r * (xs[None] - mu[:, None]) ** 2).sum(1) / nk) + 1e-3
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    if min(w) < 0.08:
        return 0.0
    return float((mu[hi] - mu[lo]) / np.sqrt(0.5 * (sd[lo] ** 2 + sd[hi] ** 2)))


def bridge_static(ft, max_gap=250, still_v=STILL_SPEED, same_spot_m=1.5):
    """Mark occluded frames as stationary (and out-of-bounds) when the ball was last seen AND next seen
    at the same spot, both times slow. Trajectory continuity, not guesswork: any movement in between
    would have to return exactly to the spot. Returns (bridged_mask, bridged_oob)."""
    n = ft["n"]
    obs = ft["ball_conf"] >= 0.99
    v = ft["ball_speed"]
    good = np.flatnonzero(obs & np.isfinite(v))
    bridged = np.zeros(n, dtype=bool)
    b_oob = np.zeros(n, dtype=bool)
    if len(good) < 2:
        return bridged, b_oob
    i, j = good[:-1], good[1:]
    gap = j - i
    cand = np.flatnonzero((gap > 1) & (gap <= max_gap))
    ball, oob = ft["ball"], ft["oob"]
    for k in cand:
        a, b = i[k], j[k]
        if v[a] < still_v and v[b] < still_v and np.hypot(*(ball[a] - ball[b])) < same_spot_m:
            bridged[a + 1:b] = True
            if oob[a] and oob[b]:
                b_oob[a + 1:b] = True
    return bridged, b_oob


def cues(ft, tag_dead=None, extras=False):
    n = ft["n"]
    obs = ft["ball_conf"] >= 0.99
    v = ft["ball_speed"]
    vfin = np.isfinite(v)
    bridged, b_oob = bridge_static(ft)
    known_v = (obs & vfin) | bridged
    known_o = obs | bridged

    oob_flag = ft["oob"] | b_oob
    oob = _roll_known(oob_flag.astype(float), known_o, 51, 8, full_known=16)
    still_flag = (np.nan_to_num(v, nan=9.0) < STILL_SPEED) | bridged
    still = _roll_known(still_flag.astype(float), known_v, 51, 12, full_known=20)

    act = ft["activity"]
    fin = act[np.isfinite(act)]
    if len(fin) > 1000:
        q5, q35 = np.percentile(fin, [5, 35])
        quiet = np.nan_to_num(np.clip((q35 - act) / max(q35 - q5, 1e-3), 0.0, 1.0), nan=0.0)
    else:
        quiet = np.zeros(n)

    obs_v = obs & vfin
    fast = _roll_known((np.nan_to_num(v, nan=0.0) > FAST_SPEED).astype(float), obs_v, 51, 5, full_known=15)
    near = np.minimum(ft["dH"], ft["dA"])
    vv = np.nan_to_num(v, nan=0.0)
    dribble = ((vv >= 1.0) & (vv <= 7.0) & (near < 2.0)).astype(float)
    drib = _roll_known(dribble, obs_v, 41, 10, full_known=25)
    live = np.clip(np.maximum(fast, 0.8 * drib), 0, 1)
    tag = np.zeros(n) if tag_dead is None else np.asarray(tag_dead, dtype=float)
    sep = activity_separation(np.log(np.maximum(act, 0.05)))
    # Data-adaptive weight: a clearly bimodal activity distribution makes 'quiet' trustworthy on its own
    # (weight up to 1.8 >= ENTER); otherwise it stays a corroborating cue only (weight 1.0 < ENTER).
    w_quiet = W_QUIET + 0.8 * float(np.clip((sep - 1.5) / 1.0, 0.0, 1.0))
    out = {"oob": oob, "still": still, "quiet": quiet * w_quiet / W_QUIET, "tag": tag, "live": live,
           "_sep": sep, "_w_quiet": w_quiet}
    if extras:
        if ft.get("whistle") is not None:
            out["whistle"] = np.asarray(ft["whistle"], dtype=float)
        if ft.get("vquiet") is not None:
            vq = np.asarray(ft["vquiet"], dtype=float)
            ok = np.ones(n, dtype=bool) if ft.get("hom_ok") is None else np.asarray(ft["hom_ok"], dtype=bool)
            fv = vq[np.isfinite(vq) & ok]
            if len(fv) > 1000:
                v5, v35 = np.percentile(fv, [5, 35])
                out["vquiet"] = np.nan_to_num(np.clip((v35 - vq) / max(v35 - v5, 1e-6), 0.0, 1.0), nan=0.0) * ok
    return out


def score(c):
    c = {k: v for k, v in c.items() if not k.startswith("_")}
    z = W_OOB * c["oob"] + W_STILL * c["still"] + W_QUIET * c["quiet"] + W_TAG * c["tag"] - W_LIVE * c["live"]
    if "whistle" in c:
        z = z + W_WHISTLE * c["whistle"]
    if "vquiet" in c:
        z = z + W_VQUIET * c["vquiet"]
    return z


def _grow(z, enter=ENTER, exit_=EXIT, min_core=MIN_CORE):
    zs = pd.Series(z).rolling(13, center=True, min_periods=1).mean().to_numpy()
    runs = []
    for a, b in alive_segments(zs > exit_):
        if int((zs[a:b] > enter).sum()) >= min_core:
            runs.append([a, b])
    return runs


def _first_oob(ft, a, b):
    lo, hi = max(a - 50, 0), min(a + 25, ft["n"])
    idx = np.flatnonzero(ft["oob"][lo:hi] & (ft["ball_conf"][lo:hi] >= 0.99))
    return lo + int(idx[0]) if len(idx) else None


def _restart_frame(ft, b):
    """First frame near b at which the ball genuinely leaves its resting spot (sustained), else None."""
    n = ft["n"]
    ball, obs = ft["ball"], ft["ball_conf"] >= 0.99
    lo, hi = max(b - 50, 0), max(b - 10, 1)
    idx = np.flatnonzero(obs[lo:hi])
    if len(idx) < 5:
        return None
    rest = np.median(ball[lo:hi][idx], axis=0)
    for t in range(max(b - 75, 0), min(b + 75, n - 6)):
        if not obs[t]:
            continue
        w = np.hypot(*(ball[t:t + 5] - rest).T)
        if np.isfinite(w).all() and (w > MOVED_M).all():
            return t
    return None


def dead_mask(ft, kick_rel, tag_dead=None, log=print, extras=False):
    n = ft["n"]
    c = cues(ft, tag_dead, extras)
    z = score(c)
    runs = _grow(z)
    refined = []
    for a, b in runs:
        strong = c["oob"][a:b].max() > 0.6
        if strong:
            f = _first_oob(ft, a, b)
            if f is not None:
                a = min(a, f)
        r = _restart_frame(ft, b)
        if r is not None and r > a + 10:
            b = r
        refined.append([a, b, strong])
    # merge across short gaps unless a genuine live stretch sits in between
    merged = []
    for a, b, s in refined:
        if merged and a - merged[-1][1] < MAX_GAP_MERGE:
            g0, g1 = merged[-1][1], a
            if g1 - g0 < 12 or (c["live"][g0:g1].mean() < LIVE_KEEP):
                merged[-1][1] = max(b, merged[-1][1])
                merged[-1][2] = merged[-1][2] or s
                continue
        merged.append([a, b, s])
    mask = np.zeros(n, dtype=bool)
    for a, b, s in merged:
        if (b - a) >= (MIN_DEAD_STRONG if s else MIN_DEAD_WEAK):
            mask[max(a, 0):min(b, n)] = True
    mask[:kick_rel] = True
    share = float(mask[kick_rel:].mean()) if n > kick_rel else 0.0
    flags = {"profile": "v3" if extras else "v2", "extras": sorted(k for k in c if k in ("whistle", "vquiet")), "activity_sep": round(float(c["_sep"]), 2), "w_quiet": round(float(c["_w_quiet"]), 2), "dead_share_after_kickoff": round(share, 3),
             "n_dead_runs": len(alive_segments(mask[kick_rel:]))}
    log("[dead] {}".format(flags))
    return mask, flags
