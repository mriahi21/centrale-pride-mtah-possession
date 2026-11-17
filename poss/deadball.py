"""Dead-ball (D) detection from ball-free and ball-based cues, with hysteresis.

Principles (see CHALLENGE.md): a missing ball is NOT a dead ball. Dead = a stoppage, not a possession.
Cues, each a soft log-odds-like term (weights are football logic, NOT tuned on any labels):
  * ball observed outside the lines                      -> strong dead
  * ball (observed) stationary for ~1.5 s                -> dead
  * players collectively slow (walking) vs. the half's own activity distribution
    (unsupervised 2-component mixture on log median speed, per half; disabled if not bimodal)
  * fast ball -> strongly alive
  * Smart Tagging set-piece window -> small dead bonus (it marks restarts; the interval is a phase, not a label)
Region growing with a high/low double threshold and minimum durations gives clean stoppage intervals.
"""
import numpy as np
import pandas as pd

import poss  # noqa: F401
from poss.model import alive_segments

FPS = 25
W_OOB, W_STILL, W_LOW, W_TAG, BIAS, W_FAST = 2.5, 1.2, 2.0, 0.8, -1.0, 3.0
HI, LO = 0.25, -0.1          # region-growing thresholds on smoothed score
MIN_CORE = 40                # frames above HI needed to accept a stoppage (1.6 s)
MIN_DEAD = 50                # shortest stoppage kept (2 s)
MIN_DEAD_TAG = 25            # ... when a set-piece tag backs it
MAX_ALIVE_GAP = 25           # alive gaps shorter than this between stoppages are filled
STILL_SPEED, FAST_SPEED = 0.6, 6.0


def _roll(x, win, minp):
    return pd.Series(x).rolling(win, center=True, min_periods=minp).mean().to_numpy()


def gmm_low_posterior(log_act, iters=40):
    """1-D two-component EM. Returns (p_low per frame in [0,1], ok flag). ok=False if not bimodal."""
    x = log_act[np.isfinite(log_act)]
    if len(x) < 5000:
        return np.full(len(log_act), 0.5), False
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
    sep = (mu[hi] - mu[lo]) / np.sqrt(0.5 * (sd[lo] ** 2 + sd[hi] ** 2))
    ok = bool(sep >= 1.5 and w[lo] >= 0.08 and w[hi] >= 0.08)
    with np.errstate(over="ignore", invalid="ignore"):
        pl = w[lo] / sd[lo] * np.exp(-0.5 * ((log_act - mu[lo]) / sd[lo]) ** 2)
        ph = w[hi] / sd[hi] * np.exp(-0.5 * ((log_act - mu[hi]) / sd[hi]) ** 2)
    p = pl / (pl + ph + 1e-12)
    return np.where(np.isfinite(log_act), p, 0.5), ok


def dead_score(ft, tag_dead=None):
    """Per-frame dead score z (>0 leans dead) and a small dict of component flags."""
    n = ft["n"]
    oob_known = ft["ball_conf"] >= 0.99
    oob = np.where(oob_known, ft["oob"].astype(float), np.nan)
    oob_frac = np.nan_to_num(_roll(oob, 25, 5), nan=0.0)

    v = ft["ball_speed"]
    known = np.isfinite(v) & (ft["ball_conf"] >= 0.99)
    still = np.where(known, (v < STILL_SPEED).astype(float), np.nan)
    still_frac = np.nan_to_num(_roll(still, 41, 10), nan=0.0)
    fast = np.where(known, (v > FAST_SPEED).astype(float), np.nan)
    fast_frac = np.nan_to_num(_roll(fast, 51, 5), nan=0.0)

    act = ft["activity"]
    p_low, ok = gmm_low_posterior(np.log(np.maximum(act, 0.05)))
    low = np.clip(np.log(np.clip(p_low, 1e-3, 1 - 1e-3) / (1 - np.clip(p_low, 1e-3, 1 - 1e-3))) / 4.0, -1, 1)
    if not ok:
        # Not bimodal: fall back to a rank-based score within the half (unsupervised, scale-free).
        # Frames at/below the half's 5th activity percentile -> +1, at the 35th -> 0, clearly above -> negative.
        fin = act[np.isfinite(act)]
        if len(fin) > 1000:
            q5, q35 = np.percentile(fin, [5, 35])
            low = np.nan_to_num(np.clip((q35 - act) / max(q35 - q5, 1e-3), -1, 1), nan=0.0) * 0.75
        else:
            low = np.zeros(n)

    z = BIAS + W_OOB * oob_frac + W_STILL * still_frac + W_LOW * low - W_FAST * fast_frac
    if tag_dead is not None:
        z = z + W_TAG * tag_dead
    return z, {"activity_bimodal": ok}


def grow_regions(z, hi=HI, lo=LO, min_core=MIN_CORE):
    """Contiguous regions where z > lo that contain >= min_core frames with z > hi."""
    zs = np.nan_to_num(_roll(z, 25, 1), nan=-5.0)
    keep = np.zeros(len(z), dtype=bool)
    for a, b in alive_segments(zs > lo):
        if int((zs[a:b] > hi).sum()) >= min_core:
            keep[a:b] = True
    return keep


def clean_runs(dead, tag_dead=None, min_dead=MIN_DEAD, min_dead_tag=MIN_DEAD_TAG, max_gap=MAX_ALIVE_GAP):
    out = dead.copy()
    # fill short alive gaps between stoppages
    for a, b in alive_segments(~out):
        if 0 < a and b < len(out) and (b - a) < max_gap:
            out[a:b] = True
    # drop very short stoppages (unless a set-piece tag overlaps)
    for a, b in alive_segments(out):
        backed = tag_dead is not None and tag_dead[a:b].max() > 0
        if (b - a) < (min_dead_tag if backed else min_dead):
            out[a:b] = False
    return out


def dead_mask(ft, kick_rel, tag_dead=None, log=print, share_range=(0.08, 0.55)):
    """Boolean D mask for one half (pre-kickoff frames are always dead)."""
    n = ft["n"]
    z, flags = dead_score(ft, tag_dead)
    hi, lo = HI, LO
    mask = None
    for attempt in range(4):
        mask = clean_runs(grow_regions(z, hi, lo), tag_dead)
        mask[:kick_rel] = True
        share = float(mask[kick_rel:].mean()) if n > kick_rel else 0.0
        if share > share_range[1]:
            hi, lo = hi + 0.3, lo + 0.3
        elif share < share_range[0] and attempt < 2:
            hi, lo = hi - 0.3, lo - 0.3
        else:
            break
    flags.update({"dead_share_after_kickoff": round(share, 3), "hi": round(hi, 2),
                  "n_dead_runs": len(alive_segments(mask[kick_rel:]))})
    log("[dead] {}".format(flags))
    return mask, flags
