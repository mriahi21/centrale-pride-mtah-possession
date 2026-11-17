"""v2 control evidence, switch penalty and restart priors (profile "v2").

Goals: an interpolated or glitchy ball must not decide ownership; a real reception/turnover with sustained
contact should move the boundary promptly; a loose ball, a deflection or a pass in flight should not.
Everything is data-adaptive or football-logic; nothing uses labels.
"""
import numpy as np
import pandas as pd

import poss  # noqa: F401
from poss.model import alive_segments

SIGMA_M = 2.0
V_FULL, V_ZERO = 4.0, 12.0
W_INTERP = 0.3            # ownership weight of an interpolated frame (short, plausible gaps only)
MAX_INTERP_GAP = 25       # frames (1 s) for ownership evidence; longer gaps are neutral
MAX_BRIDGE_SPEED = 30.0   # m/s implied across a bridged gap
EDGE_W = 0.7              # observed frames within 3 frames of an occlusion: touch/occlusion uncertainty
CONTACT_M, CONTACT_MARGIN_M, CONTACT_VMAX = 1.8, 1.0, 8.0
SUSTAIN_WIN, SUSTAIN_FRAC = 13, 0.8
CONTACT_BOOST = 0.5
DWELL_S = 0.6             # seconds of typical-strength opposing control to flip possession (plateau in synthetic lambda sweep; high lambda hurts under low ball coverage)
LAMBDA_MIN, LAMBDA_MAX = 4.0, 30.0
K_RESTART = 8.0           # first sustained contact after a restart -> initial ownership prior


def _rmean(x, win):
    return pd.Series(x.astype(float)).rolling(win, center=True, min_periods=1).mean().to_numpy()


def frame_weights(ft):
    """Per-frame reliability weight of the ball position for ownership decisions."""
    conf = ft["ball_conf"]
    n = ft["n"]
    ball = ft["ball"]
    w = np.where(conf >= 0.99, 1.0, 0.0)
    interp = (conf > 0) & (conf < 0.99)
    for a, b in alive_segments(interp):
        gap = b - a
        if gap > MAX_INTERP_GAP or a == 0 or b >= n:
            continue
        if conf[a - 1] < 0.99 or conf[b] < 0.99:
            continue
        speed = np.hypot(*(ball[b] - ball[a - 1])) / ((gap + 1) / 25.0)
        if np.isfinite(speed) and speed <= MAX_BRIDGE_SPEED:
            w[a:b] = W_INTERP
    unreliable = (conf < 0.99)
    near_gap = _rmean(unreliable, 7) > 0
    w = np.where((conf >= 0.99) & near_gap, w * EDGE_W, w)
    return w


def control_evidence(ft, sigma=SIGMA_M):
    """(u, aux): signed evidence (+ = home) and sustained-contact flags."""
    with np.errstate(over="ignore", invalid="ignore"):
        fh = np.nan_to_num(np.exp(-0.5 * (ft["dH"] / sigma) ** 2), nan=0.0, posinf=0.0)
        fa = np.nan_to_num(np.exp(-0.5 * (ft["dA"] / sigma) ** 2), nan=0.0, posinf=0.0)
    v = ft["ball_speed"]
    g = np.where(np.isnan(v), 0.7, np.clip(1.0 - 0.9 * (v - V_FULL) / (V_ZERO - V_FULL), 0.1, 1.0))
    w = frame_weights(ft)
    base = (fh - fa) * g * w

    obs_ok = (ft["ball_conf"] >= 0.99) & (np.isnan(v) | (v < CONTACT_VMAX))
    dH, dA = ft["dH"], ft["dA"]
    with np.errstate(invalid="ignore"):
        close_h = (dH <= CONTACT_M) & (dA - dH >= CONTACT_MARGIN_M) & obs_ok
        close_a = (dA <= CONTACT_M) & (dH - dA >= CONTACT_MARGIN_M) & obs_ok
    sust_h = _rmean(close_h, SUSTAIN_WIN) >= SUSTAIN_FRAC
    sust_a = _rmean(close_a, SUSTAIN_WIN) >= SUSTAIN_FRAC

    support = _rmean(np.abs(base) > 0.3, SUSTAIN_WIN)       # isolated touches are discounted
    u = base * (0.5 + 0.5 * np.clip(support / 0.7, 0.0, 1.0))
    u = u + CONTACT_BOOST * (sust_h.astype(float) - sust_a.astype(float)) * (w >= 0.99)
    return u.astype(np.float32), {"sust_h": sust_h, "sust_a": sust_a, "w": w}


def dwell_lambda(u_list, dwell_s=DWELL_S, fps=25):
    """Switch penalty = dwell time x typical strength of confident evidence (no match-wide count target)."""
    mags = np.concatenate([np.abs(u[np.abs(u) >= 0.2]) for u in u_list]) if u_list else np.zeros(0)
    m = float(np.median(mags)) if len(mags) >= 50 else 0.5
    return float(np.clip(dwell_s * fps * m, LAMBDA_MIN, LAMBDA_MAX))


def restart_priors(aux, segs, window=100):
    """Initial ownership prior per alive segment from the first sustained contact after the restart."""
    pri = []
    for a, b in segs:
        e = min(a + window, b)
        sh, sa = aux["sust_h"][a:e], aux["sust_a"][a:e]
        both = np.flatnonzero(sh | sa)
        if not len(both):
            pri.append(0.0)
        else:
            i = both[0]
            pri.append(K_RESTART if sh[i] else -K_RESTART)
    return pri
