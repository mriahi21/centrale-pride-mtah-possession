"""Automatic alignment of the broadcast tracking to the tactical timeline, and ball fusion.

The broadcast recording has its own timeline (cuts, replays, no halftime). Both ATDs report the ball in the
same normalised pitch coordinates, so alignment is done in tracking space: for every 10 s window of broadcast
ball positions we find, by FFT cross-correlation over the whole tactical ball track, the tactical start frame that
minimises the mean squared distance between co-observed ball positions. A window is accepted only if
  * enough co-observed frames, a small RMS distance, and a clear margin over the second-best candidate (so a
    stationary ball that matches anywhere is rejected), and
  * a neighbouring window agrees on the same offset (or the margin is very large).
Cuts, zooms and replays that do not match are simply left unmapped. Fusion then copies the broadcast ball into
tactical frames where the tactical ball is missing, only on accepted windows. Elsewhere the tactical evidence stands
alone. No video is decoded; no labels are used.
"""
import numpy as np

import poss  # noqa: F401

PITCH = np.array([105.3, 68.0])
WIN, STEP = 250, 125
MIN_OVERLAP = 80
MIN_SPREAD_M = 6.0         # co-observed ball positions must span >= this (std x + std y): a stationary ball matches anywhere
MAX_RMS_M = 4.0
RATIO = 2.0                 # second-best MSD must be >= RATIO**2 x best MSD
EXCLUDE = 100               # frames around the best candidate excluded when looking for the runner-up


def _metres(ball_norm, flip=False):
    b = np.asarray(ball_norm, dtype=np.float64).copy()
    if flip:
        b = 1.0 - b
    return b * PITCH


def align(tact_ball_norm, bc_ball_norm, flip=False, win=WIN, step=STEP, log=print):
    """Per-window alignment table. Returns dict of arrays (one entry per window) + parameters."""
    T = _metres(tact_ball_norm)
    B = _metres(bc_ball_norm, flip)
    Nt, Nb = len(T), len(B)
    mt = np.isfinite(T[:, 0])
    Tz = np.where(mt[:, None], T, 0.0)
    nfft = 1 << int(np.ceil(np.log2(Nt + win)))
    V0 = np.fft.rfft(mt.astype(np.float64), nfft)
    Vx = np.fft.rfft(Tz[:, 0], nfft)
    Vy = np.fft.rfft(Tz[:, 1], nfft)
    V3 = np.fft.rfft((Tz ** 2).sum(1), nfft)
    starts = np.arange(0, max(Nb - win, 0) + 1, step)
    out = {k: np.full(len(starts), np.nan) for k in ("d", "rms", "rms2", "count", "spread")}
    for w, b0 in enumerate(starts):
        Bw = B[b0:b0 + win]
        mb = np.isfinite(Bw[:, 0])
        if mb.sum() < MIN_OVERLAP:
            continue
        Bz = np.where(mb[:, None], Bw, 0.0)
        A0 = np.fft.rfft(mb * (Bz ** 2).sum(1), nfft)
        Ax = np.fft.rfft(Bz[:, 0], nfft)
        Ay = np.fft.rfft(Bz[:, 1], nfft)
        A3 = np.fft.rfft(mb.astype(np.float64), nfft)
        S = np.fft.irfft(V0 * np.conj(A0) - 2 * Vx * np.conj(Ax) - 2 * Vy * np.conj(Ay) + V3 * np.conj(A3), nfft)
        C = np.fft.irfft(V0 * np.conj(A3), nfft)
        n_d = Nt - win + 1
        S, C = S[:n_d], C[:n_d]
        msd = np.where(C >= MIN_OVERLAP, S / np.maximum(C, 1.0), np.inf)
        d = int(np.argmin(msd))
        if not np.isfinite(msd[d]):
            continue
        lo, hi = max(d - EXCLUDE, 0), min(d + EXCLUDE + 1, n_d)
        rest = msd.copy()
        rest[lo:hi] = np.inf
        out["d"][w] = d
        out["rms"][w] = np.sqrt(max(msd[d], 0.0))
        out["rms2"][w] = np.sqrt(rest.min()) if np.isfinite(rest.min()) else np.inf
        out["count"][w] = C[d]
        co = mb & mt[d:d + win]
        pts = Bw[co]
        out["spread"][w] = float(pts[:, 0].std() + pts[:, 1].std()) if len(pts) > 2 else 0.0
    out["starts"] = starts
    out["flip"] = flip
    out["win"] = win
    return out


def accept(al, max_rms=MAX_RMS_M, ratio=RATIO):
    """Boolean per window: confident AND supported by a neighbour (same offset) or a very large margin."""
    with np.errstate(invalid="ignore", divide="ignore"):
        moving = np.nan_to_num(al["spread"], nan=0.0) >= MIN_SPREAD_M
        conf = (np.isfinite(al["d"]) & moving & (al["rms"] <= max_rms)
                & (al["rms2"] >= ratio * np.maximum(al["rms"], 0.25)))
        big = conf & (al["rms2"] >= 2 * ratio * np.maximum(al["rms"], 0.25))
    delta = al["d"] - al["starts"]
    ok = np.zeros(len(conf), dtype=bool)
    for w in np.flatnonzero(conf):
        sup = False
        for v in range(max(0, w - 3), min(len(conf), w + 4)):
            if v != w and conf[v] and abs(delta[v] - delta[w]) <= 2:
                sup = True
                break
        ok[w] = sup or big[w]
    return ok


def best_orientation(tact_ball, bc_ball, log=print):
    """Try both orientations; keep the one with more accepted windows."""
    res = []
    for flip in (False, True):
        al = align(tact_ball, bc_ball, flip=flip)
        res.append((int(accept(al).sum()), flip, al))
    res.sort(key=lambda r: -r[0])
    n, flip, al = res[0]
    log("[broadcast] orientation flip={} accepted_windows={} (other: {})".format(flip, n, res[1][0]))
    return al, flip


def frame_map(al, ok, n_broadcast, return_win=False):
    """t_of_b [n_broadcast] int (-1 = unmapped); best (lowest RMS) accepted window wins.
    With return_win=True also returns win_of_b (window index that supplied each frame, -1 = unmapped)."""
    t_of_b = np.full(n_broadcast, -1, dtype=np.int64)
    win_of_b = np.full(n_broadcast, -1, dtype=np.int64)
    best = np.full(n_broadcast, np.inf)
    win = al["win"]
    for w in np.flatnonzero(ok):
        b0 = int(al["starts"][w])
        d = int(al["d"][w])
        r = al["rms"][w]
        sl = slice(b0, min(b0 + win, n_broadcast))
        upd = best[sl] > r
        k = np.arange(b0, min(b0 + win, n_broadcast))
        t_of_b[k[upd]] = d + (k[upd] - b0)
        win_of_b[k[upd]] = w
        best[sl] = np.where(upd, r, best[sl])
    return (t_of_b, win_of_b) if return_win else t_of_b


def window_confidence(al):
    """Per-window confidence = runner-up RMS / best RMS (>= RATIO for accepted windows; larger = more distinctive)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return al["rms2"] / np.maximum(al["rms"], 0.25)


def source_frames(bc_ball_norm, t_of_b, n_tactical):
    """src_b [n_tactical] int: the broadcast frame supplying tactical frame t (first mapped frame whose broadcast ball
    is seen), -1 = none. bc_ball_norm must already be in tactical orientation."""
    bc = np.asarray(bc_ball_norm, dtype=np.float64)
    src = np.full(n_tactical, -1, dtype=np.int64)
    mapped = np.flatnonzero((t_of_b >= 0) & (t_of_b < n_tactical))
    seen = np.isfinite(bc[mapped, 0])
    cand_b, cand_t = mapped[seen], t_of_b[mapped[seen]]
    if len(cand_t):
        ut, first = np.unique(cand_t, return_index=True)
        src[ut] = cand_b[first]
    return src


def gap_lengths(ball_norm):
    """Per-frame length of the run of missing ball frames containing it (0 where observed)."""
    miss = ~np.isfinite(np.asarray(ball_norm)[:, 0])
    out = np.zeros(len(miss), dtype=np.int64)
    e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    for s_, f_ in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        out[s_:f_] = f_ - s_
    return out


def fuse_ball(tact_ball_norm, bc_ball_norm, t_of_b, flip, n_tactical, min_gap=0):
    """Fill tactical ball gaps from mapped broadcast frames. Returns (fused_norm, filled_mask, stats).
    min_gap: only gaps (runs of missing tactical ball) at least this long are filled; shorter ones stay with the
    short-gap interpolation. Tactical observations are never overwritten."""
    fused = np.asarray(tact_ball_norm, dtype=np.float32).copy()
    bc = np.asarray(bc_ball_norm, dtype=np.float64)
    if flip:
        bc = 1.0 - bc
    src = source_frames(bc, t_of_b, n_tactical)
    missing = ~np.isfinite(fused[:, 0])
    filled = missing & (src >= 0)
    if min_gap > 0:
        filled &= gap_lengths(tact_ball_norm) >= min_gap
    fused[filled] = bc[src[filled]]
    mapped = np.flatnonzero((t_of_b >= 0) & (t_of_b < n_tactical))
    seen = np.isfinite(bc[mapped, 0])
    cand_b, cand_t = mapped[seen], t_of_b[mapped[seen]]
    t_missing = missing[cand_t]
    both = ~t_missing
    dist = np.hypot(*((np.asarray(tact_ball_norm, dtype=np.float64)[cand_t[both]] - bc[cand_b[both]]) * PITCH).T) if both.any() else np.zeros(0)
    stats = {
        "broadcast_frames_mapped": int(len(mapped)),
        "broadcast_mapped_frac": float(len(mapped)) / max(len(t_of_b), 1),
        "tactical_ball_gaps_filled": int(filled.sum()),
        "agreement_median_m": float(np.median(dist)) if len(dist) else float("nan"),
        "agreement_p90_m": float(np.percentile(dist, 90)) if len(dist) else float("nan"),
    }
    return fused, filled, stats


def verify_map(t_of_b, tact_ball_norm, bc_ball_norm, flip, radius=75, tol_m=3.0, min_co=5, min_frac=0.8):
    """Per-frame verification of a window-derived mapping: a mapped broadcast frame is kept only if, in the +/-radius
    frames around it, the co-observed ball positions agree (<= tol_m) for >= min_frac of at least min_co co-observed
    frames. This removes frames of a window that straddle a cut/replay boundary. Returns a new t_of_b."""
    import pandas as pd
    nb = len(t_of_b)
    T = _metres(tact_ball_norm)
    B = _metres(bc_ball_norm, flip)
    out = t_of_b.copy()
    mapped = (t_of_b >= 0) & (t_of_b < len(T))
    co = np.zeros(nb, dtype=bool)
    agree = np.zeros(nb, dtype=bool)
    idx = np.flatnonzero(mapped)
    tb = T[t_of_b[idx]]
    bb = B[idx]
    both = np.isfinite(tb[:, 0]) & np.isfinite(bb[:, 0])
    d = np.hypot(tb[:, 0] - bb[:, 0], tb[:, 1] - bb[:, 1])
    co[idx[both]] = True
    agree[idx[both]] = d[both] <= tol_m
    win = 2 * radius + 1
    co_n = pd.Series(co.astype(float)).rolling(win, center=True, min_periods=1).sum().to_numpy()
    ag_n = pd.Series(agree.astype(float)).rolling(win, center=True, min_periods=1).sum().to_numpy()
    keep = (co_n >= min_co) & (ag_n >= min_frac * np.maximum(co_n, 1.0))
    out[~keep] = -1
    return out
