"""Possession-state model: soft ball-control evidence + a two-state Viterbi with a switch penalty.

Why this and not a per-frame vote
- A possession is an interval. The metric (spellF1) counts maximal H/A runs, so flicker is poison
  and an unwarranted split is as bad as a missed hand-over.
- Evidence is *soft*: a player within ~2 m of a slow ball says a lot; a fast ball flying past a
  defender says nothing; a missing ball says nothing (it is never turned into "dead").
- The Viterbi switch penalty (lambda) is the hysteresis: possession only changes hands when the
  other team's accumulated control evidence outweighs lambda. The switch lands where the opposing
  run of evidence *starts*, so timing is not delayed by the dwell requirement.
"""
import numpy as np

import poss  # noqa: F401

H, A, D = 0, 1, 2

SIGMA_M = 2.0               # control kernel width (m)
V_FULL, V_ZERO = 4.0, 12.0  # ball speed (m/s) above which proximity is progressively discounted
LAMBDA_GRID = (3, 5, 8, 12, 20, 30, 45, 65, 90, 130)
TARGET_MEAN_SPELL_S = 12.0  # generic football prior for mean possession length (NOT tuned on any GT)


def control_evidence(ft, sigma=SIGMA_M):
    """Signed per-frame evidence in [-1, 1]; positive = home, 0 = no information."""
    with np.errstate(over="ignore", invalid="ignore"):
        fh = np.exp(-0.5 * (ft["dH"] / sigma) ** 2)
        fa = np.exp(-0.5 * (ft["dA"] / sigma) ** 2)
    fh = np.nan_to_num(fh, nan=0.0, posinf=0.0)
    fa = np.nan_to_num(fa, nan=0.0, posinf=0.0)
    v = ft["ball_speed"]
    g = np.where(np.isnan(v), 0.7, np.clip(1.0 - 0.9 * (v - V_FULL) / (V_ZERO - V_FULL), 0.1, 1.0))
    return ((fh - fa) * g * ft["ball_conf"]).astype(np.float32)


def viterbi_two_state(u, lam, prior=0.0):
    """Minimise sum(-u_t if H else +u_t) + lam * (#switches). Returns int8 array of 0 (H) / 1 (A).

    `prior` > 0 biases the first frame towards H, < 0 towards A (in the same units as u).
    """
    n = len(u)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    uu = u.astype(np.float64).tolist()
    stay_h = np.zeros(n, dtype=bool)  # True if best predecessor of H at t is H
    stay_a = np.zeros(n, dtype=bool)
    ch, ca = -uu[0] - prior, uu[0] + prior
    for t in range(1, n):
        sh = ch <= ca + lam
        sa = ca <= ch + lam
        nh = (ch if sh else ca + lam) - uu[t]
        na = (ca if sa else ch + lam) + uu[t]
        stay_h[t], stay_a[t] = sh, sa
        ch, ca = nh, na
    state = 0 if ch <= ca else 1
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = state
        if t:
            if state == 0:
                state = 0 if stay_h[t] else 1
            else:
                state = 1 if stay_a[t] else 0
    return out


def alive_segments(alive):
    """[(start, stop)] maximal True runs of a boolean array."""
    a = np.concatenate(([0], alive.astype(np.int8), [0]))
    e = np.diff(a)
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def decode_alive(u, alive, lam, seg_priors=None, min_run=25):
    """Per-alive-segment Viterbi. Returns int8 codes (H/A) where alive, D elsewhere."""
    out = np.full(len(u), D, dtype=np.int8)
    for i, (a, b) in enumerate(alive_segments(alive)):
        prior = 0.0 if seg_priors is None else seg_priors[i]
        s = viterbi_two_state(u[a:b], lam, prior)
        out[a:b] = _merge_short_runs(s, min_run)
    return out


def _merge_short_runs(s, min_run):
    """Absorb runs shorter than min_run into their neighbour(s) when the segment has several runs."""
    s = s.copy()
    for _ in range(3):
        br = np.flatnonzero(np.diff(s) != 0) + 1
        bounds = np.concatenate(([0], br, [len(s)]))
        if len(bounds) <= 2:
            break
        changed = False
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            if b - a < min_run:
                left = s[a - 1] if a > 0 else None
                right = s[b] if b < len(s) else None
                new = left if left is not None else right
                if new is not None and new != s[a]:
                    s[a:b] = new
                    changed = True
        if not changed:
            break
    return s


def count_spells(codes):
    """Number of maximal H/A runs, treating D as a stoppage (like the scorer)."""
    c = codes
    alive = c != D
    if not alive.any():
        return 0
    change = (c[1:] != c[:-1]) & alive[1:]
    return int(alive[0]) + int(change.sum())


def choose_lambda(u, alive, fps=25, target_s=TARGET_MEAN_SPELL_S, grid=LAMBDA_GRID, seg_priors=None):
    """Pick the switch penalty whose mean possession length is closest (log scale) to the prior."""
    n_alive = int(alive.sum())
    best, best_gap, table = grid[len(grid) // 2], np.inf, []
    for lam in grid:
        codes = decode_alive(u, alive, lam, seg_priors)
        k = count_spells(codes)
        mean_s = n_alive / fps / max(k, 1)
        gap = abs(np.log(max(mean_s, 1e-3) / target_s))
        table.append((lam, k, mean_s))
        if gap < best_gap:
            best, best_gap = lam, gap
    return best, table
