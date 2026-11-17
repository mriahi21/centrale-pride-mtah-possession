"""Does the tactical video / audio add evidence the tracking can use? Verification with negative controls.

Video motion energy (a "quiet pitch" cue) can be fooled by the camera itself (panning, zooming, a bad calibration
makes the whole pitch "move"). So the cue is checked and gated against the camera motion derived from the tactical
homography, its clock is checked against the tracking clock block by block, and every statistic is compared with a
null distribution obtained by circularly time-shifting the cue (a shifted cue keeps its marginal distribution and
its autocorrelation but loses any real link to what is happening on the pitch).

References used for the checks come from the TRACKING only (player activity, ball out-of-bounds, stoppages found by
the tracking-only v2 model). No labels. A cue is kept only if it carries information about stoppages that is not
already in the tracking activity (incremental AUC against out-of-bounds, residual after regressing out activity and
camera motion) and beats its time-shifted controls. The verdicts are verification, not accuracy.
"""
import numpy as np

import poss  # noqa: F401
from poss import video_activity

STRIDE = 5
PAN_STILL_PX = 3.0            # px moved by the pitch centre per 5 frames (1920x1080) below which the camera counts as still
ZOOM_STILL = 0.004            # |log scale change| per 5 frames
RULES = {
    "min_still_coverage": 0.30,     # camera-still graded frames with a reliable homography
    "max_block_lag": 12,            # frames, every 15000-frame block (clock agrees with the tracking clock)
    "min_r_over_null": 0.15,        # r(video, tracking activity) - max r of the time-shifted controls
    "min_partial_r": 0.20,          # r(video, activity | camera motion)
    "min_incremental_auc": 0.55,    # residual cue vs ball-out-of-bounds
    "min_auc_over_null": 0.03,
    "min_whistle_hit_over_null": 0.05,
    "whistle_max_ratio": 0.97,
}
N_NULL = 40


def camera_motion(hom, stride=STRIDE, smooth=51):
    """(pan_px, zoom) per frame: movement of the projected pitch centre and |log scale change| between frames f-stride and f.
    NaN where the homography is invalid at either frame. Smoothed with a rolling max so a pan blocks its whole window."""
    import pandas as pd
    H = hom["H"]
    n = len(H)
    px = np.array([.25, .5, .75, .5, .5])
    py = np.array([.5, .5, .5, .25, .75])
    P = np.stack([px, py, np.ones(5)])
    with np.errstate(invalid="ignore", divide="ignore"):
        Q = H @ P
        x, y = Q[:, 0, :] / Q[:, 2, :], Q[:, 1, :] / Q[:, 2, :]
        cx, cy = x.mean(1), y.mean(1)
        spread = np.sqrt(((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2).mean(1))
        pan = np.full(n, np.nan)
        zoom = np.full(n, np.nan)
        pan[stride:] = np.hypot(cx[stride:] - cx[:-stride], cy[stride:] - cy[:-stride])
        zoom[stride:] = np.abs(np.log(spread[stride:] / spread[:-stride]))
    pan[~np.isfinite(pan)] = np.nan
    zoom[~np.isfinite(zoom)] = np.nan
    if smooth > 1:
        pan = pd.Series(pan).rolling(smooth, center=True, min_periods=1).max().to_numpy()
        zoom = pd.Series(zoom).rolling(smooth, center=True, min_periods=1).max().to_numpy()
    return pan, zoom


def camera_still(hom_ok, pan, zoom):
    return np.asarray(hom_ok, dtype=bool) & np.isfinite(pan) & np.isfinite(zoom) & (pan <= PAN_STILL_PX) & (zoom <= ZOOM_STILL)


def _resid(y, Z):
    """Residual of y after least-squares regression on the columns of Z (+ intercept), NaN-aware (NaN rows stay NaN)."""
    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    out = np.full(len(y), np.nan)
    if m.sum() < 100:
        return out
    A = np.column_stack([np.ones(m.sum()), Z[m]])
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    out[m] = y[m] - A @ beta
    return out


def partial_corr(x, y, Z):
    rx, ry = _resid(x, Z), _resid(y, Z)
    m = np.isfinite(rx) & np.isfinite(ry)
    if m.sum() < 100 or rx[m].std() == 0 or ry[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(rx[m], ry[m])[0, 1])


def auc(score, positive, max_n=200000, seed=0):
    """P(score of a positive frame > score of a negative frame) (Mann-Whitney), NaN-aware, subsampled."""
    m = np.isfinite(score)
    s, p = score[m], positive[m]
    if len(s) > max_n:
        k = np.random.default_rng(seed).choice(len(s), max_n, replace=False)
        s, p = s[k], p[k]
    npos, nneg = int(p.sum()), int((~p).sum())
    if npos < 50 or nneg < 50:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _shifts(n, k=N_NULL, seed=0, lo=3000):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, max(n - lo, lo + 1), size=k)


def block_lags(a, b, block=15000, max_lag=100, step=3, min_valid=3000):
    """Best alignment lag of two series per 15000-frame block (clock agreement along the whole match)."""
    out = []
    for s in range(0, len(a) - block // 2, block):
        x, y = a[s:s + block], b[s:s + block]
        if (np.isfinite(x) & np.isfinite(y)).sum() < min_valid:
            continue
        lag, r, _ = video_activity.lag_correlation(x, y, max_lag=max_lag, step=step)
        out.append((s, int(lag), float(r)))
    return out


def verify_video(vseries, tracking_act, hom, hom_ok, graded, ref, seed=0):
    """Verify the video motion-energy cue. Returns dict(verdict, lines, gated, stats).

    ref: {"oob": bool[n] ball out of bounds (tracking), "dead": bool[n] stoppage frames of the tracking-only v2 model}.
    gated: the cue with camera-busy / unreliable-calibration frames set to NaN (the model reads NaN as 'no evidence').
    """
    n = len(vseries)
    pan, zoom = camera_motion(hom)
    still = camera_still(hom_ok, pan, zoom)
    g = np.asarray(graded, dtype=bool)
    cov = float((still & g).sum() / max(g.sum(), 1))
    busy = g & ~still & np.asarray(hom_ok, dtype=bool)
    lines, checks = [], {}

    # 1. camera motion: does panning/zooming drive the motion energy, and does it create false "quiet/active" cues?
    cam = np.column_stack([np.nan_to_num(pan, nan=0.0), np.nan_to_num(zoom, nan=0.0)])
    use = g & np.asarray(hom_ok, dtype=bool)
    v_use = np.where(use, vseries, np.nan)
    a_use = np.where(use, tracking_act, np.nan)
    r_cam = float(np.corrcoef(v_use[np.isfinite(v_use) & np.isfinite(pan)], pan[np.isfinite(v_use) & np.isfinite(pan)])[0, 1]) \
        if (np.isfinite(v_use) & np.isfinite(pan)).sum() > 100 else float("nan")
    p_r = partial_corr(v_use, a_use, cam)
    lines.append("camera: still on {:.1%} of graded frames; corr(video energy, camera pan) = {:.2f}; corr(video, tracking activity | camera) = {:.2f}".format(cov, r_cam, p_r))
    vv = np.where(g, vseries, np.nan)
    fin = np.isfinite(vv)
    if fin.sum() > 1000:
        q35 = np.nanpercentile(vv[fin & still], 35) if (fin & still).sum() > 1000 else np.nanpercentile(vv[fin], 35)
        hi_act = np.isfinite(tracking_act) & (tracking_act >= np.nanpercentile(tracking_act[g], 50))
        fr = lambda m: float(np.mean(vv[m & fin & hi_act] <= q35)) if (m & fin & hi_act).sum() > 200 else float("nan")  # noqa: E731
        f_still, f_busy = fr(still), fr(busy)
        lines.append("false 'quiet' cue while the players are running: camera still {:.1%}, camera moving/uncalibrated {:.1%}".format(f_still, f_busy))
    checks["still_cov"] = cov >= RULES["min_still_coverage"]

    # 2. clock: per-block lag against tracking activity (a constant or drifting offset would show here)
    gated = np.where(still, vseries, np.nan)
    bl = block_lags(np.where(g, gated, np.nan), np.where(g, tracking_act, np.nan))
    lags = [b[1] for b in bl]
    lines.append("clock: per-block best lag (frames) {}; max |lag| {}".format(lags, max([abs(x) for x in lags], default="n/a")))
    checks["clock"] = bool(lags) and max(abs(x) for x in lags) <= RULES["max_block_lag"]

    # 3. correlation with tracking activity vs time-shifted controls (camera-still frames only)
    m0 = g & still
    def r_of(v):
        mm = m0 & np.isfinite(v) & np.isfinite(tracking_act)
        return float(np.corrcoef(v[mm], tracking_act[mm])[0, 1]) if mm.sum() > 1000 else float("nan")
    r_obs = r_of(vseries)
    nulls = [r_of(np.roll(vseries, int(s))) for s in _shifts(n, seed=seed)]
    nulls = [x for x in nulls if np.isfinite(x)]
    # the cue is "quiet when players are quiet": activity and video energy are positively related, so compare r directly
    null_max = max(nulls) if nulls else float("nan")
    lines.append("r(video, tracking activity) on camera-still frames = {:.2f}; time-shifted controls: max {:.2f}, median {:.2f}".format(
        r_obs, null_max, float(np.median(nulls)) if nulls else float("nan")))
    checks["r_over_null"] = bool(np.isfinite(r_obs) and np.isfinite(null_max) and r_obs - null_max >= RULES["min_r_over_null"])
    checks["partial_r"] = bool(np.isfinite(p_r) and p_r >= RULES["min_partial_r"])

    # 4. incremental information about stoppages beyond tracking activity: AUC of the residual cue against tracking-only references
    qv = -vseries                                           # quiet = high score
    res_q = _resid(np.where(m0, qv, np.nan), np.column_stack([np.nan_to_num(tracking_act, nan=0.0), cam]))
    inc = {}
    for name in ("oob", "dead"):
        pos = np.asarray(ref[name], dtype=bool) & m0
        a_obs = auc(np.where(m0, res_q, np.nan), np.asarray(ref[name], dtype=bool))
        null_a = []
        for s in _shifts(n, k=N_NULL, seed=seed + 1):          # shift the residual: keeps its autocorrelation, breaks its link to the reference
            x = auc(np.roll(res_q, int(s)), np.asarray(ref[name], dtype=bool))
            if np.isfinite(x):
                null_a.append(x)
        inc[name] = (a_obs, max(null_a) if null_a else float("nan"))
        lines.append("incremental AUC (residual quiet cue vs tracking-derived {}): {:.3f}; time-shifted max {:.3f}; positives {}".format(
            name, a_obs, inc[name][1], int(pos.sum())))
    a, nm = inc["oob"]
    checks["incremental"] = bool(np.isfinite(a) and a >= RULES["min_incremental_auc"]
                                 and (not np.isfinite(nm) or a - nm >= RULES["min_auc_over_null"]))
    verdict = all(checks.values())
    lines.append("checks: " + ", ".join("{}={}".format(k, "ok" if v else "FAIL") for k, v in checks.items()))
    return {"verdict": bool(verdict), "checks": checks, "lines": lines, "gated": gated, "still_cov": cov}


def verify_whistle(event_frames, tracking_act, graded, ref_dead_starts, n, seed=0):
    """Verify whistle events: tracking activity must drop after them, they must coincide with tracking-derived stoppage
    starts more often than time-shifted copies do. Returns dict(verdict, lines)."""
    ev = np.asarray(sorted(event_frames), dtype=np.int64)
    g = np.asarray(graded, dtype=bool)

    def ratio(frames):
        out = []
        for f in frames:
            if 150 < f < n - 200 and g[f]:
                pre, post = np.nanmean(tracking_act[f - 125:f]), np.nanmean(tracking_act[f + 25:f + 150])
                if np.isfinite(pre) and np.isfinite(post) and pre > 0:
                    out.append(post / pre)
        return float(np.median(out)) if len(out) >= 10 else float("nan")

    def hit(frames):
        if not len(ref_dead_starts) or not len(frames):
            return float("nan")
        d = np.abs(np.asarray(ref_dead_starts)[:, None] - np.asarray(frames)[None, :]).min(1)
        return float(np.mean(d <= 75))

    r_obs, h_obs = ratio(ev), hit(ev)
    rng = np.random.default_rng(seed)
    r_null, h_null = [], []
    for _ in range(N_NULL):
        sh = ((ev + int(rng.integers(500, max(n - 500, 501)))) % n)
        r_null.append(ratio(sh))
        h_null.append(hit(sh))
    r_null = [x for x in r_null if np.isfinite(x)]
    h_null = [x for x in h_null if np.isfinite(x)]
    r5 = float(np.percentile(r_null, 5)) if r_null else float("nan")
    h95 = float(np.percentile(h_null, 95)) if h_null else float("nan")
    lines = ["whistle: {} events; median post/pre tracking activity {:.2f} (time-shifted 5th pct {:.2f}); "
             "hit rate on tracking-derived stoppage starts (+-3 s) {:.2f} (time-shifted 95th pct {:.2f})".format(
                 len(ev), r_obs, r5, h_obs, h95)]
    ok_ratio = bool(np.isfinite(r_obs) and r_obs <= RULES["whistle_max_ratio"] and (not np.isfinite(r5) or r_obs < r5))
    ok_hit = bool(np.isfinite(h_obs) and (not np.isfinite(h95) or h_obs >= h95 + RULES["min_whistle_hit_over_null"]))
    lines.append("checks: activity_drop={}, beats_time_shift_hit_rate={}".format("ok" if ok_ratio else "FAIL", "ok" if ok_hit else "FAIL"))
    return {"verdict": bool(ok_ratio and ok_hit), "lines": lines}
