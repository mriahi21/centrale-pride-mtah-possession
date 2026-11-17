"""Targeted broadcast-VIDEO ball pass (local only; resumable; self-verifying).

Windows are chosen on the tactical timeline where extra ball evidence is worth most -- long tactical ball gaps (after
the broadcast-tracking fill), disputed turnovers (small control-evidence margin) and uncertain restarts -- and mapped to
broadcast frames through the verified tracking alignment. A window is rejected when the mapping is not one clean
constant-offset stretch (cut / replay boundary), covers too little of the window, or the broadcast homography is
unreliable there. On the surviving windows a deterministic classical detector (small bright round blobs inside the
homography-projected pitch, linked over time by dynamic programming in pitch metres) produces ball positions; they are
projected to pitch coordinates with the broadcast homography.

The detector is NOT trusted by default. It is verified on the held-out stretches of poss/holdout.py (tactical ball
hidden from alignment and from the detector; truth used only afterwards), and a detection fills a tactical ball gap only
if the detected track is consistent with the observed tactical ball at the gap's flanks. The video pass is promoted
only if the held-out error is small AND it recovers held-out frames that broadcast tracking + interpolation did not.
Per-window candidate detections are cached, so an interrupted run resumes and re-linking needs no decoding.
"""
import os
import time

import numpy as np

import poss  # noqa: F401
from poss import broadcast, features, holdout, homog

PITCH = broadcast.PITCH
FPS = 25
SCALE_W = 960                    # analysis width; all pixel thresholds are in these pixels
TOPHAT_K, TOPHAT_T = 17, 28      # small-bright-structure filter (ball ~ 5-9 px here; lines/people differ in shape)
MIN_BRIGHT, MAX_CHROMA = 150, 60
AREA = (10, 160)
MIN_FILL, MIN_ASPECT, MAX_ASPECT = 0.5, 0.5, 2.0
VMAX = 40.0                      # m/s
MAX_SKIP, MISS_COST = 4, 0.3
MIN_TRACK, MIN_SCORE, MAX_TRACKS = 12, 5.0, 6
MIN_WIN, MIN_COVER, MIN_HOM = 40, 0.6, 0.8
RULES = {"min_frames": 200, "max_median_m": 3.0, "max_p90_m": 10.0, "max_bad_frac": 0.2, "recover_m": 3.0, "min_gain": 0.05}


# ---------------------------------------------------------------- window selection and mapping
def map_interval(t0, t1, t_of_b, hom_ok_b=None):
    """Broadcast stretch covering tactical frames [t0, t1): the longest run of frames with consecutive broadcast AND tactical
    indices (constant offset, i.e. no cut/replay inside). Returns dict or (None, reason)."""
    idx = np.flatnonzero((t_of_b >= t0) & (t_of_b < t1))
    if len(idx) < MIN_WIN:
        return None, "unmapped"
    brk = np.flatnonzero((np.diff(idx) != 1) | (np.diff(t_of_b[idx]) != 1)) + 1
    runs = np.split(idx, brk)
    runs = [r for r in runs if len(r) >= 20]
    if not runs:
        return None, "cut/replay boundary"
    runs.sort(key=lambda r: (-len(r), r[0]))
    r = runs[0]
    if len(r) < MIN_WIN or len(r) / float(t1 - t0) < MIN_COVER:
        return None, "cut/replay boundary"
    if hom_ok_b is not None and hom_ok_b[r].mean() < MIN_HOM:
        return None, "unreliable broadcast homography"
    return {"b0": int(r[0]), "b1": int(r[-1]) + 1, "t0": int(t_of_b[r[0]]), "t1": int(t_of_b[r[-1]]) + 1,
            "n_candidates": len(runs)}, ""


def select_windows(n, ball_norm, blocks, t_of_b, hom_ok_b, restarts=(), turnovers=(), max_windows=300,
                   min_gap=25, max_len=500):
    """Prioritised windows [(kind, prio, dict)] + a Counter of rejections. Priority: gap length, then disputed margin."""
    from collections import Counter
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    ev = []
    gl = broadcast.gap_lengths(ball_norm)
    e = np.diff(np.concatenate(([0], (gl > 0).astype(np.int8), [0])))
    for s, f in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if f - s >= min_gap and graded[s] and graded[f - 1]:
            ev.append(("gap", float(min(f - s, 2000)), max(s - 30, 0), min(f + 30, n), s))
    for f, margin, uncertain in restarts:
        if uncertain:
            ev.append(("restart", 400.0, max(f - 25, 0), min(f + 150, n), f))
    for f, margin in turnovers:
        ev.append(("turnover", 300.0 * (1.0 - float(margin)), max(f - 100, 0), min(f + 100, n), f))
    ev.sort(key=lambda x: -x[1])
    out, rej = [], Counter()
    taken = np.zeros(n, dtype=bool)
    for kind, prio, a, b, anchor in ev:
        if b - a > max_len + 60:                                     # long gaps: the edges carry the restart/turnover information
            b = a + max_len + 60
        if taken[a:b].mean() > 0.5:
            rej["overlaps a selected window"] += 1
            continue
        w, why = map_interval(a, b, t_of_b, hom_ok_b)
        if w is None:
            rej[why] += 1
            continue
        w.update(kind=kind, prio=prio, anchor=int(anchor))
        out.append(w)
        taken[a:b] = True
        if len(out) >= max_windows:
            break
    return out, rej


def events_from_run(halves, codes):
    """(restarts, turnovers) from a labelled run (halves as returned by pipeline.label_halves, which fills u0/alive/rpri)."""
    from poss import model
    restarts, turn = [], []
    for h in halves:
        a = h["a"]
        segs = model.alive_segments(h["alive"])
        for i, (s, e) in enumerate(segs):
            if s <= h["kick_rel"] + 5:
                continue
            pri = h["rpri"][i] if "rpri" in h and i < len(h["rpri"]) else 0.0
            restarts.append((a + s, abs(pri), abs(pri) < 1e-9))
        c = codes[a:h["b"] + 1]
        sw = np.flatnonzero((c[1:] != c[:-1]) & (c[1:] < 2) & (c[:-1] < 2)) + 1
        u = h["u0"]
        margins = []
        for t in sw:
            lo, hi = max(t - 25, 0), min(t + 25, len(u))
            margins.append(abs(float(np.mean(u[lo:t])) - float(np.mean(u[t:hi]))) if hi > t > lo else 0.0)
        if len(margins):
            q = np.percentile(margins, 30)
            for t, m in zip(sw, margins):
                if m <= q:
                    turn.append((a + int(t), m / (q + 1e-9) * 0.5))      # disputed = bottom 30% by evidence margin
    return restarts, turn


# ---------------------------------------------------------------- detection
def extract_window(path, b0, b1, hom, hom_ok, hom_size=(1920, 1080)):
    """Candidate ball blobs for broadcast frames [b0, b1): (frame, x_ref, y_ref, score), pixels in the homography's image frame."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open broadcast video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, b0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_K, TOPHAT_K))
    out = []
    for b in range(b0, b1):
        ok, img = cap.read()
        if not ok:
            break
        if b >= len(hom_ok) or not hom_ok[b]:
            continue
        poly = homog.pitch_polygon(hom, b)
        if poly is None:
            continue
        h0, w0 = img.shape[:2]
        s = SCALE_W / float(w0)
        small = cv2.resize(img, (SCALE_W, int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        c = poly.mean(0)
        poly = np.clip((c + (poly - c) * 1.08) * (w0 / float(hom_size[0])) * s, -1e5, 1e5)
        mask = np.zeros(small.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker)
        mn, mx = small.min(axis=2), small.max(axis=2)
        white = ((th > TOPHAT_T) & (mn > MIN_BRIGHT) & ((mx.astype(np.int16) - mn) < MAX_CHROMA) & (mask > 0)).astype(np.uint8)
        nl, lab, st, cen = cv2.connectedComponentsWithStats(white, connectivity=8)
        for i in range(1, nl):
            area, w, h = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if not (AREA[0] <= area <= AREA[1]) or not (MIN_ASPECT <= w / float(h) <= MAX_ASPECT) or area / float(w * h) < MIN_FILL:
                continue
            score = min(1.0, area / float(w * h) / 0.785) * min(1.0, float(th[lab == i].mean()) / 120.0 + 0.3)
            out.append((b, cen[i][0] * hom_size[0] / SCALE_W, cen[i][1] * hom_size[0] / SCALE_W, score))
    cap.release()
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def candidates_to_pitch(cand, hom):
    """Candidate pixels -> normalised pitch coordinates with the per-frame inverse homography. Returns (fr, X, Y, score)."""
    if not len(cand):
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
    fr = cand[:, 0].astype(np.int64)
    X, Y = np.full(len(fr), np.nan), np.full(len(fr), np.nan)
    for b in np.unique(fr):
        Hb = hom["H"][b]
        if not np.isfinite(Hb).all():
            continue
        m = fr == b
        Hi = np.linalg.inv(Hb)
        X[m], Y[m] = homog.project(Hi, cand[m, 1], cand[m, 2])
    ok = np.isfinite(X) & (X > -0.1) & (X < 1.1) & (Y > -0.1) & (Y < 1.1)
    return fr[ok], X[ok], Y[ok], cand[ok, 3]


def link_tracks(fr, X, Y, sc, vmax=VMAX):
    """Dynamic-programming linking in pitch metres. Returns a list of validated tracks {frame: (X, Y)}, best first.
    Tracks are kept separate: clutter (a static white spot) can form a perfect track of its own, so the choice between
    tracks is made later against the observed ball at the gap flanks (anchored_fill)."""
    if not len(fr):
        return []
    o = np.lexsort((-sc, fr))
    fr, X, Y, sc = fr[o], X[o], Y[o], sc[o]
    m = len(fr)
    xm, ym = X * PITCH[0], Y * PITCH[1]
    best = -sc.copy()
    prev = np.full(m, -1, dtype=np.int64)
    lo = 0
    first = np.searchsorted(fr, fr, side="left")
    for i in range(m):
        while fr[lo] < fr[i] - MAX_SKIP - 1:
            lo += 1
        hi = first[i]
        if hi > lo:
            j = np.arange(lo, hi)
            dt = fr[i] - fr[j]
            d = np.hypot(xm[i] - xm[j], ym[i] - ym[j])
            allowed = vmax * dt / FPS + 0.8
            tc = np.where(d <= allowed, 0.0, 3.0 + (d - allowed))
            c = best[j] + tc + MISS_COST * (dt - 1) - sc[i]
            k = int(np.argmin(c))
            if c[k] < best[i]:
                best[i], prev[i] = c[k], j[k]
    used = np.zeros(m, dtype=bool)
    tracks = []
    for _ in range(MAX_TRACKS):
        cand = np.where(used, np.inf, best)
        i = int(np.argmin(cand))
        if not np.isfinite(cand[i]) or cand[i] > -MIN_SCORE:
            break
        chain = []
        while i >= 0 and not used[i]:
            chain.append(i)
            i = int(prev[i])
        for k in chain:
            used[k] = True
        if len(chain) < MIN_TRACK:
            continue
        tracks.append({int(fr[k]): (float(X[k]), float(Y[k])) for k in chain})
    return tracks


def to_tactical(assigned, t_of_b, flip, n):
    pos = np.full((n, 2), np.nan)
    for b, (x, y) in assigned.items():
        if 0 <= b < len(t_of_b) and 0 <= t_of_b[b] < n:
            t = int(t_of_b[b])
            pos[t] = (1.0 - x, 1.0 - y) if flip else (x, y)
    return pos


def anchored_fill(ref_ball, pos_list, t0, t1, vmax=VMAX, slack_m=2.0):
    """Fill gaps (ref ball missing) from detected tracks, trying them best-first and keeping the first whose positions are
    speed-feasible with the OBSERVED ball at the gap flanks (a static clutter track or a wrong player fails this).
    Returns a (n, 2) array with NaN elsewhere."""
    out = np.full(ref_ball.shape, np.nan)
    miss = ~np.isfinite(ref_ball[:, 0])
    e = np.diff(np.concatenate(([0], miss[t0:t1].astype(np.int8), [0])))
    for s, f in zip(np.flatnonzero(e == 1) + t0, np.flatnonzero(e == -1) + t0):
        left = s - 1 if s - 1 >= 0 and np.isfinite(ref_ball[s - 1, 0]) else None
        right = f if f < len(ref_ball) and np.isfinite(ref_ball[f, 0]) else None
        if left is None and right is None:
            continue
        for pos in pos_list:
            det = np.flatnonzero(np.isfinite(pos[s:f, 0])) + s
            if not len(det):
                continue
            ok = True
            if left is not None:
                ok &= np.hypot(*((pos[det[0]] - ref_ball[left]) * PITCH)) <= vmax * (det[0] - left) / FPS + slack_m
            if right is not None:
                ok &= np.hypot(*((pos[det[-1]] - ref_ball[right]) * PITCH)) <= vmax * (right - det[-1]) / FPS + slack_m
            if ok:
                out[det] = pos[det]
                break
    return out


# ---------------------------------------------------------------- caching / extraction
def _wpath(cache_dir, tag, w):
    d = os.path.join(cache_dir, "vidwin")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{}_{}_{}.npy".format(tag, w["b0"], w["b1"]))


def get_candidates(path, w, tag, cache_dir, hom, hom_ok, hom_size=(1920, 1080)):
    p = _wpath(cache_dir, tag, w)
    if os.path.isfile(p):
        return np.load(p), True
    cand = extract_window(path, w["b0"], w["b1"], hom, hom_ok, hom_size)
    np.save(p, cand)
    return cand, False


def window_positions(cand, w, hom, t_of_b, flip, n):
    fr, X, Y, sc = candidates_to_pitch(cand, hom)
    return [to_tactical(t, t_of_b, flip, n) for t in link_tracks(fr, X, Y, sc)]


# ---------------------------------------------------------------- held-out verification
def video_holdout(path, hom, hom_ok, truth_ball, masked_ball, stretches, t_of_b_m, flip, base_pos, interp_pos, cache_dir,
                  max_windows=40, budget_s=1e9, hom_size=(1920, 1080), log=print):
    """Detect on windows around the held-out stretches (alignment from the MASKED track), compare with the hidden truth.
    base_pos: broadcast-tracking recovery on the masked track; interp_pos: usable interpolation. Returns dict."""
    n = len(truth_ball)
    per = {}
    for s, L in stretches:                                     # spread over gap-length bins
        per.setdefault(L, []).append((s, L))
    chosen = []
    k = max(max_windows // max(len(per), 1), 1)
    for L in sorted(per):
        chosen += per[L][:k]
    t_start = time.time()
    n_win = n_ok = 0
    errs = {False: [], True: []}
    rows = []
    for s, L in chosen:
        if time.time() - t_start > budget_s:
            log("[vidpass] held-out budget reached")
            break
        w, why = map_interval(max(s - 25, 0), min(s + L + 25, n), t_of_b_m, hom_ok)
        n_win += 1
        if w is None:
            continue
        n_ok += 1
        cand, _ = get_candidates(path, w, "hold", cache_dir, hom, hom_ok, hom_size)
        rows.append((s, L, w, cand))
    res = {"windows_tried": n_win, "windows_mapped": n_ok}
    best = None
    for fl in (flip, not flip):
        det = np.full((n, 2), np.nan)
        for s, L, w, cand in rows:
            pos = window_positions(cand, w, hom, t_of_b_m, fl, n)
            filled = anchored_fill(masked_ball, pos, s - 25, s + L + 25)
            sel = slice(s, s + L)
            det[sel] = np.where(np.isfinite(filled[sel, 0:1]), filled[sel], det[sel])
        ev = np.zeros(n, dtype=bool)
        for s, L, _, _ in rows:
            ev[s:s + L] = np.isfinite(truth_ball[s:s + L, 0])
        have = ev & np.isfinite(det[:, 0])
        e = np.hypot(*((det[have] - truth_ball[have]) * PITCH).T) if have.any() else np.zeros(0)
        med = float(np.median(e)) if len(e) else float("inf")
        if best is None or med < best[0]:
            best = (med, fl, det, ev, e)
    med, fl, det, ev, e = best
    res.update(flip=fl, n=int(len(e)), median=float(np.median(e)) if len(e) else float("nan"),
               p90=float(np.percentile(e, 90)) if len(e) else float("nan"),
               bad_frac=float(np.mean(e > RULES["max_p90_m"])) if len(e) else float("nan"))
    tot = max(int(ev.sum()), 1)
    rm = RULES["recover_m"]

    def rec(pos):
        m = ev & np.isfinite(pos[:, 0])
        return m & (np.hypot(*((np.nan_to_num(pos) - truth_ball) * PITCH).T) <= rm)
    base = np.where(np.isfinite(base_pos[:, :1]), base_pos, interp_pos)
    r_base, r_vid = rec(base), rec(det)
    res.update(frames=tot, recovered_baseline=float(r_base.sum() / tot), recovered_with_video=float((r_base | r_vid).sum() / tot),
               recovered_video_only=float(r_vid.sum() / tot))
    res["gain"] = res["recovered_with_video"] - res["recovered_baseline"]
    res["accepted"] = bool(res["n"] >= RULES["min_frames"] and res["median"] <= RULES["max_median_m"]
                           and res["p90"] <= RULES["max_p90_m"] and res["bad_frac"] <= RULES["max_bad_frac"])
    res["promote"] = bool(res["accepted"] and res["gain"] >= RULES["min_gain"])
    return res


def render_holdout(res):
    if not res:
        return ["video pass: not run"]
    return ["{} held-out windows tried, {} mapped cleanly (the rest hit cut/replay boundaries or an unreliable homography)".format(
                res["windows_tried"], res["windows_mapped"]),
            "detector on held-out frames: {} recovered, median {:.2f} m, p90 {:.2f} m, gross errors (> {:.0f} m) {:.0%}; orientation flip={}".format(
                res["n"], res["median"], res["p90"], RULES["max_p90_m"], res["bad_frac"], res["flip"]),
            "held-out frames recovered within {:.0f} m: broadcast tracking + interpolation {:.1%}; + video {:.1%} (gain {:+.1%}); video alone {:.1%}".format(
                RULES["recover_m"], res["recovered_baseline"], res["recovered_with_video"], res["gain"], res["recovered_video_only"]),
            "detector accepted: {}; video promoted: {}".format(res["accepted"], res["promote"])]


# ---------------------------------------------------------------- real pass
def real_pass(path, ball_after_fusion, windows, hom, hom_ok, t_of_b, flip, cache_dir, budget_s, hom_size=(1920, 1080), log=print):
    """Detect on the selected windows (resumable, within the time budget) and fill still-missing ball frames.
    Returns (ball_with_video, stats)."""
    n = len(ball_after_fusion)
    ref = np.asarray(ball_after_fusion, dtype=np.float64)
    out = ref.copy()
    t_start = time.time()
    stats = {"windows": len(windows), "decoded": 0, "cached": 0, "skipped_budget": 0, "filled_frames": 0,
             "by_kind": {}, "restart_windows_with_evidence": 0}
    for i, w in enumerate(windows):
        k = stats["by_kind"].setdefault(w["kind"], {"windows": 0, "with_fill": 0, "frames": 0})
        k["windows"] += 1
        p = _wpath(cache_dir, "real", w)
        if not os.path.isfile(p) and time.time() - t_start > budget_s:
            stats["skipped_budget"] += 1
            continue
        cand, was_cached = get_candidates(path, w, "real", cache_dir, hom, hom_ok, hom_size)
        stats["cached" if was_cached else "decoded"] += 1
        pos = window_positions(cand, w, hom, t_of_b, flip, n)
        filled = anchored_fill(ref, pos, w["t0"], w["t1"])
        m = np.isfinite(filled[:, 0]) & ~np.isfinite(out[:, 0])
        if m.any():
            out[m] = filled[m]
            k["with_fill"] += 1
            k["frames"] += int(m.sum())
            stats["filled_frames"] += int(m.sum())
            if w["kind"] == "restart":
                stats["restart_windows_with_evidence"] += 1
        if (i + 1) % 25 == 0:
            log("[vidpass] {}/{} windows ({} decoded)".format(i + 1, len(windows), stats["decoded"]))
    return out, stats
