"""Ball-recovery holdout: how accurately do broadcast tracking (and, later, video) recover tactical ball positions?

No possession labels are involved. We take stretches where the tactical ball was *reliably observed* (>= 95 % of the
stretch observed, outlier-free, with observed flanks), hide them, and measure how well the hidden positions are
recovered by
  * short-gap linear interpolation between the observed flanks (what the model does today, <= 2 s), and
  * the broadcast tracking, aligned on the MASKED tactical track (so the masked observations take no part in the
    alignment or in the gap filling; they are used only afterwards as truth),
  * their hybrid (broadcast where available, else usable interpolation).
Errors are reported per gap-length bin and per alignment-confidence tercile. The same masks are reused by the video
pass (poss/vidpass.py). The result also fixes two broadcast-fusion parameters by held-out ball error only:
the verify radius and the minimum gap length for which broadcast fills (shorter gaps stay with interpolation).
"""
import numpy as np

import poss  # noqa: F401
from poss import broadcast, features

PITCH = broadcast.PITCH
GAP_BINS = (10, 25, 50, 125, 250, 500)
RADII = (75, 150, 300)
# predeclared acceptance of broadcast ball recovery (ball position only, no possession labels)
RULES = {"min_frames": 300, "max_median_m": 2.5, "max_p90_m": 8.0}


def select_masks(ball_norm, blocks, per_bin=15, seed=0, flank=8, min_obs=0.95, spacing=300, bins=GAP_BINS):
    """Non-overlapping stretches (start, length) of reliably observed tactical ball inside the graded blocks."""
    clean = features.remove_ball_outliers(np.asarray(ball_norm, dtype=np.float64) * PITCH)
    obs = np.isfinite(clean[:, 0]).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(obs)))
    n = len(obs)
    rng = np.random.default_rng(seed)
    taken = np.zeros(n, dtype=bool)
    out = []
    for L in sorted(bins, reverse=True):
        valid = np.zeros(n, dtype=bool)
        for _, a, b in blocks:
            s = np.arange(a + flank, b + 1 - L - flank)
            if not len(s):
                continue
            frac = (cs[s + L] - cs[s]) / L
            fl = (cs[s] - cs[s - flank] == flank) & (cs[s + L + flank] - cs[s + L] == flank)
            valid[s[(frac >= min_obs) & fl]] = True
        cand = np.flatnonzero(valid)
        rng.shuffle(cand)
        got = 0
        for s in cand:
            if got >= per_bin:
                break
            lo, hi = max(s - spacing, 0), min(s + L + spacing, n)
            if taken[lo:hi].any():
                continue
            taken[s:s + L] = True
            out.append((int(s), int(L)))
            got += 1
    return sorted(out)


def apply_masks(ball_norm, stretches):
    masked = np.asarray(ball_norm, dtype=np.float64).copy()
    stretch_id = np.full(len(masked), -1, dtype=np.int64)
    for i, (s, L) in enumerate(stretches):
        masked[s:s + L] = np.nan
        stretch_id[s:s + L] = i
    return masked, stretch_id


def _err(pred_norm, truth_norm):
    return np.hypot(*((pred_norm - truth_norm) * PITCH).T)


def _summ(err):
    err = err[np.isfinite(err)]
    if not len(err):
        return {"n": 0, "median": float("nan"), "p90": float("nan"), "within2": float("nan")}
    return {"n": int(len(err)), "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
            "within2": float(np.mean(err <= 2.0))}


def evaluate(tact_ball_norm, bc_ball_norm, blocks, per_bin=15, seed=0, radii=RADII, log=print, extra_sources=None):
    """Run the holdout. Returns a dict with the masks, per-frame predictions and the summary tables.

    extra_sources: optional {name: array [n,2] (normalised, NaN = none)} of recoveries computed by the caller on the same
    masked track (the video pass); they get the same tables.
    """
    T = np.asarray(tact_ball_norm, dtype=np.float64)
    n = len(T)
    stretches = select_masks(T, blocks, per_bin=per_bin, seed=seed)
    Tm, sid = apply_masks(T, stretches)
    ev = sid >= 0
    Lof = np.zeros(n, dtype=np.int64)
    for s, L in stretches:
        Lof[s:s + L] = L
    ev &= np.isfinite(features.remove_ball_outliers(T * PITCH)[:, 0])          # truth frames only (reliable observations)
    truth = T
    run = broadcast.gap_lengths(Tm)                                            # actual gap run seen by the model

    clean_m = features.remove_ball_outliers(Tm * PITCH)
    ib, _, _ = features.interpolate_ball(clean_m, max_gap=10 ** 9)
    interp_all = ib / PITCH
    interp_all[~np.isfinite(clean_m[:, 0]) & (run == 0)] = np.nan
    usable = run <= features.MAX_INTERP_GAP                                     # the model interpolates only these
    interp = np.where(usable[:, None], interp_all, np.nan)

    nb = len(bc_ball_norm)
    al, flip = broadcast.best_orientation(Tm, bc_ball_norm, log=log)
    ok = broadcast.accept(al)
    conf_w = broadcast.window_confidence(al)
    bc_o = np.asarray(bc_ball_norm, dtype=np.float64)
    bc_o = 1.0 - bc_o if flip else bc_o
    t0, win_of_b = broadcast.frame_map(al, ok, nb, return_win=True)
    per_radius = {}
    tmaps = {}
    for r in radii:
        t_of_b = broadcast.verify_map(t0, Tm, bc_ball_norm, flip, radius=r)
        src = broadcast.source_frames(bc_o, t_of_b, n)
        pos = np.full((n, 2), np.nan)
        conf = np.full(n, np.nan)
        has = src >= 0
        pos[has] = bc_o[src[has]]
        conf[has] = conf_w[np.maximum(win_of_b[src[has]], 0)]
        per_radius[r] = {"pos": pos, "conf": conf}
        tmaps[r] = t_of_b

    def tables(pred, conf=None):
        rows = []
        for L in sorted(set(GAP_BINS) | {0}):
            sel = ev & ((Lof == L) if L else np.ones(n, dtype=bool))
            if not sel.any():
                continue
            have = np.isfinite(pred[sel, 0])
            e = _err(pred[sel][have], truth[sel][have])
            row = {"bin": L, "frames": int(sel.sum()), "coverage": float(have.mean())}
            row.update(_summ(e))
            rows.append(row)
        return rows

    res = {"stretches": stretches, "flip": flip, "windows_accepted": int(ok.sum()), "windows_total": int(len(ok)),
           "radii": {}, "interp": tables(interp), "interp_all": tables(interp_all)}
    for r, d in per_radius.items():
        pos = d["pos"]
        hyb = np.where(np.isfinite(pos[:, :1]), pos, interp)
        res["radii"][r] = {"broadcast": tables(pos), "hybrid": tables(hyb)}
        sel = ev & np.isfinite(pos[:, 0])
        res["radii"][r]["overall"] = _summ(_err(pos[sel], truth[sel]))
        res["radii"][r]["overall"]["coverage"] = float(sel.sum() / max(ev.sum(), 1))
        # error vs interpolation on frames where both exist, per bin
        both = {}
        for L in GAP_BINS:
            s_ = ev & (Lof == L) & np.isfinite(pos[:, 0]) & np.isfinite(interp[:, 0])
            if s_.sum() >= 30:
                both[L] = {"n": int(s_.sum()), "bc_median": float(np.median(_err(pos[s_], truth[s_]))),
                           "interp_median": float(np.median(_err(interp[s_], truth[s_])))}
        res["radii"][r]["both"] = both
        # by alignment confidence tercile
        c = d["conf"][sel]
        if len(c) >= 30:
            q = np.percentile(c, [33.3, 66.7])
            tt = []
            for name, m in (("low", c <= q[0]), ("mid", (c > q[0]) & (c <= q[1])), ("high", c > q[1])):
                e = _err(pos[sel][m], truth[sel][m])
                tt.append(dict(name=name, conf_range=(float(c[m].min()) if m.any() else float("nan"),
                                                     float(c[m].max()) if m.any() else float("nan")), **_summ(e)))
            res["radii"][r]["by_confidence"] = tt
    for name, pos in (extra_sources or {}).items():
        sel = ev & np.isfinite(pos[:, 0])
        res.setdefault("extra", {})[name] = {"tables": tables(pos), "overall": _summ(_err(pos[sel], truth[sel]))}
    res["n_eval_frames"] = int(ev.sum())
    res["_arrays"] = {"sid": sid, "ev": ev, "Lof": Lof, "Tm": Tm, "interp": interp, "run": run,
                      "bc_pos": {r: per_radius[r]["pos"] for r in per_radius},
                      "t_of_b": tmaps, "flip": flip}
    return res


def decide(res):
    """Predeclared choice of verify radius and min fill gap from held-out ball error only.
    Returns {accept, radius, min_gap, reasons}."""
    reasons, best = [], None
    for r, d in res["radii"].items():
        o = d["overall"]
        good = (o["n"] >= RULES["min_frames"] and o["median"] <= RULES["max_median_m"] and o["p90"] <= RULES["max_p90_m"])
        reasons.append("radius {}: recovered {} held-out frames ({:.1%} of masked), median {:.2f} m, p90 {:.2f} m -> {}".format(
            r, o["n"], o["coverage"], o["median"], o["p90"], "ok" if good else "fails"))
        if good and (best is None or o["coverage"] > res["radii"][best]["overall"]["coverage"]):
            best = r
    if best is None:
        return {"accept": False, "radius": RADII[0], "min_gap": features.MAX_INTERP_GAP + 1, "reasons": reasons}
    both = res["radii"][best]["both"]
    short = [L for L in GAP_BINS if L <= features.MAX_INTERP_GAP and L in both]
    short_ok = all(both[L]["bc_median"] <= both[L]["interp_median"] for L in short)
    min_gap = 0 if short_ok else features.MAX_INTERP_GAP + 1
    reasons.append("chosen radius {}; broadcast {} interpolation on short gaps -> fills gaps >= {} frames".format(
        best, "matches/beats" if short_ok else "does not beat", min_gap or 1))
    return {"accept": True, "radius": best, "min_gap": min_gap, "reasons": reasons}


def render(res, dec, title="Held-out ball recovery (position error, no possession labels)"):
    f = lambda v: "-" if v is None or not np.isfinite(v) else "{:.2f}".format(v)  # noqa: E731
    L = ["## " + title, "",
         "{} masked stretches, {} held-out frames; alignment (on the masked track): flip={}, {}/{} windows accepted".format(
             len(res["stretches"]), res["n_eval_frames"], res["flip"], res["windows_accepted"], res["windows_total"]), ""]
    for r in dec["reasons"]:
        L.append("- " + r)
    r = dec["radius"]
    d = res["radii"][r]
    L += ["", "### by gap length (verify radius {})".format(r), "",
          "| gap | frames | interp cover | interp med/p90 (m) | broadcast cover | broadcast med/p90 (m) | within 2 m | hybrid cover | hybrid med/p90 |",
          "|---|---|---|---|---|---|---|---|---|"]
    ib = {x["bin"]: x for x in res["interp"]}
    hb = {x["bin"]: x for x in d["hybrid"]}
    for x in d["broadcast"]:
        i, h = ib.get(x["bin"], {}), hb.get(x["bin"], {})
        L.append("| {} | {} | {:.0%} | {}/{} | {:.0%} | {}/{} | {} | {:.0%} | {}/{} |".format(
            x["bin"] or "all", x["frames"], i.get("coverage", 0), f(i.get("median")), f(i.get("p90")), x["coverage"],
            f(x["median"]), f(x["p90"]), "-" if not np.isfinite(x["within2"]) else "{:.0%}".format(x["within2"]),
            h.get("coverage", 0), f(h.get("median")), f(h.get("p90"))))
    if d.get("by_confidence"):
        L += ["", "### broadcast error by alignment confidence tercile", "", "| tercile | ratio range | frames | median | p90 | within 2 m |",
              "|---|---|---|---|---|---|"]
        for t in d["by_confidence"]:
            L.append("| {} | {}-{} | {} | {} | {} | {:.0%} |".format(t["name"], f(t["conf_range"][0]), f(t["conf_range"][1]),
                                                                  t["n"], f(t["median"]), f(t["p90"]), t["within2"]))
    for name, e in (res.get("extra") or {}).items():
        L += ["", "### " + name, "", "| gap | frames | coverage | median | p90 | within 2 m |", "|---|---|---|---|---|---|"]
        for x in e["tables"]:
            L.append("| {} | {} | {:.0%} | {} | {} | {:.0%} |".format(x["bin"] or "all", x["frames"], x["coverage"], f(x["median"]),
                                                                  f(x["p90"]), x["within2"] if np.isfinite(x["within2"]) else 0))
    return L
