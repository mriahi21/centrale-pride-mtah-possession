"""Optional signal streams for profile v3, each verified before it may influence a label.

collect(...) builds, per stream, the signal AND a verification verdict; a stream that fails verification (or whose
input is missing/corrupt) is left out and the reason is recorded. The tracking-only model therefore always remains
runnable and is the fallback everywhere a stream is absent or unverified.

Streams: broadcast ball fusion, tactical-video motion energy, tactical-audio whistles, homography reliability.
All processing is local; arrays are cached under <data_root>/cache (never in git). Nothing here uses labels.
"""
import json
import os

import numpy as np

import poss  # noqa: F401
from games import GAME, SOURCES, asset_paths
from poss import atd, audio, broadcast, features, holdout, homog, media, video_activity, videocheck

# Predeclared acceptance rules (verification only; not accuracy)
RULES = {
    "broadcast_min_mapped_frac": 0.15,
    "broadcast_max_median_err_m": 2.0,
    "broadcast_max_p90_err_m": 6.0,
    "video_max_abs_lag_frames": 12,
    "video_min_corr": 0.4,
    "video_frame_tolerance": 2,         # |frames - 199319|
    "audio_events_range": (30, 700),
    "audio_max_post_pre_ratio": 0.97,   # tracking activity should drop after whistles
}


def poss_runs(mask):
    a = np.concatenate(([0], mask.astype(np.int8), [0]))
    e = np.diff(a)
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def _cache(cache_dir, name):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def _row(report, asset, inspected, ok, detail, used, blocker=""):
    report.append({"asset": asset, "inspected": inspected, "processed_ok": ok, "detail": detail,
                   "used_for": used, "blocker": blocker})


def tracking_activity(data, n_frames):
    return features.player_activity(data["pspeed"], None)


def holdout_result(data, bc_ball, blocks, cache_dir, log):
    """Held-out ball-recovery study (cached): decision + rendered report lines + the arrays the video pass reuses
    (alignment on the MASKED track, broadcast-tracking recovery, interpolation). See poss/holdout.py."""
    hpath, apath = _cache(cache_dir, "holdout_ball.json"), _cache(cache_dir, "holdout_arrays.npz")
    if os.path.isfile(hpath) and os.path.isfile(apath):
        with open(hpath, encoding="utf-8") as fh:
            return json.load(fh)
    res = holdout.evaluate(data["ball"], bc_ball, blocks, log=log)
    dec = holdout.decide(res)
    A = res["_arrays"]
    r = dec["radius"]
    np.savez_compressed(apath, t_of_b=A["t_of_b"][r], flip=bool(A["flip"]), bc_pos=A["bc_pos"][r].astype(np.float32),
                        interp=A["interp"].astype(np.float32))
    H = {"dec": dec, "lines": holdout.render(res, dec), "stretches": res["stretches"]}
    with open(hpath, "w", encoding="utf-8") as fh:
        json.dump(H, fh)
    return H


def broadcast_stream(data_root, data, n_frames, cache_dir, report, log, blocks=None):
    paths = asset_paths(GAME, data_root, "broadcast")
    for k in ("atd_raw", "atd_meta"):
        if not os.path.isfile(paths[k]):
            _row(report, "broadcast/" + k, False, False, "file missing", "ball fusion", "file not present locally")
            return None
    nb = SOURCES["broadcast"]["n_video_frames"]
    bc = atd.read_atd(paths["atd_raw"], nb)
    dec = {"accept": True, "radius": 75, "min_gap": 0, "reasons": []}
    if blocks is not None:
        H = holdout_result(data, bc["ball"], blocks, cache_dir, log)
        dec = H["dec"]
        os.makedirs(os.path.join(data_root, "reports"), exist_ok=True)
        with open(os.path.join(data_root, "reports", "holdout_ball.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(H["lines"]) + "\n")
    cpath = _cache(cache_dir, "broadcast_align.npz")
    if os.path.isfile(cpath):
        z = np.load(cpath)
        t0, flip = z["t0"], bool(z["flip"])
    else:
        al, flip = broadcast.best_orientation(data["ball"], bc["ball"], log=log)
        t0 = broadcast.frame_map(al, broadcast.accept(al), nb)
        np.savez_compressed(cpath, t0=t0, flip=flip)
    t_of_b = broadcast.verify_map(t0, data["ball"], bc["ball"], flip, radius=dec["radius"])
    fused, filled, stats = broadcast.fuse_ball(data["ball"], bc["ball"], t_of_b, flip, n_frames, min_gap=dec["min_gap"])
    passed = bool(dec["accept"] and stats["broadcast_mapped_frac"] >= RULES["broadcast_min_mapped_frac"]
                  and np.isfinite(stats["agreement_median_m"])
                  and stats["agreement_median_m"] <= RULES["broadcast_max_median_err_m"]
                  and stats["agreement_p90_m"] <= RULES["broadcast_max_p90_err_m"])
    detail = ("flip={}; mapped {:.1%} of broadcast frames; tactical ball gaps filled {} (min gap {}); agreement median {:.2f} m, "
              "p90 {:.2f} m; held-out recovery check {}; rules {}").format(
        flip, stats["broadcast_mapped_frac"], stats["tactical_ball_gaps_filled"], dec["min_gap"] or 1,
        stats["agreement_median_m"], stats["agreement_p90_m"], "ACCEPT" if dec["accept"] else "REJECT",
        "PASS" if passed else "FAIL")
    _row(report, "broadcast/atd_raw+atd_meta", True, True, detail,
         "aligned broadcast ball fills tactical gaps" + ("" if passed else " (NOT applied: verification failed)"))
    map_state = {"t_of_b": t_of_b, "flip": flip, "bc_ball": bc["ball"]}
    # broadcast homography: validity, cut/replay stretches, and how they relate to the alignment
    hp = paths["homography"]
    if os.path.isfile(hp):
        try:
            bh = homog.load(hp)
            valid = bh["valid"][:nb]
            inv_runs = [(a, b) for a, b in poss_runs(~valid)]
            longest = max((b - a for a, b in inv_runs), default=0)
            m = t_of_b >= 0
            n_ = min(len(valid), len(m))
            both = "mapped frames with valid broadcast homography {:.1%}; unmapped frames without a valid homography {:.1%}".format(
                float(valid[:n_][m[:n_]].mean()) if m[:n_].any() else float("nan"),
                float((~valid[:n_][~m[:n_]]).mean()) if (~m[:n_]).any() else float("nan"))
            _row(report, "broadcast/homography", True, True,
                 "rows {}; valid {:.1%}; {} invalid stretches (longest {} frames = cuts/replays/close-ups); {}".format(
                     len(valid), valid.mean(), len(inv_runs), longest, both),
                 "explains unmapped broadcast stretches (not fed to the model)")
        except Exception as e:
            _row(report, "broadcast/homography", True, False, "failed: {}".format(e), "alignment report", str(e)[:200])
    return {"ball": fused if passed else None, "passed": passed, "stats": stats, "map": map_state, "holdout": dec}


def homography_stream(data_root, data, n_frames, report, log):
    paths = asset_paths(GAME, data_root, "tactical")
    if not os.path.isfile(paths["homography"]):
        _row(report, "tactical/homography", False, False, "file missing", "camera reliability / video mask",
             "file not present locally")
        return None
    hom = homog.load(paths["homography"])
    ok = homog.reliable(hom)
    cons = homog.consistency(hom, data["players"])
    inside = 100 * cons["inside_frac"] if np.isfinite(cons["inside_frac"]) else float("nan")
    detail = ("rows {}; valid {:.1%}; reliable (err<={}) {:.1%}; tracking projects inside the image for {:.1f}% of "
              "players ({} frames checked)").format(len(ok), hom["valid"].mean(), homog.ERR_OK, ok.mean(), inside,
                                                    cons["frames_checked"])
    _row(report, "tactical/homography", True, True, detail, "reliability gate + pitch mask for the video cue")
    return {"hom": hom, "ok": ok, "consistency": cons}


def video_stream(data_root, data, n_frames, blocks, hom_res, cache_dir, report, log, max_frames=None, ref=None):
    paths = asset_paths(GAME, data_root, "tactical")
    vpath = paths["video"]
    if not os.path.isfile(vpath):
        _row(report, "tactical/video", False, False, "file missing", "video motion cue / audio", "file not present locally")
        return None, None
    info = media.probe_video(vpath)
    if not info.get("ok"):
        _row(report, "tactical/video", True, False, "unreadable: " + str(info.get("error")),
             "video motion cue / audio", "cannot open")
        return None, None
    frame_ok = abs(info["frames"] - 199319) <= RULES["video_frame_tolerance"]
    detail = "frames {} (expected 199319, {}), {:.3f} fps, {}x{}, audio: {}".format(
        info["frames"], "OK" if frame_ok else "MISMATCH", info["fps"], info["width"], info["height"],
        info["audio_desc"] or "none")
    series, v_ok = None, False
    if frame_ok and hom_res is not None:
        cpath = _cache(cache_dir, "video_activity.npz")
        if os.path.isfile(cpath):
            z = np.load(cpath)
            idx, act = z["idx"], z["act"]
        else:
            poly = None
            for f in range(blocks[0][1] + 1500, blocks[0][2], 500):    # a reliable frame inside the first half
                if hom_res["ok"][f]:
                    poly = homog.pitch_polygon(hom_res["hom"], f)
                    break
            mask = video_activity.pitch_mask(poly, (info["width"], info["height"]))
            idx, act = video_activity.video_activity(vpath, mask, max_frames=max_frames, log=log)
            if max_frames is None:
                np.savez_compressed(cpath, idx=idx, act=act)
        series = video_activity.to_frame_series(idx, act, n_frames)
        graded = np.zeros(n_frames, dtype=bool)
        for _, a, b in blocks:
            graded[a:b + 1] = True
        ta = tracking_activity(data, n_frames)
        if ref is not None:      # camera-motion gating, clock per block, time-shifted controls, incremental evidence
            vc = videocheck.verify_video(series, ta, hom_res["hom"], hom_res["ok"], graded, ref)
            v_ok = vc["verdict"]
            if v_ok:
                series = vc["gated"]
            detail += "; " + " ; ".join(vc["lines"]) + (" -> VERIFIED" if v_ok else " -> NOT verified (cue disabled)")
        else:
            va = np.where(graded & hom_res["ok"], series, np.nan)
            lag, rb, r0 = video_activity.lag_correlation(va, np.where(graded, ta, np.nan), max_lag=100, step=5)
            v_ok = bool(abs(lag) <= RULES["video_max_abs_lag_frames"] and rb >= RULES["video_min_corr"])
            detail += "; motion energy vs tracking activity: best lag {} frames, r={:.2f} -> {}".format(
                lag, rb, "ALIGNED" if v_ok else "NOT verified (cue disabled)")
    else:
        detail += "; video cue skipped ({})".format("frame-count mismatch" if not frame_ok else "no homography")
    _row(report, "tactical/video", True, True, detail,
         "pitch motion energy cue (weak, corroborating)" + ("" if v_ok else " [not applied]"))
    w_cue = None
    if info.get("has_audio"):
        try:
            x = audio.decode_audio(vpath, max_seconds=None if max_frames is None else max_frames / 25.0)
            ev = audio.whistle_events(x)
            lo, hi = RULES["audio_events_range"]
            ta = tracking_activity(data, n_frames)
            frames = [int(round((a + info.get("audio_start_s", 0.0)) * 25)) for a, _, _ in ev]
            graded = np.zeros(n_frames, dtype=bool)
            for _, a, b in blocks:
                graded[a:b + 1] = True
            if ref is not None:
                d = np.asarray(ref["dead"], dtype=bool)
                dstarts = np.flatnonzero(d[1:] & ~d[:-1]) + 1
                dstarts = dstarts[graded[dstarts]]
                vw = videocheck.verify_whistle(frames, ta, graded, dstarts, n_frames)
                a_ok = bool(lo <= len(ev) <= hi and vw["verdict"])
                txt = " ; ".join(vw["lines"])
            else:
                ratios = []
                for f in frames:
                    if 150 < f < n_frames - 200:
                        pre, post = np.nanmean(ta[f - 125:f]), np.nanmean(ta[f + 25:f + 150])
                        if np.isfinite(pre) and np.isfinite(post) and pre > 0:
                            ratios.append(post / pre)
                rmed = float(np.median(ratios)) if ratios else float("nan")
                a_ok = bool(lo <= len(ev) <= hi and np.isfinite(rmed) and rmed <= RULES["audio_max_post_pre_ratio"])
                txt = "median post/pre tracking activity {:.2f}".format(rmed)
            _row(report, "tactical/audio", True, True,
                 "{}; {} whistle-like events; {} -> {}".format(info["audio_desc"], len(ev), txt,
                                                              "VERIFIED" if a_ok else "NOT verified (cue disabled)"),
                 "whistle cue (weak, corroborating)" + ("" if a_ok else " [not applied]"))
            if a_ok:
                w_cue = audio.whistle_cue(n_frames, ev, offset_s=info.get("audio_start_s", 0.0))
        except Exception as e:                                   # keep the tracking-only path alive
            _row(report, "tactical/audio", True, False, "audio processing failed: {}".format(e), "whistle cue",
                 str(e)[:200])
    else:
        _row(report, "tactical/audio", True, True, "no audio stream in the tactical video", "whistle cue",
             "no audio track (cue unavailable)")
    return (series if v_ok else None), w_cue


def collect(data_root, data, blocks, n_frames=199319, enable=("broadcast", "video", "audio", "homography"),
            cache_dir=None, log=print, max_frames=None, ref=None, state=None):
    """Return (data_out, extras, applied, report). ref = {'oob','dead'} tracking-derived reference masks for the cue checks. data_out has the (possibly fused) ball."""
    cache_dir = cache_dir or os.path.join(data_root, "cache")
    report, extras, applied = [], {}, []
    data_out = dict(data)
    if "broadcast" in enable:
        bc = None
        try:
            bc = broadcast_stream(data_root, data, n_frames, cache_dir, report, log, blocks)
        except Exception as e:
            _row(report, "broadcast", True, False, "failed: {}".format(e), "ball fusion", str(e)[:200])
        if state is not None and bc:
            state["broadcast"] = bc
            state["cache_dir"] = cache_dir
        if bc and bc["ball"] is not None:
            data_out["ball"] = bc["ball"]
            applied.append("broadcast")
    hom_res = None
    if "homography" in enable or "video" in enable:
        try:
            hom_res = homography_stream(data_root, data, n_frames, report, log)
        except Exception as e:
            _row(report, "tactical/homography", True, False, "failed: {}".format(e), "reliability", str(e)[:200])
        if hom_res is not None and "homography" in enable:
            extras["hom_ok"] = hom_res["ok"]
            applied.append("homography")
    if "video" in enable or "audio" in enable:
        try:
            vseries, wcue = video_stream(data_root, data, n_frames, blocks, hom_res if "video" in enable else None,
                                         cache_dir, report, log, max_frames, ref)
        except Exception as e:
            vseries, wcue = None, None
            _row(report, "tactical/video", True, False, "failed: {}".format(e), "video/audio cues", str(e)[:200])
        if vseries is not None and "video" in enable:
            extras["vquiet"] = vseries
            applied.append("video")
        if wcue is not None and "audio" in enable:
            extras["whistle"] = wcue
            applied.append("audio")
    return data_out, extras, applied, report
