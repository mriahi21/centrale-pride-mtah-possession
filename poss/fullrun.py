"""Full-bundle LOCAL workflow: inventory -> asset inspection -> signal verification -> candidate variants -> report.

    python -m poss.fullrun --data-root <DATA> --team-name "Centrale Pride" --out outputs\\full

Everything stays on your machine: the CSVs go to --out/<variant>/, the report to <data-root>/reports/. Nothing is
printed beyond aggregates, and none of it should be pasted into a hosted assistant. No labels are used anywhere; the
public ground truth is not read. Candidate selection is by predeclared verification + plausibility gates only.
"""
import argparse
import csv
import hashlib
import os
import sys
import time

import numpy as np

import poss  # noqa: F401
from evaluation import possession_metrics as pm
from games import GAME, GAMES, SOURCES, asset_paths
from poss import atd, bundle, gates, homog, holdout as holdout_mod, make_submission as ms, media, pipeline, signals, tags as tagmod, vidpass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_core_assets(data_root, data, tg, report):
    """Rows for assets that no signal stream reports on (tracking, tags, broadcast video)."""
    n = GAMES[GAME]["n_video_frames"]
    seen = float(np.mean(np.isfinite(data["ball"][:, 0])))
    slots = int(np.isfinite(data["players"][..., 0]).any(axis=0).sum())
    signals._row(report, "tactical/atd_raw", True, True,
                 "{} records read; ball visible on {:.1%} of frames; {} slots ever active".format(data["n_read"], seen, slots),
                 "core tracking")
    signals._row(report, "tactical/atd_meta", True, True, "2 teams, {} slots mapped".format(data["n_slots"]), "team per slot")
    na = sum(1 for t in tg if t["team"] is None)
    signals._row(report, "tactical/smart_tagging", True, bool(tg),
                 "{} instances; {} without team (N/A)".format(len(tg), na), "soft evidence / restart priors")
    bp = asset_paths(GAME, data_root, "broadcast")
    if os.path.isfile(bp["video"]):
        try:
            info = media.probe_video(bp["video"])
            nb = SOURCES["broadcast"]["n_video_frames"]
            signals._row(report, "broadcast/video", True, bool(info.get("ok")),
                         "frames {} (broadcast ATD has {}), {:.3f} fps, {}x{}, audio: {}".format(
                             info.get("frames"), nb, info.get("fps", 0), info.get("width"), info.get("height"),
                             info.get("audio_desc") or "none"),
                         "probed only; alignment is done in tracking space")
        except Exception as e:
            signals._row(report, "broadcast/video", True, False, "probe failed: {}".format(e), "probe", str(e)[:200])
    else:
        signals._row(report, "broadcast/video", False, False, "not present locally", "probe only",
                     "file not downloaded (14 GB; needs free disk space)")


def run_variant(name, profile, data_for, extras, team, blocks, n, tg, out_dir, team_name, log):
    halves = pipeline.prepare_halves(data_for, team, blocks, pipeline.KICKOFFS, extras)
    codes, info = pipeline.label_halves(halves, n, tags=tg, profile=profile, log=lambda *a: None)
    path = ms.write_csv(os.path.join(out_dir, name, team_name + ".csv"), codes)
    reason = ms.validate(path, n)
    ok, msgs, stats = gates.check(codes, blocks, pipeline.KICKOFFS)
    return {"name": name, "profile": profile, "codes": codes, "path": path, "format": reason or "OK",
            "sha": sha256(path), "gates_ok": ok, "gate_msgs": msgs, "stats": stats, "lambda": info["lambda"],
            "extras": sorted(extras)}


def run_video_pass(root, data, data_out, extras, team, blocks, n, tg, state, inv, args, sreport, log):
    """Targeted broadcast-video pass. Returns (result dict | None, lines). Everything is cached and resumable."""
    import json
    lines = []
    bp = asset_paths(GAME, root, "broadcast")
    row = next((r for r in inv if r["source"] == "broadcast" and r["asset"] == "video"), None)
    if args.no_video_pass:
        return None, ["video pass disabled (--no-video-pass)"]
    if not row or not row["size_ok"]:
        signals._row(sreport, "broadcast/video", row is not None and row["where"] != "missing", False,
                     "not complete locally", "targeted ball-detection windows", "download incomplete or missing (14 GB)")
        return None, ["broadcast video not complete locally: video pass skipped"]
    bc = state.get("broadcast")
    if not bc or "map" not in bc:
        return None, ["broadcast tracking alignment unavailable: video pass skipped"]
    cache_dir = state["cache_dir"]
    try:
        info = media.probe_video(bp["video"])
        hom = homog.load(bp["homography"])
        hok = homog.reliable(hom)
        hom_size = (info["width"], info["height"])
        A = np.load(os.path.join(cache_dir, "holdout_arrays.npz"))
        with open(os.path.join(cache_dir, "holdout_ball.json"), encoding="utf-8") as fh:
            stretches = [tuple(x) for x in json.load(fh)["stretches"]]
        masked, _ = holdout_mod.apply_masks(data["ball"], stretches)
        budget = args.video_budget_min * 60.0
        log("   video pass: held-out verification of the ball detector ...")
        hold = vidpass.video_holdout(bp["video"], hom, hok, data["ball"], masked, stretches, A["t_of_b"], bool(A["flip"]),
                                     A["bc_pos"].astype(np.float64), A["interp"].astype(np.float64), cache_dir,
                                     budget_s=0.25 * budget, hom_size=hom_size, log=log)
        lines += ["held-out verification of the video ball detector (hidden tactical ball, alignment from the masked track):"]
        lines += ["  " + x for x in vidpass.render_holdout(hold)]
        res = {"hold": hold}
        if not hold["accepted"]:
            lines.append("detector NOT accepted: the real pass was not run; broadcast tracking stays the only broadcast signal")
            signals._row(sreport, "broadcast/video", True, True, " ; ".join(lines[1:])[:900],
                         "targeted windows (held-out verification FAILED: video not used)")
            return res, lines
        ball0 = data_out["ball"]
        halves = pipeline.prepare_halves(data_out, team, blocks, pipeline.KICKOFFS, extras)
        codes, _ = pipeline.label_halves(halves, n, tags=tg, profile="v3", log=lambda *a: None)
        restarts, turnovers = vidpass.events_from_run(halves, codes)
        t_of_b = bc["map"]["t_of_b"]
        wins, rej = vidpass.select_windows(n, ball0, blocks, t_of_b, hok, restarts, turnovers, max_windows=args.video_windows)
        lines.append("windows selected {} ({}); rejected: {}".format(
            len(wins), ", ".join("{} {}".format(k, sum(1 for w in wins if w["kind"] == k)) for k in ("gap", "turnover", "restart")),
            dict(rej)))
        flip = bool(hold["flip"])
        ball_v, st = vidpass.real_pass(bp["video"], ball0, wins, hom, hok, t_of_b, flip, cache_dir, 0.75 * budget, hom_size, log)
        lines.append("decoded {} windows ({} from cache, {} left for the next run by the time budget); tactical ball frames filled by video {}; "
                     "restart windows with video evidence {}; by kind {}".format(
                         st["decoded"], st["cached"], st["skipped_budget"], st["filled_frames"],
                         st["restart_windows_with_evidence"], st["by_kind"]))
        dv = dict(data_out)
        dv["ball"] = ball_v.astype(np.float32)
        res.update(ball=dv, stats=st)
        signals._row(sreport, "broadcast/video", True, True, " ; ".join(lines[1:])[:900],
                     "targeted ball detections fill remaining tactical ball gaps"
                     + ("" if hold["promote"] else " (candidate only: no held-out gain)"))
        return res, lines
    except Exception as e:                                       # the tracking-only / broadcast-tracking candidates stay valid
        signals._row(sreport, "broadcast/video", True, False, "video pass failed: {}".format(e), "targeted windows", str(e)[:200])
        return None, ["video pass failed: {}".format(e)]


def write_reports(rep_dir, assets, variants, diffs, decision, applied):
    os.makedirs(rep_dir, exist_ok=True)
    with open(os.path.join(rep_dir, "asset_report.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["asset", "present", "size_ok", "inspected", "processed_ok", "detail", "used_for", "blocker"])
        w.writeheader()
        for r in assets:
            w.writerow(r)
    lines = ["# Full-bundle report (LOCAL ONLY -- do not paste into hosted tools or commit)", "",
             "## Assets", "", "| asset | present | size ok | inspected | processed | what was found | used for | blocker |",
             "|---|---|---|---|---|---|---|---|"]
    for r in assets:
        lines.append("| {asset} | {present} | {size_ok} | {inspected} | {processed_ok} | {detail} | {used_for} | {blocker} |".format(**r))
    lines += ["", "## Signals applied after verification: " + (", ".join(applied) or "none (tracking-only)"), "",
              "## Candidate variants", "", "| variant | profile | signals | format | gates | lambda | sha256 |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        lines.append("| {name} | {profile} | {sig} | {format} | {g} | {lambda:.1f} | {sha} |".format(
            sig=",".join(v["extras"]) or "-", g="PASS" if v["gates_ok"] else "FAIL: " + "; ".join(v["gate_msgs"]), **v))
    lines += ["", "### per-half aggregates", ""]
    for v in variants:
        for st in v["stats"]:
            lines.append("- {} half {half}: dead {dead_share:.1%}, {possessions} possessions, mean {mean_spell_s}s, top-team share {team_share}".format(v["name"], **st))
    lines += ["", "## How candidates differ (no ground truth; says nothing about which is right)", ""]
    lines += ["- " + d for d in diffs]
    lines += ["", "## Decision (verification + gates only; no accuracy claim)", ""] + ["- " + d for d in decision]
    with open(os.path.join(rep_dir, "full_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=os.environ.get("MTAH_DATA_DIR") or "data")
    ap.add_argument("--downloads", default=os.path.join(os.path.expanduser("~"), "Downloads"))
    ap.add_argument("--team-name", default="Centrale Pride")
    ap.add_argument("--out", default=os.path.join("outputs", "full"))
    ap.add_argument("--no-video-pass", action="store_true", help="skip the targeted broadcast-video ball pass")
    ap.add_argument("--video-budget-min", type=float, default=40.0, help="time budget (minutes) for decoding broadcast windows; resumable")
    ap.add_argument("--video-windows", type=int, default=300, help="max number of broadcast windows to detect on")
    ap.add_argument("--max-video-frames", type=int, default=None, help="smoke test: only decode this many frames")
    args = ap.parse_args(argv)
    t0 = time.time()
    root = args.data_root
    n = GAMES[GAME]["n_video_frames"]
    blocks = pipeline.default_blocks()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    log("== 1/5 inventory")
    still = bundle.in_progress(args.downloads)
    if still:
        log("   downloads still in progress (skipped): {}".format(still))
    bundle.adopt_downloads(root, args.downloads, log=log)
    inv = bundle.rows(root, args.downloads)
    for r in inv:
        log("   {:<10} {:<14} {:<10} size_ok={}".format(r["source"], r["asset"], r["where"], r["size_ok"]))

    meta, raw = asset_paths(GAME, root)["atd_meta"], asset_paths(GAME, root)["atd_raw"]
    log("== 2/5 tracking + tags")
    data = atd.read_atd(raw, n)
    team = atd.team_of_slots(meta)
    tg = tagmod.load_tags(root, log=lambda *a: None)

    state = {}
    pre = {}
    for name in ("v1", "v2"):       # tracking-only candidates; v2 also provides reference masks for the video/audio cue checks
        pre[name] = run_variant(name, name, data, {}, team, blocks, n, tg, args.out, args.team_name, log)
        log("   {:<18} format={} gates={} sha256={}".format(name, pre[name]["format"], "PASS" if pre[name]["gates_ok"] else "FAIL", pre[name]["sha"][:16]))
    ref = ms.reference_masks(data, team, blocks, n, pre["v2"]["codes"])

    log("== 3/5 signals (broadcast alignment + held-out ball check, homography, video/audio checks) -- this is the long step")
    data_out, extras, applied, sreport = signals.collect(root, data, blocks, n, log=log, max_frames=args.max_video_frames, ref=ref, state=state)
    inspect_core_assets(root, data, tg, sreport)

    log("== 4/5 candidate variants")
    base = {"v1": ("v1", data, {}), "v2": ("v2", data, {})}
    variants_spec = dict(base)
    ball_for = lambda use: data_out if use else data  # noqa: E731
    sub = lambda keys: {k: v for k, v in extras.items() if k in keys}  # noqa: E731
    has_b = "broadcast" in applied
    variants_spec["v3_final"] = ("v3", ball_for(True), sub(extras))
    if has_b:
        variants_spec["v3_no_broadcast"] = ("v3", data, sub(extras))
        variants_spec["v3_broadcast_only"] = ("v3", data_out, {})
    for key, label in (("vquiet", "video"), ("whistle", "audio"), ("hom_ok", "homography")):
        if key in extras:
            variants_spec["v3_no_" + label] = ("v3", ball_for(True), sub([k for k in extras if k != key]))
    if "vquiet" in extras and "hom_ok" in extras:
        variants_spec["v3_video_only"] = ("v3", data, sub(["vquiet", "hom_ok"]))
    if "whistle" in extras:
        variants_spec["v3_audio_only"] = ("v3", data, sub(["whistle"]))
    results = []
    for name, (profile, dfor, ex) in variants_spec.items():
        if name in pre:
            results.append(pre[name])
            continue
        r = run_variant(name, profile, dfor, ex, team, blocks, n, tg, args.out, args.team_name, log)
        results.append(r)
        log("   {:<18} format={} gates={} sha256={}".format(name, r["format"], "PASS" if r["gates_ok"] else "FAIL", r["sha"][:16]))

    log("== 4b/5 broadcast video pass (targeted windows; resumable)")
    vres, vlines = run_video_pass(root, data, data_out, extras, team, blocks, n, tg, state, inv, args, sreport, log)
    video_variant = None
    if vres and vres.get("ball") is not None:
        video_variant = "v3_vidpass"
        r = run_variant(video_variant, "v3", vres["ball"], sub(extras), team, blocks, n, tg, args.out, args.team_name, log)
        results.append(r)
        log("   {:<18} format={} gates={} sha256={}".format(video_variant, r["format"], "PASS" if r["gates_ok"] else "FAIL", r["sha"][:16]))
    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, "reports", "video_pass.md"), "w", encoding="utf-8") as fh:
        fh.write("# Broadcast video pass (LOCAL ONLY)\n\n" + "\n".join("- " + x for x in vlines) + "\n")

    log("== 5/5 report")
    graded = np.zeros(n, dtype=bool)
    runs = []
    for _, a, b in blocks:
        graded[a:b + 1] = True
        runs.append((a, b + 1))
    by = {r["name"]: r for r in results}
    diffs = []
    ref = by["v2"]["codes"]
    for r in results:
        if r["name"] in ("v1", "v2"):
            continue
        agree = float(np.mean(r["codes"][graded] == ref[graded]))
        sp = pm.spell_scores(ref, r["codes"], runs)["mean_f1"]
        toggled_d = int(np.sum((r["codes"][graded] == pm.D) != (ref[graded] == pm.D)))
        diffs.append("{} vs v2: frame agreement {:.2%}, possession-spell F1 {:.3f}, frames whose dead/alive status differs {}".format(
            r["name"], agree, sp, toggled_d))
    agree = float(np.mean(by["v1"]["codes"][graded] == ref[graded]))
    diffs.append("v1 vs v2: frame agreement {:.2%}, possession-spell F1 {:.3f}".format(
        agree, pm.spell_scores(by["v1"]["codes"], ref, runs)["mean_f1"]))

    decision = ["signals that passed verification: {}".format(", ".join(applied) or "none")]
    for r in sreport:
        decision.append("{} -> {}".format(r["asset"], r["detail"][:240]))
    f = by["v3_final"]
    if f["format"] == "OK" and f["gates_ok"]:
        rec = "v3_final" if applied else "v2"
    elif by["v2"]["format"] == "OK" and by["v2"]["gates_ok"]:
        rec = "v2"
    else:
        rec = "v1"
    for x in vlines:
        decision.append("video pass: " + x)
    if video_variant and vres["hold"]["promote"]:
        v = by[video_variant]
        if v["format"] == "OK" and v["gates_ok"] and rec == "v3_final":
            rec = video_variant
            decision.append("video pass PROMOTED: held-out ball recovery improved (ball position only; not a possession-accuracy claim)")
    elif video_variant:
        decision.append("video pass NOT promoted: no held-out recovery gain over broadcast tracking + interpolation; v3_vidpass is only a comparison file")
    decision.append("RECOMMENDED CANDIDATE (by verification + plausibility gates only): {} -> {}".format(rec, by[rec]["path"]))
    decision.append("an accuracy claim would need independent labels, which are unavailable; keep frozen-v1 as the validated fallback")

    asset_rows = []
    srep = {r["asset"]: r for r in sreport}
    for r in inv:
        key = "{}/{}".format(r["source"], r["asset"])
        info = srep.get(key) or next((v for k, v in srep.items() if k.startswith(r["source"] + "/" + r["asset"].split("_")[0])
                                      or (r["asset"] in ("atd_raw", "atd_meta") and k.startswith(r["source"] + "/atd_raw+atd_meta"))), None)
        asset_rows.append({"asset": key, "present": r["where"], "size_ok": r["size_ok"],
                           "inspected": info["inspected"] if info else False, "processed_ok": info["processed_ok"] if info else False,
                           "detail": info["detail"] if info else "not processed", "used_for": r["use"],
                           "blocker": (info["blocker"] if info else "") or ("" if r["size_ok"] else "missing or incomplete")})
    for r in sreport:   # audio has no file of its own
        if r["asset"].endswith("/audio"):
            asset_rows.append({"asset": r["asset"], "present": "in tactical video", "size_ok": "n/a", "inspected": r["inspected"],
                               "processed_ok": r["processed_ok"], "detail": r["detail"], "used_for": r["used_for"], "blocker": r["blocker"]})
    rep_dir = os.path.join(root, "reports")
    write_reports(rep_dir, asset_rows, results, diffs, decision, applied)
    log("   report: {}".format(os.path.join(rep_dir, "full_report.md")))
    log("   RECOMMENDED: {}  ({:.0f} s total)".format(rec, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
