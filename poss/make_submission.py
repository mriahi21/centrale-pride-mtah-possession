"""One command: data bundle -> <TEAM_NAME>.csv (+ official validation).

    python -m poss.make_submission --team-name MyTeam            # data from ./data or $MTAH_DATA_DIR
    python -m poss.make_submission --team-name MyTeam --baseline # official baseline_smooth50 fallback

Everything runs locally; the only outputs are the CSV and console diagnostics (counts/shares, no per-frame data).
The run is deterministic (no randomness): same bundle + same code -> same file.
"""
import argparse
import os
import sys

import numpy as np

import poss  # noqa: F401
from evaluation import possession_metrics as pm
from games import GAME, GAMES, DATA_DIR, require, source_frames
from poss import atd, gates, pipeline, signals, tags as tagmod

LETTERS = np.array(["H", "A", "D", "X"])


def write_csv(path, codes):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes))
    return path


def validate(path, n_rows):
    """Same checks as evaluation/validate_submission.py (it calls these very functions)."""
    table = pm.read_label_csv(path)
    return pm.validate_submission(table, n_rows)


def summarize(codes, blocks, log=print):
    """Console-only sanity summary with football-plausibility warnings."""
    warn = []
    for half, a, b in blocks:
        c = codes[a:b + 1]
        runs = [(0, len(c))]
        spells = pm.possession_spells(c, runs)
        n_alive = int(np.sum((c == pm.H) | (c == pm.A)))
        share_d = float(np.mean(c == pm.D))
        mean_s = n_alive / 25.0 / max(len(spells), 1)
        log("[summary] half {}: H {:.1%} A {:.1%} D {:.1%}; {} possessions, mean {:.1f}s".format(
            half, np.mean(c == pm.H), np.mean(c == pm.A), share_d, len(spells), mean_s))
        if not (0.05 <= share_d <= 0.6):
            warn.append("half {}: dead share {:.0%} looks implausible".format(half, share_d))
        if not (4.0 <= mean_s <= 40.0):
            warn.append("half {}: mean possession {:.1f}s looks implausible".format(half, mean_s))
    for w in warn:
        log("[WARN] " + w)
    return warn


def build_baseline(data_dir, log=print):
    from starter import baseline
    meta, raw = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n = source_frames("tactical", GAME)
    c = baseline.nearest_player_labels(raw, baseline.team_of_each_track(meta), n, GAMES[GAME]["pitch"])
    labels = baseline.smooth(c, 50)
    codes = np.array([{"H": 0, "A": 1, "D": 2}[x] for x in labels], dtype=np.int8)
    out = np.full(n, 3, dtype=np.int8)
    for _, a, b in pipeline.default_blocks():
        out[a:b + 1] = codes[a:b + 1]
    return out


def reference_masks(data, team, blocks, n, v2_codes):
    """Tracking-only reference masks for the video/audio cue checks: ball out-of-bounds and the stoppages of the
    tracking-only v2 model. Shared by fullrun and generate_v3 so both produce identical candidates."""
    oob = np.zeros(n, dtype=bool)
    for h in pipeline.prepare_halves(data, team, blocks, pipeline.KICKOFFS):
        oob[h["a"]:h["b"] + 1] = h["ft"]["oob"]
    return {"oob": oob, "dead": np.asarray(v2_codes) == pm.D}


def generate_v3(raw, meta, data_dir, tg, disable=(), lam=None, use_dead=True, log=print, video_pass=False,
                video_budget_min=40.0, video_windows=300):
    """Profile v3 = v2 + optional verified signals (broadcast ball fusion, video motion, audio whistles, homography).
    Each stream is applied only if it passes its predeclared verification; otherwise tracking-only evidence stands.
    video_pass=True additionally runs the targeted broadcast-video ball pass (promoted only if verified on held-out
    ball positions; see poss/vidpass.py). Same code path as poss.fullrun's v3_final / v3_vidpass."""
    from types import SimpleNamespace
    from poss import bundle, fullrun
    n = GAMES[GAME]["n_video_frames"]
    blocks = pipeline.default_blocks()
    data = atd.read_atd(raw, n)
    team = atd.team_of_slots(meta)
    enable = tuple(k for k in ("broadcast", "video", "audio", "homography") if k not in disable)
    h2 = pipeline.prepare_halves(data, team, blocks, pipeline.KICKOFFS)
    v2_codes, _ = pipeline.label_halves(h2, n, tags=tg, profile="v2", log=lambda *a: None)
    ref = reference_masks(data, team, blocks, n, v2_codes)
    state = {}
    data_out, extras, applied, report = signals.collect(data_dir, data, blocks, n, enable=enable, log=log, ref=ref, state=state)
    for r in report:
        log("[signals] {}: {}".format(r["asset"], r["detail"]))
    log("[signals] applied: {}".format(applied or "none (tracking-only)"))
    if video_pass and "broadcast" in applied:
        args = SimpleNamespace(no_video_pass=False, video_budget_min=video_budget_min, video_windows=video_windows)
        res, lines = fullrun.run_video_pass(data_dir, data, data_out, extras, team, blocks, n, tg, state,
                                            bundle.rows(data_dir), args, [], log)
        for x in lines:
            log("[video pass] " + x)
        if res and res.get("ball") is not None and res["hold"]["promote"]:
            data_out = res["ball"]
            log("[video pass] PROMOTED (held-out ball recovery improved)")
        else:
            log("[video pass] not promoted: keeping broadcast tracking only")
    halves = pipeline.prepare_halves(data_out, team, blocks, pipeline.KICKOFFS, extras)
    codes, _ = pipeline.label_halves(halves, n, tags=tg, use_dead=use_dead, lam=lam, profile="v3", log=log)
    return codes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-name", default="Centrale Pride", help="used for the file name <team>.csv")
    ap.add_argument("--data-dir", default=None, help="bundle root (default $MTAH_DATA_DIR or ./data)")
    ap.add_argument("--out", default="outputs", help="output folder (gitignored)")
    ap.add_argument("--baseline", action="store_true", help="write the official baseline_smooth50 instead")
    ap.add_argument("--profile", choices=pipeline.PROFILES, default="v1", help="v1 = frozen validated candidate (default)")
    ap.add_argument("--disable", nargs="*", default=[], choices=["broadcast", "video", "audio", "homography"],
                    help="profile v3 only: leave these optional signal streams out (ablation / fallback)")
    ap.add_argument("--video-pass", action="store_true", help="profile v3 only: add the targeted broadcast-video ball pass (verified, resumable)")
    ap.add_argument("--video-budget-min", type=float, default=40.0)
    ap.add_argument("--video-windows", type=int, default=300)
    ap.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    ap.add_argument("--no-dead", action="store_true", help="only pre-kickoff dead ball (ablation)")
    ap.add_argument("--lambda", dest="lam", type=float, default=None, help="fix the switch penalty (default: prior-based)")
    args = ap.parse_args(argv)

    n_rows = GAMES[GAME]["n_video_frames"]
    data_dir = args.data_dir or DATA_DIR
    path = os.path.join(args.out, "{}.csv".format(args.team_name))

    if args.baseline:
        codes = build_baseline(data_dir)
    else:
        meta, raw = require(["atd_meta", "atd_raw"], GAME, data_dir)
        tg = [] if args.no_tags else tagmod.load_tags(data_dir)
        if args.profile == "v3":
            codes = generate_v3(raw, meta, data_dir, tg, args.disable, args.lam, not args.no_dead, video_pass=args.video_pass,
                                video_budget_min=args.video_budget_min, video_windows=args.video_windows)
        else:
            codes, _ = pipeline.generate_labels(raw, meta, tags=tg, use_dead=not args.no_dead, lam=args.lam, profile=args.profile)

    summarize(codes, pipeline.default_blocks())
    ok, msgs, _ = gates.check(codes, pipeline.default_blocks(), pipeline.KICKOFFS)
    print("[gates] plausibility gates (not accuracy): {}".format("PASS" if ok else "FAIL"))
    for m in msgs:
        print("[gates]   - " + m)
    write_csv(path, codes)
    reason = validate(path, n_rows)
    if reason:
        print("VALIDATION FAILED: {}".format(reason))
        return 1
    print("wrote {}  ({} rows) -- internal validation OK".format(path, n_rows))
    print("now run:  python evaluation/validate_submission.py {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
