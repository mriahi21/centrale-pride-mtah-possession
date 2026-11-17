"""Automated LOCAL decision summary for the final candidate.

    python scripts/decide_candidate.py --data-root <DATA>

Reads the reports written by poss.fullrun (full_report.md, video_pass.md, holdout_ball.md, asset_report.csv, cache/holdout_ball.json),
checks they come from the current code, applies the predeclared selection rules and writes
  <data-root>/reports/decision.md          full local detail (numbers, hashes, paths) -- keep local
  <data-root>/reports/decision_status.txt  coarse status only: yes/no flags, candidate name, flags -- no numbers, no hashes
Rules (verification and plausibility only; better ball recovery is NOT an accuracy claim for full-match spellF1):
  v3_vidpass  only if: its CSV is format-valid and passes the gates, the real video pass ran completely, the held-out detector
              was accepted AND promoted (error limits + recovery gain beyond broadcast tracking + interpolation), its gross
              error rate on held-out frames is <= 10 %, >= 500 held-out frames were recovered, and it filled frames in the
              real windows.
  v3_final    if valid + gates PASS and at least one v3 signal was verified (broadcast tracking / cues); otherwise
  v2          if valid + gates PASS; otherwise the frozen v1 file.
No labels and no public-sample scores are read.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_GROSS = 0.10
MIN_HELD = 500


def read(path):
    return open(path, encoding="utf-8").read() if os.path.isfile(path) else ""


def code_matches(run_commit):
    """True if poss/, games.py and evaluation/ are identical between the run's commit and the working tree."""
    if not run_commit:
        return None
    try:
        r = subprocess.run(["git", "diff", "--quiet", run_commit, "--", "poss", "games.py", "evaluation"], cwd=ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "poss", "games.py", "evaluation"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return r.returncode == 0 and not dirty
    except Exception:
        return None


def gate_categories(cell):
    """Coarse failure categories (names + direction only; no numbers) from a report's 'gates'/'format' cell."""
    out = []
    for msg in re.split(r";\s*", cell.replace("FAIL:", "")):
        m = re.search(r"dead share (\d+)%", msg)
        if m:
            out.append("dead-share-" + ("low" if int(m.group(1)) < 50 else "high"))
        m = re.search(r"(\d+) possessions outside", msg)
        if m:
            out.append("possession-count-" + ("low" if int(m.group(1)) < 80 else "high"))
        m = re.search(r"mean spell ([\d.]+)s", msg)
        if m:
            out.append("mean-spell-" + ("short" if float(m.group(1)) < 5 else "long"))
        if "one team owns" in msg:
            out.append("one-team-dominates")
        if "shorter than 1 s" in msg:
            out.append("flicker")
        if " X on graded" in msg:
            out.append("X-on-graded")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=os.environ.get("MTAH_DATA_DIR") or "data")
    args = ap.parse_args()
    rep = os.path.join(args.data_root, "reports")
    full, vid, hold = read(os.path.join(rep, "full_report.md")), read(os.path.join(rep, "video_pass.md")), read(os.path.join(rep, "holdout_ball.md"))
    assets = []
    if os.path.isfile(os.path.join(rep, "asset_report.csv")):
        with open(os.path.join(rep, "asset_report.csv"), newline="", encoding="utf-8") as fh:
            assets = list(csv.DictReader(fh))
    run_commit = read(os.path.join(rep, "run2_commit.txt")).strip()
    have_reports = bool(full and hold and assets)

    variants = {}
    for line in full.splitlines():
        m = re.match(r"\| (v\d\w*) \| (\w+) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| ([\d.]+) \| ([0-9a-f]{64}) \|", line)
        if m:
            variants[m.group(1)] = {"format": m.group(4).strip(), "gates": m.group(5).strip(), "sha": m.group(7)}
    ok = {k: v["format"] == "OK" and v["gates"] == "PASS" for k, v in variants.items()}

    def asset(name):
        return next((a for a in assets if a["asset"] == name), None)

    bc = asset("broadcast/atd_raw")
    bc_applied = bool(bc and "rules PASS" in bc["detail"] and "ACCEPT" in bc["detail"])
    vcue = asset("tactical/video")
    acue = asset("tactical/audio")
    hom = asset("tactical/homography")
    video_cue = bool(vcue and "-> VERIFIED" in vcue["detail"])
    audio_cue = bool(acue and "-> VERIFIED" in acue["detail"])
    hom_ok = bool(hom and hom["processed_ok"] == "True")

    m = re.search(r"detector on held-out frames: (\d+) recovered, median ([\d.]+) m, p90 ([\d.]+) m, gross errors \(> \d+ m\) (\d+)%", vid)
    held_n = int(m.group(1)) if m else 0
    gross = int(m.group(4)) / 100.0 if m else 1.0
    accepted = "detector accepted: True" in vid
    promoted = "video promoted: True" in vid
    m2 = re.search(r"decoded (\d+) windows \((\d+) from cache, (\d+) left for the next run", vid)
    skipped = int(m2.group(3)) if m2 else None
    m3 = re.search(r"tactical ball frames filled by video (\d+)", vid)
    filled = int(m3.group(1)) if m3 else 0
    complete = bool(m2) and skipped == 0

    cm = code_matches(run_commit)
    notes = []
    v_ok = ok.get("v3_vidpass", False)
    vidpass_pick = bool(v_ok and promoted and accepted and gross <= MAX_GROSS and held_n >= MIN_HELD and filled > 0 and complete)
    if "v3_vidpass" in variants and not vidpass_pick:
        why = [n for n, c in (("candidate invalid or gates fail", not v_ok), ("held-out gain/limits not met", not promoted),
                              ("detector not accepted", not accepted), ("gross-error rate above 10%", gross > MAX_GROSS),
                              ("too few held-out frames", held_n < MIN_HELD), ("no frames filled in real windows", filled == 0),
                              ("video pass incomplete (time budget)", not complete)) if c]
        notes.append("v3_vidpass not recommended: " + "; ".join(why))
    any_v3_signal = bc_applied or video_cue or audio_cue or hom_ok
    if vidpass_pick:
        rec, flags = "v3_vidpass", "--profile v3 --video-pass"
    elif ok.get("v3_final") and any_v3_signal:
        rec, flags = "v3_final", "--profile v3"
    elif ok.get("v2"):
        rec, flags = "v2", "--profile v2"
    else:
        rec, flags = "v1 (frozen)", "--profile v1"
    if not have_reports:
        rec, flags = "v1 (frozen)", "--profile v1"
        notes.append("reports missing: run not finished or not from this folder")
    if cm is False:
        notes.append("code changed since the run: re-run before relying on this decision")

    with open(os.path.join(rep, "decision_status.txt"), "w", encoding="utf-8") as fh:
        yn = lambda b: "unknown" if b is None else ("yes" if b else "no")  # noqa: E731
        fh.write("\n".join([
            "reports_present: " + yn(have_reports),
            "code_unchanged_since_run: " + yn(cm),
            "candidates_valid_and_gates_pass: " + (",".join(sorted(k for k, v in ok.items() if v)) or "none"),
            "candidates_failing: " + (",".join(sorted(k for k, v in ok.items() if not v)) or "none"),
            *["gate_or_format_failures[{}]: {}".format(k, ",".join(gate_categories(variants[k]["gates"])) or ("format" if variants[k]["format"] != "OK" else "?"))
              for k in sorted(variants) if not ok[k]],
            "broadcast_tracking_verified_on_heldout: " + yn(bc_applied),
            "tactical_video_cue_verified: " + yn(video_cue),
            "whistle_cue_verified: " + yn(audio_cue),
            "homography_used: " + yn(hom_ok),
            "video_detector_verified_on_heldout: " + yn(accepted),
            "video_promoted_by_heldout_gain: " + yn(promoted),
            "video_gross_error_rate_ok: " + yn(gross <= MAX_GROSS if m else None),
            "video_pass_complete: " + yn(complete if m2 else None),
            "video_frames_filled_in_real_windows: " + yn(filled > 0),
            "selection_windows: detector constants predeclared (not tuned on any real window); broadcast radius/min-gap chosen on the same held-out stretches that are reported (3 radii, 2 gap options)",
            "RECOMMENDED: " + rec,
            "FLAGS: " + flags,
            "NOTES: " + (" | ".join(notes) or "none"),
        ]) + "\n")
    with open(os.path.join(rep, "decision.md"), "w", encoding="utf-8") as fh:
        fh.write("# Decision detail (LOCAL ONLY)\n\n")
        for k, v in sorted(variants.items()):
            fh.write("- {}: format {}, gates {}, sha256 {}\n".format(k, v["format"], v["gates"], v["sha"]))
        fh.write("\nheld-out detector frames {}, gross-error rate {:.0%}, real frames filled {}\n".format(held_n, gross, filled))
        fh.write("recommended {} ({}), file outputs\\full\\{}\\<team>.csv\n".format(rec, flags, rec.split()[0]))
        fh.write("run commit {}\n".format(run_commit))
    print(read(os.path.join(rep, "decision_status.txt")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
