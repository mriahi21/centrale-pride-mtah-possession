"""Build a clean, code-only snapshot of this repo for the separate PUBLIC submission repository.

    python scripts/make_public_snapshot.py --out public_snapshot --team-name "Centrale Pride" --profile v3 --flags=--video-pass

Allow-list only: code, tests (synthetic), vendored upstream code, pinned requirements, the technical note.
Excluded on purpose: CLAUDE.md, PUBLIC_REPO_CONFLICT.md, docs/FROZEN_CANDIDATE.md, git history, anything not on the list.
A safety scan fails the build if it finds data-like files, e-mail addresses, tokens, or local user paths.
This script does not publish anything; publishing is a separate, explicit step (see docs/PUBLISH_STEPS.md).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW_PREFIXES = ("poss/", "tests/", "evaluation/", "starter/", "scripts/")
ALLOW_FILES = {"games.py", "LICENSE", "requirements.txt", "requirements-pinned.txt", "requirements-full.txt", "requirements-dev.txt",
               ".gitignore", "docs/TECHNICAL_NOTE.md"}
BAD_EXT = (".csv", ".xml", ".mp4", ".npz", ".npy", ".pkl", ".parquet", ".gz", ".zip", ".pdf", ".json")
PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "github token": re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "local user path": re.compile(r"[A-Za-z]:\\Users\\|/Users/[a-z]+/|/home/[a-z]+/", re.I),
    "private data root": re.compile(r"C:[\\/]+mtah", re.I),
}
README = """# {team} — MTAH 2027 Phase 1 (ball possession)

Per-frame labels `H/A/D/X` for the BAR–ATH tactical video. Training-free temporal state model: soft ball-control
evidence + Viterbi hysteresis, corroborated stoppage detection, Smart Tagging as capped soft evidence; profile `v3`
adds optional signals (broadcast tracking, tactical video/audio, broadcast-video ball detections) that are each applied
only if they pass predeclared verification checks. No model is trained on any labels.
Technical note: `docs/TECHNICAL_NOTE.md`. Challenge: https://github.com/metrica-sports/more-than-a-hack-2027

The match data is **not** included and must not be committed (see `.gitignore`).

## Reproduce (needs the gated data bundle)
Layout: `<DATA>/tactical/BAR-ATH_tactical{{.mp4,_atd.txt,_atd.xml,_homography.csv,_smart_tagging.xml}}` and
`<DATA>/broadcast/BAR-ATH_broadcast{{.mp4,_atd.txt,_atd.xml,_homography.csv}}` (profiles v1/v2 need only the tactical ATD files).
```bash
pip install -r requirements-full.txt      # pinned; Python 3.14 tested
python -m poss.make_submission --data-dir <DATA> --team-name "{team}" --profile {profile} {flags} --out outputs
python evaluation/validate_submission.py "outputs/{team}.csv"
python -m pytest -q tests                 # synthetic wiring tests, no match data
```
Deterministic: the same bundle and commit give the same file (compare SHA-256). Intermediate arrays are cached under
`<DATA>/cache` (never commit it); deleting the cache forces recomputation from the raw inputs.
`games.py`, `evaluation/`, `starter/` are vendored unmodified from the official repository (MIT).
"""


def tracked_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [p.strip() for p in out.splitlines() if p.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="public_snapshot")
    ap.add_argument("--team-name", default="Centrale Pride")
    ap.add_argument("--profile", default="v1")
    ap.add_argument("--flags", default="", help="extra make_submission flags to print in the README, e.g. --video-pass")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    copied, problems = [], []
    for rel in tracked_files():
        if not (rel in ALLOW_FILES or rel.startswith(ALLOW_PREFIXES)):
            continue
        if rel.lower().endswith(BAD_EXT):
            problems.append("data-like file on allow-list: " + rel)
            continue
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
        copied.append(rel)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(team=args.team_name, profile=args.profile, flags=args.flags))

    for rel in copied + ["README.md"]:
        path = os.path.join(out, rel)
        if os.path.basename(rel) == "make_public_snapshot.py":
            continue                                   # contains the patterns themselves
        try:
            text = open(path, encoding="utf-8", errors="strict").read()
        except UnicodeDecodeError:
            problems.append("binary file: " + rel)
            continue
        for name, pat in PATTERNS.items():
            m = pat.search(text)
            if m:
                problems.append("{} in {}: {}".format(name, rel, m.group(0)[:40]))

    print("snapshot: {} files -> {}".format(len(copied) + 1, out))
    if problems:
        print("SAFETY SCAN FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("safety scan: clean (no data-like files, e-mails, tokens, or local paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
