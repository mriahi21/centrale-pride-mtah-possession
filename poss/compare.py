"""LOCAL comparison of candidate profiles on the restricted bundle. Console output only (aggregates); run it in
your own terminal. It writes each candidate to its own folder so the frozen v1 file is never overwritten:

    python -m poss.compare --data-dir <DATA> --team-name "Centrale Pride" --out outputs

  outputs/cmp_v1/<team>.csv         frozen v1 reproduced
  outputs/cmp_v2/<team>.csv         v2
  outputs/cmp_v2_notags/<team>.csv  v2 without Smart Tagging

For each: official-format validation, SHA-256, predeclared plausibility gates (poss/gates.py), and how much the
candidates differ from each other (frame agreement; possession-spell F1 of one against the other). There is no ground
truth here: agreement says how different candidates are, never which is right.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

import poss  # noqa: F401
from evaluation import possession_metrics as pm
from games import DATA_DIR, GAME, GAMES, require
from poss import atd, gates, make_submission as ms, pipeline, tags as tagmod

CANDIDATES = (("v1", "v1", True), ("v2", "v2", True), ("v2_notags", "v2", False))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--team-name", default="Centrale Pride")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    data_dir = args.data_dir or DATA_DIR
    n_rows = GAMES[GAME]["n_video_frames"]
    blocks = pipeline.default_blocks()

    meta, raw = require(["atd_meta", "atd_raw"], GAME, data_dir)
    data = atd.read_atd(raw, n_rows)
    team = atd.team_of_slots(meta)
    halves = pipeline.prepare_halves(data, team, blocks, pipeline.KICKOFFS)
    tg = tagmod.load_tags(data_dir, log=lambda *a: None)

    quiet = lambda *a: None  # noqa: E731
    results = {}
    for name, profile, use_tags in CANDIDATES:
        codes, info = pipeline.label_halves(halves, n_rows, tags=tg if use_tags else [], profile=profile, log=quiet)
        path = ms.write_csv(os.path.join(args.out, "cmp_" + name, args.team_name + ".csv"), codes)
        reason = ms.validate(path, n_rows)
        ok, msgs, stats = gates.check(codes, blocks, pipeline.KICKOFFS)
        results[name] = codes
        print("\n== {}  lambda={:.1f}".format(name, info["lambda"]))
        print("   file: {}".format(path))
        print("   sha256: {}".format(sha256(path)))
        print("   format: {}".format("OK" if not reason else "FAIL: " + reason))
        for st in stats:
            print("   half {half}: dead {dead_share:.1%}, {possessions} possessions, mean {mean_spell_s}s, "
                  "top-team share {team_share}".format(**st))
        print("   gates: {}".format("PASS" if ok else "FAIL"))
        for m in msgs:
            print("     - " + m)

    print("\n== how different are the candidates (no ground truth; says nothing about which is right)")
    graded = np.zeros(n_rows, dtype=bool)
    runs = []
    for _, a, b in blocks:
        graded[a:b + 1] = True
        runs.append((a, b + 1))
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            agree = float(np.mean(a[graded] == b[graded]))
            sp = pm.spell_scores(a, b, runs)["mean_f1"]
            print("   {} vs {}: frame agreement {:.1%}, possession-spell F1 {:.3f}".format(names[i], names[j], agree, sp))
    print("\nRe-validate the exact upload file with: python evaluation/validate_submission.py \"<path to csv>\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
