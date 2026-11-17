"""Build a full-size SYNTHETIC bundle and run the full-bundle workflow on it (wiring smoke test, no match data).

    python scripts/smoke_full_synthetic.py <empty_dir>
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from poss import fullrun, synth_full  # noqa: E402


def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    print("building synthetic bundle in", root, flush=True)
    sim = synth_full.build(root, seed=3)
    rc = fullrun.main(["--data-root", root, "--downloads", os.path.join(root, "_none"),
                       "--out", os.path.join(root, "out")])
    # SYNTHETIC-only scoring against the simulator's truth (idealised cues: gains are optimistic by construction)
    from evaluation import possession_metrics as pm
    gt = pm.label_codes(sim["truth"])
    print("\nsynthetic truth scores (simulator only):  variant  spellF1  macroF1  D-F1")
    for v in sorted(os.listdir(os.path.join(root, "out"))):
        table = pm.read_label_csv(os.path.join(root, "out", v, "Centrale Pride.csv"))
        r = pm.evaluate(gt, pm.label_codes(table))
        print("  {:<18} {:.3f}  {:.3f}  {:.2f}".format(v, r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
