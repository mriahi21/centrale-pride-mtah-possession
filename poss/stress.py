"""Synthetic stress tests / candidate comparison (no match data).

    python -m poss.stress            # table: profile x ball-missing coverage x tag quality

Reports spellF1 first, macroF1 second, plus D-F1. Synthetic results never establish a full-match gain;
they check robustness and regressions only.
"""
import copy
import sys
import tempfile

import numpy as np

import poss  # noqa: F401
from evaluation import possession_metrics as pm
from poss import control, pipeline, synth, tags as tg

N, BLOCKS = 24000, ((1, 1000, 11999), (2, 13000, 23499))
KICK = {1: 1000, 2: 13000}


def corrupt_tags(tag_list, mode, seed=0):
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(tag_list)
    for t in out:
        if mode == "late":            # tags delayed by 4 s
            t["start"] += 100
            t["end"] += 100
        elif mode == "wrong":         # 35% of team tags flipped
            if t["team"] is not None and rng.random() < 0.35:
                t["team"] = 1 - t["team"]
        elif mode == "none":
            t["team"] = None
    return out


def run_case(profile, ball_missing, tag_mode, seed, lam=None, use_tags=True, bundle=None, log=None):
    sim = synth.simulate(n_frames=N, blocks=BLOCKS, seed=seed, second_half_pre_dead=0, ball_missing=ball_missing)
    d = tempfile.mkdtemp()
    p = synth.write_bundle(sim, d)
    tl = tg.load_tags(d, log=lambda *a: None)
    tl = corrupt_tags(tl, tag_mode, seed) if tag_mode != "good" else tl
    gt = pm.label_codes(sim["truth"])
    c, info = pipeline.generate_labels(p["atd_raw"], p["atd_meta"], n_frames=N, blocks=list(BLOCKS), kickoffs=KICK,
                                       tags=tl if use_tags else [], log=lambda *a: None, profile=profile, lam=lam)
    r = pm.evaluate(gt, c)
    return r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"], info["lambda"]


def main():
    print("{:<10} {:>7} {:<7} {:>8} {:>8} {:>6} {:>7}".format("profile", "missing", "tags", "spellF1", "macroF1", "D-F1", "lambda"))
    for ball_missing in (0.2, 0.41, 0.6):
        for tag_mode in ("good", "late", "wrong", "none"):
            for profile in ("v1", "v2"):
                rows = [run_case(profile, ball_missing, tag_mode, s) for s in (31, 32)]
                m = np.mean([r[:3] for r in rows], axis=0)
                print("{:<10} {:>7} {:<7} {:>8.3f} {:>8.3f} {:>6.2f} {:>7.1f}".format(
                    profile, ball_missing, tag_mode, m[0], m[1], m[2], np.mean([r[3] for r in rows])))


if __name__ == "__main__":
    sys.exit(main())
