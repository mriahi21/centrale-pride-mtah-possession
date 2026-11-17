"""LOCAL aggregate diagnostic: why do the v1 / v2 stoppage detectors differ on the real data?

    python scripts/diag_dead.py --data-root <DATA>

Prints only per-half aggregate shares (no per-frame data, no labels). Intended to explain a plausibility-gate failure,
not to tune anything on labels.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths  # noqa: E402
from poss import atd, control, deadball, deadball2, pipeline, tags as tagmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("MTAH_DATA_DIR") or "data")
    a = ap.parse_args()
    n = GAMES[GAME]["n_video_frames"]
    paths = asset_paths(GAME, a.data_root)
    data = atd.read_atd(paths["atd_raw"], n)
    team = atd.team_of_slots(paths["atd_meta"])
    tg = tagmod.load_tags(a.data_root, log=lambda *x: None)
    halves = pipeline.prepare_halves(data, team, pipeline.default_blocks(), pipeline.KICKOFFS)
    for h in halves:
        ft, k = h["ft"], h["kick_rel"]
        post = slice(k, ft["n"])
        bonus = tagmod.tag_dead_bonus(h, tg) if tg else None
        m1, _ = deadball.dead_mask(ft, k, bonus, log=lambda *x: None)
        m2, f2 = deadball2.dead_mask(ft, k, bonus, log=lambda *x: None)
        c = deadball2.cues(ft, bonus)
        z = deadball2.score(c)
        share = lambda m: float(np.mean(m[post]))  # noqa: E731
        print("half {}: ball observed {:.0%} | v1 dead {:.1%} | v2 dead {:.1%} ({} runs)".format(
            h["half"], float(np.mean(ft["ball_obs"][post])), share(m1), share(m2), f2["n_dead_runs"]))
        print("   v2 cue shares (value>0.6): oob {:.1%}  still {:.1%}  quiet {:.1%}  tag {:.1%}  live-veto {:.1%} | activity separation {:.2f} (quiet weight {:.2f})".format(
            *[float(np.mean(c[x][post] > 0.6)) for x in ("oob", "still", "quiet", "tag", "live")], c["_sep"], c["_w_quiet"]))
        print("   v2 score>enter {:.1%} | score>exit {:.1%} | v1/v2 dead agreement {:.1%}".format(
            float(np.mean(z[post] > deadball2.ENTER)), float(np.mean(z[post] > deadball2.EXIT)),
            float(np.mean(m1[post] == m2[post]))))
        print("   ball speed<0.6 m/s on observed frames {:.1%} | ball out of bounds {:.1%} | players quiet(<q35) {:.1%}".format(
            float(np.nanmean(ft["ball_speed"][post][ft["ball_obs"][post]] < 0.6)), float(np.mean(ft["oob"][post])),
            float(np.mean(c["quiet"][post] > 0))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
