"""Smart Tagging (automatic, inferred) as SOFT evidence only.

* Team tag -> H/A (Barcelona -> H, Athletic -> A); `N/A` (all throw-ins) -> no team evidence.
* We do not assume what a team tag *means* for each phase code (e.g. whether `DEFENDING ...` names the
  defending or the possessing side). The sign of each tag kind is learned without labels: agreement
  between the tag's team and the ball-evidence-only decode (a pseudo-label produced by an algorithm).
  Kinds with weak or ambiguous agreement are ignored.
* Set-piece intervals mark restarts: they give a small dead-ball bonus and a team prior for the
  possession that starts at the restart.
Intervals are phases, not possessions -- they are never copied across as labels.
"""
import os

import numpy as np

import poss  # noqa: F401
from games import GAME, asset_paths

K_PHASE = 0.12      # per-frame evidence added inside a phase interval (small vs. ball evidence ~0.5-0.9)
K_PRIOR = 12.0      # initial-state bias (Viterbi cost units) for the possession after a restart
MIN_FRAMES_PHASE, MIN_FRAMES_SP = 750, 500
AGREE_HI, AGREE_LO = 0.62, 0.38


def team_code(name):
    """0 = home (Barcelona), 1 = away (Athletic), None = N/A or unknown."""
    s = (name or "").strip().lower()
    if "barc" in s:
        return 0
    if s.startswith("at") and ("hl" in s or "ath" in s or "atl" in s):
        return 1
    return None


def load_tags(data_dir=None, game=GAME, log=print):
    """List of dicts {code, start, end, team, type, key, set_piece}; [] if the file is absent."""
    path = asset_paths(game, data_dir)["smart_tagging"]
    if not os.path.isfile(path):
        log("[tags] no Smart Tagging file found -> proceeding without")
        return []
    from starter import load_smart_tagging as lst
    out = []
    for e in lst.load(game, data_dir):
        code = (e["code"] or "").strip().upper()
        typ = (e["tags"].get("Type") or "").strip()
        sp = code == "SET PIECES"
        out.append({"code": code, "start": e["start_frame"], "end": e["end_frame"],
                    "team": team_code(e["tags"].get("Team")), "type": typ, "set_piece": sp,
                    "key": ("SP:" + typ) if sp else code})
    log("[tags] {} instances loaded".format(len(out)))
    return out


def tag_dead_bonus(h, tags):
    """0/1 array over the half: 1 inside any set-piece window."""
    n = h["ft"]["n"]
    out = np.zeros(n, dtype=np.float32)
    for t in tags:
        if t["set_piece"]:
            a, b = t["start"] - h["a"], t["end"] - h["a"]
            if b >= 0 and a < n:
                out[max(a, 0):min(b, n)] = 1.0
    return out


def _restart_window(h, t, alive, window=125):
    """(a, b) relative frames: the first alive frames after this set piece, or None."""
    n = h["ft"]["n"]
    lo, hi = max(t["start"] - h["a"] - 75, 0), min(t["end"] - h["a"] + 75, n - 1)
    if hi <= lo:
        return None
    idx = np.flatnonzero(alive[lo:hi + 1])
    if not len(idx):
        return None
    a = lo + int(idx[0])
    return a, min(a + window, n)


def learn_signs(halves, pass1, tags, log=print):
    """{key: -1/0/+1} from agreement with the ball-only decode (pass1: dict half -> codes)."""
    agg = {}
    for h in halves:
        codes = pass1[h["half"]]
        alive = codes < 2
        for t in tags:
            if t["team"] is None:
                continue
            if t["set_piece"]:
                w = _restart_window(h, t, alive)
                if w is None:
                    continue
                a, b = w
            else:
                a, b = max(t["start"] - h["a"], 0), min(t["end"] - h["a"], h["ft"]["n"])
                if b <= a:
                    continue
            seg = codes[a:b]
            tot = int((seg < 2).sum())
            if not tot:
                continue
            agree = int((seg == t["team"]).sum())
            for key in (t["key"], "SP:*" if t["set_piece"] else None):
                if key:
                    s = agg.setdefault(key, [0, 0])
                    s[0] += agree
                    s[1] += tot
    signs = {}
    for key, (agree, tot) in agg.items():
        need = MIN_FRAMES_SP if key.startswith("SP:") else MIN_FRAMES_PHASE
        rate = agree / float(tot) if tot else 0.5
        sign = 0
        if tot >= need:
            sign = 1 if rate >= AGREE_HI else (-1 if rate <= AGREE_LO else 0)
        signs[key] = sign
    log("[tags] learned signs (kind: agreement/frames/sign): " + "; ".join(
        "{}:{:.2f}/{}/{:+d}".format(k, agg[k][0] / max(agg[k][1], 1), agg[k][1], v) for k, v in sorted(signs.items())))
    return signs


def _sign_for(signs, t):
    if not t["set_piece"]:
        return signs.get(t["key"], 0)
    s = signs.get(t["key"], 0)
    return s if s != 0 else signs.get("SP:*", 0)


def phase_evidence(h, tags, signs):
    """Additive per-frame evidence from phase tags (positive = home), clipped."""
    n = h["ft"]["n"]
    out = np.zeros(n, dtype=np.float32)
    for t in tags:
        if t["set_piece"] or t["team"] is None:
            continue
        s = _sign_for(signs, t)
        if s == 0:
            continue
        a, b = max(t["start"] - h["a"], 0), min(t["end"] - h["a"], n)
        if b > a:
            out[a:b] += s * K_PHASE * (1.0 if t["team"] == 0 else -1.0)
    return np.clip(out, -2 * K_PHASE, 2 * K_PHASE)


def segment_priors(h, segs, tags, signs):
    """One initial-state prior per alive segment (positive favours H) from set-piece team tags."""
    pri = []
    for (a, b) in segs:
        best, best_d = 0.0, 1e9
        for t in tags:
            if not t["set_piece"] or t["team"] is None:
                continue
            s = _sign_for(signs, t)
            if s == 0:
                continue
            lo, hi = t["start"] - h["a"] - 75, t["end"] - h["a"] + 75
            if lo <= a <= hi:
                d = abs(a - (t["start"] - h["a"]))
                if d < best_d:
                    best_d = d
                    best = s * K_PRIOR * (1.0 if t["team"] == 0 else -1.0)
        pri.append(best)
    return pri


# ---------------------------------------------------------------------------------------------------------
# v2 calibration (profile "v2"): more conservative sign learning and strong tracking always wins.
# ---------------------------------------------------------------------------------------------------------
AGREE_HI2, AGREE_LO2 = 0.68, 0.32
MIN_FRAMES_PHASE_HALF, MIN_FRAMES_SP_HALF = 400, 150
K_PHASE2, CAP_PHASE2 = 0.10, 0.15
K_PRIOR2 = 8.0
STRONG_BALL = 0.5          # |ball evidence| at/above which tags contribute nothing


def learn_signs_v2(halves, pass1, tags, log=print):
    """Sign per tag kind, only when BOTH halves independently agree (same direction, enough frames).

    The ball-only decode and the tags are related signals, so a kind needs consistent agreement in each
    half separately; conflicting or thin kinds stay at 0 (ignored)."""
    agg = {}
    for h in halves:
        codes = pass1[h["half"]]
        alive = codes < 2
        for t in tags:
            if t["team"] is None:
                continue
            if t["set_piece"]:
                w = _restart_window(h, t, alive)
                if w is None:
                    continue
                a, b = w
            else:
                a, b = max(t["start"] - h["a"], 0), min(t["end"] - h["a"], h["ft"]["n"])
                if b <= a:
                    continue
            seg = codes[a:b]
            tot = int((seg < 2).sum())
            if not tot:
                continue
            agree = int((seg == t["team"]).sum())
            for key in (t["key"], "SP:*" if t["set_piece"] else None):
                if key:
                    s = agg.setdefault((key, h["half"]), [0, 0])
                    s[0] += agree
                    s[1] += tot
    signs = {}
    keys = {k for k, _ in agg}
    for key in sorted(keys):
        need = MIN_FRAMES_SP_HALF if key.startswith("SP:") else MIN_FRAMES_PHASE_HALF
        votes = []
        for half in sorted({hh["half"] for hh in halves}):
            ag, tot = agg.get((key, half), [0, 0])
            if tot < need:
                votes.append(0)
                continue
            rate = ag / float(tot)
            votes.append(1 if rate >= AGREE_HI2 else (-1 if rate <= AGREE_LO2 else 0))
        signs[key] = votes[0] if votes and all(v == votes[0] for v in votes) else 0
    log("[tags-v2] kinds with a consistent sign: {} of {}".format(sum(1 for v in signs.values() if v), len(signs)))
    return signs


def phase_evidence_v2(h, tags, signs, ball_u):
    """Capped additive team evidence from phase tags, attenuated where the ball evidence is strong."""
    n = h["ft"]["n"]
    out = np.zeros(n, dtype=np.float32)
    for t in tags:
        if t["set_piece"] or t["team"] is None:
            continue
        s = _sign_for(signs, t)
        if s == 0:
            continue
        a, b = max(t["start"] - h["a"], 0), min(t["end"] - h["a"], n)
        if b > a:
            out[a:b] += s * K_PHASE2 * (1.0 if t["team"] == 0 else -1.0)
    out = np.clip(out, -CAP_PHASE2, CAP_PHASE2)
    return (out * (1.0 - np.clip(np.abs(ball_u) / STRONG_BALL, 0.0, 1.0))).astype(np.float32)


def segment_priors_v2(h, segs, tags, signs):
    pri = segment_priors(h, segs, tags, signs)
    return [p * (K_PRIOR2 / K_PRIOR) for p in pri]
