"""End-to-end labelling: ATD -> per-half features -> evidence -> segmentation -> frame codes.

Codes: 0 H, 1 A, 2 D, 3 X (same as evaluation/possession_metrics.py). Halves are processed
separately; frames outside the graded halves are X (never scored).
"""
import numpy as np

import poss  # noqa: F401
from games import GAME, half_video_blocks, source_frames
from poss import atd, features, model

X = 3
PROFILES = ("v1", "v2", "v3")   # v1 = frozen validated candidate; v2 = stoppage/restart rework; v3 = v2 + optional video/audio/broadcast signals
# Kick-off video frames. Both are published (DATA.md / FAQ.md / starter/load_smart_tagging.py).
KICKOFFS = {1: 11185, 2: 113742}


def default_blocks():
    return half_video_blocks(GAME)


def prepare_halves(data, team, blocks, kickoffs, extras=None):
    halves = []
    for half, a, b in blocks:
        sl = slice(a, b + 1)
        kick_rel = max(0, kickoffs.get(half, a) - a)
        ft = features.compute_half_features(data["players"][sl], data["pspeed"][sl], data["ball"][sl],
                                            team, kickoff_rel=kick_rel)
        for k, v in (extras or {}).items():       # optional full-length signals (whistle, vquiet, hom_ok)
            if v is not None:
                ft[k] = np.asarray(v)[sl]
        halves.append({"half": half, "a": a, "b": b, "ft": ft, "kick_rel": kick_rel})
    return halves


def prekickoff_dead(n, kick_rel):
    dead = np.zeros(n, dtype=bool)
    dead[:kick_rel] = True
    return dead


def _choose_lambda(evid, alive, seg_priors=None):
    tot_alive = sum(int(a.sum()) for a in alive)
    best, best_gap, table = None, np.inf, []
    for cand in model.LAMBDA_GRID:
        k = sum(model.count_spells(model.decode_alive(u, a, cand, None if seg_priors is None else seg_priors[i]))
                for i, (u, a) in enumerate(zip(evid, alive)))
        mean_s = tot_alive / 25.0 / max(k, 1)
        gap = abs(np.log(max(mean_s, 1e-3) / model.TARGET_MEAN_SPELL_S))
        table.append((cand, k, round(mean_s, 1)))
        if gap < best_gap:
            best, best_gap = cand, gap
    return best, table


def label_halves(halves, n_frames, tags=None, use_dead=True, lam=None, log=print, profile="v1"):
    """Return (codes[n_frames], info).

    Pass 1: ball evidence only (+ dead mask) -> pseudo-labels used solely to learn the sign of each
    Smart Tagging kind. Pass 2: ball evidence + tag evidence + restart priors -> final decode.
    """
    from poss import deadball, deadball2, control, tags as tagmod
    v2 = profile in ("v2", "v3")
    dmod = deadball2 if v2 else deadball
    tags = tags or []
    for h in halves:
        n = h["ft"]["n"]
        if v2:
            h["u0"], h["aux"] = control.control_evidence(h["ft"])
        else:
            h["u0"] = model.control_evidence(h["ft"])
        if use_dead:
            bonus = tagmod.tag_dead_bonus(h, tags) if tags else None
            h["dead"], h["dead_flags"] = (dmod.dead_mask(h["ft"], h["kick_rel"], bonus, log=log, extras=(profile == "v3")) if v2
                                           else dmod.dead_mask(h["ft"], h["kick_rel"], bonus, log=log))
        else:
            h["dead"] = prekickoff_dead(n, h["kick_rel"])
        h["alive"] = ~h["dead"]

    info = {}
    if v2:
        lam = lam or control.dwell_lambda([h["u0"] for h in halves])
        lam1 = lam
        for h in halves:
            h["rpri"] = control.restart_priors(h["aux"], model.alive_segments(h["alive"]))
    else:
        lam1 = lam or _choose_lambda([h["u0"] for h in halves], [h["alive"] for h in halves])[0]
    signs, evid, pri = {}, [h["u0"] for h in halves], None
    if tags:
        pass1 = {h["half"]: model.decode_alive(h["u0"], h["alive"], lam1, h.get("rpri")) for h in halves}
        if v2:
            signs = tagmod.learn_signs_v2(halves, pass1, tags, log=log)
            evid = [h["u0"] + tagmod.phase_evidence_v2(h, tags, signs, h["u0"]) for h in halves]
            pri = [tagmod.segment_priors_v2(h, model.alive_segments(h["alive"]), tags, signs) for h in halves]
        else:
            signs = tagmod.learn_signs(halves, pass1, tags, log=log)
            evid = [h["u0"] + tagmod.phase_evidence(h, tags, signs) for h in halves]
            pri = [tagmod.segment_priors(h, model.alive_segments(h["alive"]), tags, signs) for h in halves]
    if v2:   # restart-contact prior is used when no (reliable) tag prior exists for that restart
        pri = [[(tp if tp != 0.0 else rp) for tp, rp in zip(pri[i] if pri else [0.0] * len(h["rpri"]), h["rpri"])]
               for i, h in enumerate(halves)]
    alive = [h["alive"] for h in halves]
    if lam is None:
        lam, table = _choose_lambda(evid, alive, pri)
        info["lambda_table"] = table
    info["lambda"], info["tag_signs"] = lam, signs

    codes = np.full(n_frames, X, dtype=np.int8)
    for i, (h, u, a) in enumerate(zip(halves, evid, alive)):
        codes[h["a"]:h["b"] + 1] = model.decode_alive(u, a, lam, None if pri is None else pri[i])
    log("[model] lambda={} (spells/mean-s table: {})".format(lam, info.get("lambda_table", "fixed")))
    return codes, info


def generate_labels(raw_path, meta_path, n_frames=None, blocks=None, kickoffs=None, tags=None, log=print, **kw):
    n_frames = n_frames or source_frames("tactical", GAME)
    blocks = blocks or default_blocks()
    kickoffs = kickoffs or KICKOFFS
    log("[atd] reading {} ...".format(raw_path))
    data = atd.read_atd(raw_path, n_frames)
    log("[atd] {}".format(atd.check_coverage(data, n_frames)))
    team = atd.team_of_slots(meta_path)
    halves = prepare_halves(data, team, blocks, kickoffs)
    return label_halves(halves, n_frames, tags=tags, log=log, **kw)
