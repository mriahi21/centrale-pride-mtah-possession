import numpy as np

import poss  # noqa: F401
from poss import tags


def _halves(n=4000):
    return [{"half": 1, "a": 0, "b": n - 1, "ft": {"n": n}}, {"half": 2, "a": n, "b": 2 * n - 1, "ft": {"n": n}}]


def _tag(key, start, end, team, sp=False):
    return {"code": key, "start": start, "end": end, "team": team, "type": "", "set_piece": sp, "key": key}


def test_sign_requires_agreement_in_both_halves():
    hs = _halves()
    pass1 = {1: np.zeros(4000, dtype=np.int8), 2: np.ones(4000, dtype=np.int8)}   # half 1 = H, half 2 = A
    tg = [_tag("BUILD UP", 100, 1100, 0), _tag("BUILD UP", 4100, 5100, 0)]        # both tagged home
    s = tags.learn_signs_v2(hs, pass1, tg, log=lambda *a: None)
    assert s["BUILD UP"] == 0                                                      # halves disagree -> ignored
    pass1b = {1: np.zeros(4000, dtype=np.int8), 2: np.zeros(4000, dtype=np.int8)}
    s2 = tags.learn_signs_v2(hs, pass1b, tg, log=lambda *a: None)
    assert s2["BUILD UP"] == 1


def test_negative_sign_learned_when_tag_names_the_other_side():
    hs = _halves()
    pass1 = {1: np.ones(4000, dtype=np.int8), 2: np.ones(4000, dtype=np.int8)}     # away in possession
    tg = [_tag("DEFENDING IN MIDDLE THIRD", 100, 1100, 0), _tag("DEFENDING IN MIDDLE THIRD", 4100, 5100, 0)]
    assert tags.learn_signs_v2(hs, pass1, tg, log=lambda *a: None)["DEFENDING IN MIDDLE THIRD"] == -1


def test_strong_ball_evidence_overrides_tags_and_unknown_team_is_neutral():
    h = _halves()[0]
    tg = [_tag("BUILD UP", 0, 1000, 0), _tag("PROGRESSION", 0, 1000, None)]
    signs = {"BUILD UP": 1, "PROGRESSION": 1}
    weak = tags.phase_evidence_v2(h, tg, signs, np.zeros(4000, dtype=np.float32))
    strong = tags.phase_evidence_v2(h, tg, signs, np.full(4000, -0.9, dtype=np.float32))
    assert 0 < weak[500] <= tags.CAP_PHASE2           # capped, home-leaning
    assert abs(strong[500]) < 1e-6                    # strong tracking wins
    tg2 = [_tag("PROGRESSION", 0, 1000, None)]
    assert abs(tags.phase_evidence_v2(h, tg2, signs, np.zeros(4000, dtype=np.float32))).max() == 0


def test_overlapping_tags_are_capped():
    h = _halves()[0]
    tg = [_tag("BUILD UP", 0, 1000, 0) for _ in range(10)]
    ev = tags.phase_evidence_v2(h, tg, {"BUILD UP": 1}, np.zeros(4000, dtype=np.float32))
    assert ev.max() <= tags.CAP_PHASE2 + 1e-6
