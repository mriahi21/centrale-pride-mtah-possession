import numpy as np
import pytest

import poss  # noqa: F401
from poss import deadball2, scenarios as sc

q = lambda *a: None  # noqa: E731
HARD_MISS = {"stoppage_ball_occluded"}   # real stoppage with an untracked ball: ambiguity is accepted


@pytest.mark.parametrize("name", sc.SCENARIOS)
@pytest.mark.parametrize("seed", [1, 2])
def test_v2_no_false_splits_no_missed(name, seed):
    ft, truth, tag = sc.scenario(name, seed)
    pred, _ = deadball2.dead_mask(ft, 0, tag, log=q)
    m = sc.stoppage_metrics(pred, truth)
    assert m["false_splits"] == 0
    if name not in HARD_MISS:
        assert m["missed"] == 0 and m["d_f1"] > 0.9


def test_quiet_players_alone_never_enter_dead():
    ft, truth, tag = sc.scenario("occluded_quiet_live", 3)
    tag[:] = 0.0                                   # no tags either
    pred, _ = deadball2.dead_mask(ft, 0, tag, log=q)
    live_only = ~truth
    assert pred[live_only].mean() < 0.05


def test_tag_alone_never_enters_dead():
    ft, truth, tag = sc.scenario("baseline", 1)
    tag[:] = 0.0
    tag[~truth] = 1.0                              # tag claims a stoppage everywhere in live play
    pred, _ = deadball2.dead_mask(ft, 0, tag, log=q)
    assert pred[~truth].mean() < 0.05


def test_genuine_short_live_stretch_is_preserved():
    ft, truth, tag = sc.scenario("short_live_between_stops", 2)
    pred, _ = deadball2.dead_mask(ft, 0, tag, log=q)
    a = int(np.flatnonzero(~truth[400:]).min()) + 400  # first live frame after the first stoppage
    assert (~pred[a + 10:a + 40]).all()


def test_restart_boundary_follows_ball_movement():
    ft, truth, tag = sc.scenario("baseline", 4)
    pred, _ = deadball2.dead_mask(ft, 0, tag, log=q)
    assert sc.stoppage_metrics(pred, truth)["restart_err"] < 10
