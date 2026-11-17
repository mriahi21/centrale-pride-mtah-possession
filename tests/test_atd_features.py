import os

import numpy as np
import pytest

import poss  # noqa: F401
from games import asset_paths
from poss import atd, features, synth


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("bundle"))
    sim = synth.simulate(n_frames=6000, blocks=((1, 500, 2999), (2, 3500, 5999)), seed=1)
    paths = synth.write_bundle(sim, d)
    return sim, paths


def test_read_atd_alignment_and_nan(bundle):
    sim, paths = bundle
    data = atd.read_atd(paths["atd_raw"], sim["n_frames"])
    assert data["n_read"] == sim["n_frames"]
    # video_frame = frame_id - 1: spot-check a frame where the simulated ball is present
    f = int(np.flatnonzero(np.isfinite(sim["ball"][:, 0]))[100])
    assert np.allclose(data["ball"][f] * [synth.L, synth.W], sim["ball"][f], atol=0.01)
    # NaN frames stay NaN (halftime: players absent)
    assert np.isnan(data["players"][3200]).all()


def test_slow_path_single_nan_groups(tmp_path):
    raw = tmp_path / "a.txt"
    raw.write_text("1:0.5,0.5,1.0;NaN;0.2,0.2,2.0:NaN\n2:0.5,0.5,1.0;NaN,NaN,NaN;0.2,0.2,2.0:0.1,0.1,3.0\n")
    d = atd.read_atd(str(raw), 3)
    assert d["players"].shape == (3, 3, 2)
    assert np.isnan(d["players"][0, 1]).all() and np.isfinite(d["players"][0, 2]).all()
    assert np.isnan(d["ball"][0]).all() and np.isfinite(d["ball"][1]).all()
    assert np.isnan(d["players"][2]).all()  # frame 3 absent


def test_team_of_slots(bundle):
    _, paths = bundle
    t = atd.team_of_slots(paths["atd_meta"])
    assert len(t) == 46 and t[:23].sum() == 0 and t[23:].sum() == 23


def test_ball_outlier_and_interp():
    b = np.full((200, 2), np.nan)
    b[:100, 0] = np.linspace(0, 10, 100)
    b[:100, 1] = 5
    b[50] = [90, 60]  # glitch
    b[120:130, 0] = np.linspace(11, 12, 10)
    b[120:130, 1] = 5
    c = features.remove_ball_outliers(b)
    assert np.isnan(c[50, 0])
    ib, conf, obs = features.interpolate_ball(c, max_gap=50)
    assert np.isfinite(ib[50, 0]) and conf[50] < 1  # bridged short gap, lower confidence
    assert np.isnan(ib[150, 0]) and conf[150] == 0   # after last observation: not bridged


def test_half_features_and_attack_direction(bundle):
    sim, paths = bundle
    data = atd.read_atd(paths["atd_raw"], sim["n_frames"])
    team = atd.team_of_slots(paths["atd_meta"])
    for half, a, b in sim["blocks"]:
        sl = slice(a, b + 1)
        ft = features.compute_half_features(data["players"][sl], data["pspeed"][sl], data["ball"][sl], team,
                                            kickoff_rel=300 if half == 2 else 0)
        # simulated: home attacks +x in half 1, -x in half 2
        assert ft["attack_dir"][0] == (1 if half == 1 else -1)
        # truth H -> nearest home player is closer than nearest away most of the time (ball observed, alive)
        lab = sim["truth"][sl]
        m = (lab == "H") & ft["ball_obs"] & (ft["ball_speed"] < 3)
        assert (ft["dH"][m] < ft["dA"][m]).mean() > 0.8
        assert ft["n"] == b - a + 1
