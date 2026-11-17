import numpy as np
import pytest

import poss  # noqa: F401
from evaluation import possession_metrics as pm
from poss import atd, model, pipeline, synth
from starter import baseline


def test_viterbi_suppresses_flicker_and_switches_on_sustained_evidence():
    u = np.ones(300, dtype=np.float32) * 0.8
    u[100:108] = -0.9          # 8-frame blip of opposing 'control'
    u[200:] = -0.8             # sustained change at 200
    s = model.viterbi_two_state(u, lam=30)
    assert (s[:200] == 0).all() and (s[200:] == 1).all()
    s2 = model.viterbi_two_state(u, lam=3)
    assert (s2[100:108] == 1).all()   # with a tiny penalty the blip survives


def test_missing_ball_carries_state_not_dead():
    u = np.zeros(500, dtype=np.float32)
    u[:50] = -0.8
    codes = model.decode_alive(u, np.ones(500, dtype=bool), lam=30)
    assert (codes == 1).all() and (codes != model.D).all()


def test_switch_timing_is_at_start_of_opposing_run():
    u = np.ones(400, dtype=np.float32) * 0.5
    u[180:] = -0.5
    s = model.viterbi_two_state(u, lam=20)
    assert 175 <= int(np.flatnonzero(s == 1)[0]) <= 185


def _synth_run(tmp_path, seed):
    n, blocks = 12000, ((1, 500, 5999), (2, 6500, 11499))
    sim = synth.simulate(n_frames=n, blocks=blocks, seed=seed, second_half_pre_dead=0)
    paths = synth.write_bundle(sim, str(tmp_path / ("b%d" % seed)), with_tags=False)
    codes, info = pipeline.generate_labels(paths["atd_raw"], paths["atd_meta"], n_frames=n, blocks=list(blocks),
                                           kickoffs={1: 500, 2: 6500}, log=lambda *a: None)
    return sim, paths, codes


def test_pipeline_wiring_and_beats_nearest_player_baseline_on_synthetic(tmp_path):
    sim, paths, codes = _synth_run(tmp_path, 3)
    gt = pm.label_codes(sim["truth"])
    assert len(codes) == 12000 and set(np.unique(codes)) <= {0, 1, 2, 3}
    assert (codes[:500] == 3).all() and (codes[6000:6500] == 3).all()  # ungraded -> X
    ours = pm.evaluate(gt, codes)
    team = atd.team_of_slots(paths["atd_meta"])
    raw = baseline.nearest_player_labels(paths["atd_raw"], team, 12000, (105.3, 68.0))
    bl = pm.label_codes(baseline.smooth(raw, 50))
    bl[gt == 3] = 3
    base = pm.evaluate(gt, bl)
    # wiring check on SYNTHETIC data only -- says nothing about the real match
    assert ours["spell_f1_mean"] >= base["spell_f1_mean"]
