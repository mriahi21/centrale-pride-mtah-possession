import numpy as np

import poss  # noqa: F401
from evaluation import possession_metrics as pm
from poss import deadball, pipeline, synth, tags


def test_team_code_mapping():
    assert tags.team_code("Barcelona") == 0
    assert tags.team_code("Atletic Club") == 1 and tags.team_code("Athletic Club") == 1
    assert tags.team_code("N/A") is None and tags.team_code(None) is None


def test_gmm_bimodal_and_unimodal():
    rng = np.random.default_rng(0)
    bi = np.concatenate([rng.normal(0.0, 0.2, 20000), rng.normal(1.2, 0.2, 20000)])
    p, ok = deadball.gmm_low_posterior(bi)
    assert ok and p[:20000].mean() > 0.9 and p[20000:].mean() < 0.1
    _, ok2 = deadball.gmm_low_posterior(rng.normal(0, 1, 40000))
    assert not ok2


def test_clean_runs_min_duration_and_gap_fill():
    d = np.zeros(1000, dtype=bool)
    d[100:110] = True            # too short -> dropped
    d[300:400] = True
    d[410:500] = True            # 10-frame alive gap -> filled
    out = deadball.clean_runs(d)
    assert not out[100:110].any() and out[300:500].all()


def test_missing_ball_alone_is_not_dead():
    n = 3000
    ft = {"n": n, "ball_conf": np.zeros(n, dtype=np.float32), "oob": np.zeros(n, dtype=bool),
          "ball_speed": np.full(n, np.nan, dtype=np.float32), "activity": np.full(n, 2.5, dtype=np.float32)}
    mask, _ = deadball.dead_mask(ft, 0, log=lambda *a: None)
    assert not mask.any()


def test_full_pipeline_with_tags_on_synthetic(tmp_path):
    n, blocks = 12000, ((1, 500, 5999), (2, 6500, 11499))
    sim = synth.simulate(n_frames=n, blocks=blocks, seed=5, second_half_pre_dead=0)
    paths = synth.write_bundle(sim, str(tmp_path))
    tg = tags.load_tags(str(tmp_path), log=lambda *a: None)
    assert tg and any(t["set_piece"] for t in tg)
    codes, info = pipeline.generate_labels(paths["atd_raw"], paths["atd_meta"], n_frames=n, blocks=list(blocks),
                                           kickoffs={1: 500, 2: 6500}, tags=tg, log=lambda *a: None)
    gt = pm.label_codes(sim["truth"])
    r = pm.evaluate(gt, codes)
    assert r["per_class"]["D"]["f1"] > 0.6          # synthetic wiring check only
    assert info["tag_signs"].get("SP:*") == 1        # synthetic tags carry the restart team -> sign learned
