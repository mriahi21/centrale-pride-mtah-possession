import numpy as np

import poss  # noqa: F401
from poss import broadcast, holdout, synth

BLOCKS = ((1, 1000, 14999), (2, 16000, 29499))


def _make(seed=3, n=30000):
    sim = synth.simulate(n_frames=n, blocks=BLOCKS, seed=seed, second_half_pre_dead=0, ball_missing=0.0)
    truth = sim["ball"] / np.array([synth.L, synth.W])
    rng = np.random.default_rng(seed)
    tact = truth.copy()
    miss = np.zeros(n, dtype=bool)
    while miss.mean() < 0.06:                              # a mostly observed tactical ball with short dropouts
        s = int(rng.integers(0, n)); miss[s:s + int(rng.integers(5, 40))] = True
    tact[miss] = np.nan
    parts = []
    for a, b in ((1500, 7500), (9000, 14000), (17000, 24000)):
        seg = truth[a:b].copy()
        m = np.zeros(len(seg), dtype=bool)
        while m.mean() < 0.3:
            s = int(rng.integers(0, len(seg))); m[s:s + int(rng.integers(10, 120))] = True
        seg[m] = np.nan
        seg += rng.normal(0, 0.3 / 105.3, seg.shape)
        parts.append(seg)
    return tact, np.concatenate(parts), truth


def test_masks_are_reliable_nonoverlapping_and_inside_graded_blocks():
    tact, bc, truth = _make()
    st = holdout.select_masks(tact, BLOCKS, per_bin=4)
    assert len(st) >= 12
    prev_end = -10 ** 9
    for s, L in st:
        assert any(a <= s and s + L <= b for _, a, b in BLOCKS)
        assert s >= prev_end + 300
        prev_end = s + L
        assert np.isfinite(tact[s:s + L, 0]).mean() >= 0.95


def test_holdout_excludes_masked_observations_and_recovers_them():
    tact, bc, truth = _make()
    res = holdout.evaluate(tact, bc, BLOCKS, per_bin=4, log=lambda *a: None)
    Tm = res["_arrays"]["Tm"]
    ev = res["_arrays"]["ev"]
    assert not np.isfinite(Tm[ev, 0]).any()                          # masked observations are hidden from alignment/filling
    dec = holdout.decide(res)
    assert dec["accept"]
    o = res["radii"][dec["radius"]]["overall"]
    assert o["n"] > 300 and o["median"] < 1.0                        # broadcast recovers hidden ball positions
    # long gaps: interpolation is not usable by the model (>2 s), broadcast still recovers
    long_bc = [x for x in res["radii"][dec["radius"]]["broadcast"] if x["bin"] == 250]
    long_ip = [x for x in res["interp"] if x["bin"] == 250]
    assert long_ip[0]["coverage"] == 0.0
    assert long_bc and long_bc[0]["coverage"] > 0.0


def test_unrelated_broadcast_is_rejected_by_the_holdout():
    tact, _, _ = _make(seed=4)
    rng = np.random.default_rng(1)
    other = np.stack([rng.uniform(0.1, 0.9, 8000), rng.uniform(0.1, 0.9, 8000)], axis=1)
    res = holdout.evaluate(tact, other, BLOCKS, per_bin=4, log=lambda *a: None)
    assert not holdout.decide(res)["accept"]


def test_fuse_min_gap_leaves_short_gaps_alone():
    tact, bc, truth = _make(seed=5)
    al, flip = broadcast.best_orientation(tact, bc, log=lambda *a: None)
    t_of_b = broadcast.verify_map(broadcast.frame_map(al, broadcast.accept(al), len(bc)), tact, bc, flip)
    f0, fill0, _ = broadcast.fuse_ball(tact, bc, t_of_b, flip, len(tact))
    f1, fill1, _ = broadcast.fuse_ball(tact, bc, t_of_b, flip, len(tact), min_gap=51)
    assert fill1.sum() < fill0.sum()
    assert (broadcast.gap_lengths(tact)[fill1] >= 51).all()
    obs = np.isfinite(tact[:, 0])
    assert np.allclose(f1[obs], tact[obs], equal_nan=True)
