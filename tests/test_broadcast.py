import numpy as np

import poss  # noqa: F401
from poss import broadcast, synth


def _make(seed=3, flip=False):
    n = 30000
    sim = synth.simulate(n_frames=n, blocks=((1, 1000, 14999), (2, 16000, 29499)), seed=seed, second_half_pre_dead=0,
                         ball_missing=0.0)
    truth_ball = sim["ball"] / np.array([synth.L, synth.W])
    rng = np.random.default_rng(seed)
    tact = truth_ball.copy()
    miss = np.zeros(n, dtype=bool)
    while miss.mean() < 0.43:
        s = int(rng.integers(0, n)); miss[s:s + int(rng.integers(20, 200))] = True
    tact[miss] = np.nan
    # broadcast timeline: three live chunks of the match + a replay of an earlier chunk (own gaps, own noise)
    chunks = [(1500, 6500), (9000, 13000), (4000, 5500), (17000, 22000)]   # (4000,5500) is a REPLAY (earlier time)
    parts, t_of_b = [], []
    for a, b in chunks:
        seg = truth_ball[a:b].copy()
        m = np.zeros(len(seg), dtype=bool)
        while m.mean() < 0.5:
            s = int(rng.integers(0, len(seg))); m[s:s + int(rng.integers(10, 150))] = True
        seg[m] = np.nan
        seg += rng.normal(0, 0.3 / 105.3, seg.shape)
        parts.append(seg)
        t_of_b.append(np.arange(a, b))
    bc = np.concatenate(parts)
    if flip:
        bc = 1.0 - bc
    return tact, bc, np.concatenate(t_of_b), truth_ball, miss


def test_alignment_recovers_offsets_including_replay():
    tact, bc, t_true, truth, miss = _make()
    al = broadcast.align(tact, bc)
    ok = broadcast.accept(al)
    t_of_b = broadcast.verify_map(broadcast.frame_map(al, ok, len(bc)), tact, bc, False)
    mapped = t_of_b >= 0
    assert mapped.mean() > 0.3          # stationary-ball stretches are ambiguous and rightly left unmapped
    err = np.abs(t_of_b[mapped] - t_true[mapped])
    assert (err <= 10).mean() > 0.99 and (err <= 2).mean() > 0.9   # accepted windows map to the right tactical frame


def test_fusion_fills_gaps_accurately_and_leaves_rest_untouched():
    tact, bc, t_true, truth, miss = _make(seed=5)
    al, flip = broadcast.best_orientation(tact, bc, log=lambda *a: None)
    assert flip is False
    ok = broadcast.accept(al)
    t_of_b = broadcast.verify_map(broadcast.frame_map(al, ok, len(bc)), tact, bc, flip)
    fused, filled, stats = broadcast.fuse_ball(tact, bc, t_of_b, flip, len(tact))
    assert stats["tactical_ball_gaps_filled"] > 500
    err = np.hypot(*((fused[filled] - truth[filled]) * broadcast.PITCH).T)
    assert np.median(err) < 1.0 and np.percentile(err, 95) < 5.0
    obs = np.isfinite(tact[:, 0])
    assert np.allclose(fused[obs], tact[obs], equal_nan=True)     # tactical observations are never overwritten
    assert stats["agreement_median_m"] < 1.0


def test_flipped_broadcast_orientation_is_detected():
    tact, bc, t_true, truth, miss = _make(seed=7, flip=True)
    al, flip = broadcast.best_orientation(tact, bc, log=lambda *a: None)
    assert flip is True


def test_unrelated_broadcast_maps_nothing():
    tact, _, _, _, _ = _make(seed=9)
    rng = np.random.default_rng(1)
    other = np.stack([rng.uniform(0.1, 0.9, 6000), rng.uniform(0.1, 0.9, 6000)], axis=1)
    other[rng.random(6000) < 0.5] = np.nan
    al = broadcast.align(tact, other)
    assert broadcast.accept(al).sum() == 0
