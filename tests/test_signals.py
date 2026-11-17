import os

import numpy as np
import pytest

import poss  # noqa: F401
from poss import audio, bundle, deadball2, homog, scenarios as sc, video_activity


def _noise_with_whistles(seconds=60, sr=16000, starts=(10.0, 25.0, 41.0), dur=1.2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.05, seconds * sr).astype(np.float32)              # crowd-like broadband noise
    t = np.arange(int(dur * sr)) / sr
    for s in starts:
        i = int(s * sr)
        x[i:i + len(t)] += 0.4 * np.sin(2 * np.pi * 3600 * t + 0.5 * np.sin(2 * np.pi * 6 * t)).astype(np.float32)
    return x


def test_whistle_detection_finds_tonal_bursts_not_noise():
    x = _noise_with_whistles()
    ev = audio.whistle_events(x)
    starts = [e[0] for e in ev]
    assert len(ev) == 3
    for s, truth in zip(starts, (10.0, 25.0, 41.0)):
        assert abs(s - truth) < 0.3
    assert audio.whistle_events(np.random.default_rng(1).normal(0, 0.05, 16000 * 60).astype(np.float32)) == []


def test_whistle_cue_window():
    cue = audio.whistle_cue(2000, [(10.0, 11.0, 9.0)], fps=25)
    assert cue[int(10 * 25)] == 1.0 and cue[int(9.2 * 25)] == 1.0 and cue[int(15.5 * 25)] == 0.0 and cue[200] == 0.0


def test_homography_reliability_and_projection(tmp_path):
    import pandas as pd
    rows = []
    for f in range(6):
        valid = f != 3
        h = [1920.0, 0, 0, 0, 1080.0, 0, 0, 0, 1.0] if valid else [np.nan] * 9
        rows.append([f] + h + [0.3 if f != 4 else 5.0, int(valid)])
    df = pd.DataFrame(rows, columns=["frame", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
                                     "reprojection_error", "valid"])
    p = tmp_path / "h.csv"
    df.to_csv(p, index=False)
    hom = homog.load(str(p))
    ok = homog.reliable(hom)
    assert list(ok) == [True, True, True, False, False, True]
    x, y = homog.project(hom["H"][0], 0.5, 0.5)
    assert (x, y) == (960.0, 540.0)
    poly = homog.pitch_polygon(hom, 0)
    assert poly.shape == (4, 2) and homog.pitch_polygon(hom, 3) is None
    players = np.full((6, 3, 2), np.nan)
    players[:, 0] = [0.5, 0.5]
    assert homog.consistency(hom, players)["inside_frac"] == 1.0


def test_bundle_inventory_flags_missing_and_truncated(tmp_path):
    rows = bundle.rows(str(tmp_path))
    assert len(rows) == 9 and all(r["where"] == "missing" for r in rows)
    folder = tmp_path / "tactical"
    folder.mkdir()
    (folder / "BAR-ATH_tactical_atd.xml").write_bytes(b"x" * 10)           # wrong size
    r = [x for x in bundle.rows(str(tmp_path)) if x["asset"] == "atd_meta" and x["source"] == "tactical"][0]
    assert r["where"] == "data_root" and not r["size_ok"]


def test_adopt_downloads_moves_only_complete_files(tmp_path):
    dl = tmp_path / "dl"
    dl.mkdir()
    (dl / "BAR-ATH_broadcast_atd.xml").write_bytes(b"x" * 64956)           # exact expected size
    (dl / "BAR-ATH_tactical_atd.xml").write_bytes(b"x" * 5)                # truncated -> left alone
    n = bundle.adopt_downloads(str(tmp_path / "root"), str(dl), log=lambda *a: None)
    assert n == 1 and (tmp_path / "root" / "broadcast" / "BAR-ATH_broadcast_atd.xml").exists()
    assert (dl / "BAR-ATH_tactical_atd.xml").exists()


def test_v3_extras_are_weak_cues_never_enter_dead_alone():
    ft, truth, tag = sc.scenario("baseline", 1)
    n = ft["n"]
    tag[:] = 0.0
    ft["whistle"] = np.where(~truth, 1.0, 0.0)               # whistle claims a stoppage everywhere in live play
    v = np.where(truth, 0.01, 0.2)
    ft["vquiet"] = v + np.random.default_rng(0).normal(0, 0.002, n)
    ft["hom_ok"] = np.ones(n, dtype=bool)
    pred, flags = deadball2.dead_mask(ft, 0, tag, log=lambda *a: None, extras=True)
    assert pred[~truth].mean() < 0.05                         # whistle alone cannot split live play
    assert flags["profile"] == "v3"


def test_v3_extras_corroborate_a_stoppage_with_an_untracked_ball():
    ft, truth, tag = sc.scenario("stoppage_ball_occluded", 1)
    tag[:] = 0.0                                              # no tag
    ft["activity"] = np.full(ft["n"], 2.0, dtype=np.float32)  # ATD quiet cue neutral
    ft["whistle"] = np.where(truth, 1.0, 0.0)
    ft["vquiet"] = np.where(truth, 0.01, 0.3).astype(float)
    ft["hom_ok"] = np.ones(ft["n"], dtype=bool)
    off, _ = deadball2.dead_mask(ft, 0, tag, log=lambda *a: None, extras=False)
    on, _ = deadball2.dead_mask(ft, 0, tag, log=lambda *a: None, extras=True)
    assert on[truth].mean() > 0.8 > off[truth].mean() + 0.5   # two independent weak cues enter; none alone do


def test_video_activity_on_synthetic_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "clip.mp4")
    w, h, fps, n = 320, 180, 25, 200
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("no mp4 writer in this OpenCV build")
    rng = np.random.default_rng(0)
    pos = rng.uniform(40, 280, (12, 2))
    for f in range(n):
        img = np.full((h, w, 3), 90, dtype=np.uint8)
        if f < 100:                                          # live: 12 blobs moving
            pos += rng.normal(0, 6, pos.shape)
        for x, y in pos:
            cv2.circle(img, (int(np.clip(x, 5, w - 5)), int(np.clip(y, 5, h - 5))), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    mask = video_activity.pitch_mask(None, (w, h), size=(w, h))
    idx, act = video_activity.video_activity(path, mask, stride=5, size=(w, h), log=lambda *a: None)
    assert len(idx) > 30
    assert act[idx < 95].mean() > 5 * max(act[idx > 110].mean(), 1e-4)       # motion then stillness
    series = video_activity.to_frame_series(idx, act, n)
    assert series.shape == (n,)
    ta = np.where(np.arange(1500) < 750, 2.0, 0.5) + np.random.default_rng(1).normal(0, 0.05, 1500)
    va = np.where(np.arange(1500) < 750, 0.2, 0.02) + np.random.default_rng(2).normal(0, 0.005, 1500)
    lag, r, r0 = video_activity.lag_correlation(va, ta, max_lag=50, step=5)
    assert abs(lag) <= 5 and r > 0.9
