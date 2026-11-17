import numpy as np
import pytest

import poss  # noqa: F401
from poss import vidpass

cv2 = pytest.importorskip("cv2")

NT, NB = 600, 450
SIZE = (960, 540)
PITCH = vidpass.PITCH


def _homography(nb):
    H = np.tile(np.array([[900.0, 0, 30], [0, 480.0, 30], [0, 0, 1.0]]), (nb, 1, 1))
    return {"H": H, "err": np.full(nb, 0.3, dtype=np.float32), "valid": np.ones(nb, dtype=bool)}


def _truth():
    t = np.arange(NT)
    x = 0.5 + 0.38 * np.sin(t / 55.0)
    y = 0.5 + 0.30 * np.sin(t / 37.0 + 1.0)
    return np.stack([x, y], axis=1)


def _t_of_b():
    m = np.full(NB, -1, dtype=np.int64)
    m[:300] = np.arange(300) + 50              # tactical 50..349
    m[320:] = np.arange(320, NB) + 100         # tactical 420..549 (a cut of 20 frames between)
    return m


def _render(path, truth, t_of_b, clutter=True):
    hom = _homography(NB)
    H = hom["H"][0]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, SIZE)
    rng = np.random.default_rng(0)
    for b in range(NB):
        img = np.full((SIZE[1], SIZE[0], 3), (40, 130, 40), dtype=np.uint8)
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        pts = [tuple(int(v) for v in vidpass.homog.project(H, cx, cy)) for cx, cy in corners]
        for a, c in zip(pts, pts[1:] + pts[:1]):
            cv2.line(img, a, c, (235, 235, 235), 2)
        cv2.line(img, vidpass.homog.project(H, 0.5, 0.0) and tuple(int(v) for v in vidpass.homog.project(H, 0.5, 0.0)),
                 tuple(int(v) for v in vidpass.homog.project(H, 0.5, 1.0)), (235, 235, 235), 2)
        for k in range(8):                                       # players: strongly coloured, larger than the ball
            px, py = 0.15 + 0.1 * k, 0.5 + 0.3 * np.sin(b / 30.0 + k)
            u, v = vidpass.homog.project(H, px, py)
            cv2.circle(img, (int(u), int(v)), 8, (30, 30, 200) if k % 2 else (200, 60, 30), -1)
        if clutter:                                              # a static white spot (penalty mark / sign)
            u, v = vidpass.homog.project(H, 0.9, 0.85)
            cv2.circle(img, (int(u), int(v)), 3, (250, 250, 250), -1)
        t = t_of_b[b]
        if t >= 0:
            u, v = vidpass.homog.project(H, truth[t, 0], truth[t, 1])
            cv2.circle(img, (int(round(u)), int(round(v))), 4, (255, 255, 255), -1)
        img = np.clip(img.astype(np.int16) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
    vw.release()
    return hom


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    truth, tob = _truth(), _t_of_b()
    path = str(d / "bc.mp4")
    hom = _render(path, truth, tob)
    return {"path": path, "hom": hom, "ok": np.ones(NB, dtype=bool), "truth": truth, "t_of_b": tob, "dir": str(d)}


def test_map_interval_accepts_clean_stretch_and_rejects_cut_boundaries(world):
    tob = world["t_of_b"]
    w, why = vidpass.map_interval(100, 250, tob, world["ok"])
    assert w and w["b0"] == 50 and w["b1"] == 200
    w, why = vidpass.map_interval(300, 440, tob, world["ok"])            # straddles the cut: tactical 350..419 has no video
    assert w is None
    bad = world["ok"].copy()
    bad[:] = False
    w, why = vidpass.map_interval(100, 250, tob, bad)
    assert w is None and "homography" in why


def test_detector_recovers_the_ball_and_ignores_clutter_and_players(world):
    w, _ = vidpass.map_interval(60, 340, world["t_of_b"], world["ok"])
    cand, cached = vidpass.get_candidates(world["path"], w, "t", world["dir"], world["hom"], world["ok"], SIZE)
    assert not cached
    cand2, cached2 = vidpass.get_candidates(world["path"], w, "t", world["dir"], world["hom"], world["ok"], SIZE)
    assert cached2 and np.array_equal(cand, cand2)                       # resumable: second call reads the cache
    pos_list = vidpass.window_positions(cand, w, world["hom"], world["t_of_b"], False, NT)
    assert pos_list
    # at least one track follows the true ball; the static spot is a separate track
    errs = []
    for pos in pos_list:
        m = np.isfinite(pos[:, 0])
        errs.append((np.median(np.hypot(*((pos[m] - world["truth"][m]) * PITCH).T)), m.sum()))
    best = min(errs, key=lambda e: e[0])
    assert best[0] < 1.0 and best[1] > 150


def test_anchoring_rejects_the_clutter_track_and_fills_the_gap_accurately(world):
    truth = world["truth"]
    ref = truth.copy()
    ref[150:250] = np.nan                                                # a 100-frame tactical gap (4 s) with observed flanks
    w, _ = vidpass.map_interval(120, 280, world["t_of_b"], world["ok"])
    cand, _ = vidpass.get_candidates(world["path"], w, "a", world["dir"], world["hom"], world["ok"], SIZE)
    pos_list = vidpass.window_positions(cand, w, world["hom"], world["t_of_b"], False, NT)
    filled = vidpass.anchored_fill(ref, pos_list, w["t0"], w["t1"])
    m = np.isfinite(filled[:, 0])
    assert m[150:250].mean() > 0.7 and not m[:150].any() and not m[250:].any()   # only gap frames are filled
    err = np.hypot(*((filled[m] - truth[m]) * PITCH).T)
    assert np.median(err) < 1.0 and np.percentile(err, 95) < 3.0
    # a wrong-place track must not pass the flank-consistency test
    fake = np.full((NT, 2), np.nan)
    fake[160:240] = [0.9, 0.85]
    assert not np.isfinite(vidpass.anchored_fill(ref, [fake], w["t0"], w["t1"])[:, 0]).any()


def test_video_holdout_verifies_detector_and_measures_gain(world):
    truth = world["truth"]
    stretches = [(70, 100), (200, 100), (430, 100)]
    masked = truth.copy()
    for s, L in stretches:
        masked[s:s + L] = np.nan
    nothing = np.full((NT, 2), np.nan)
    res = vidpass.video_holdout(world["path"], world["hom"], world["ok"], truth, masked, stretches, world["t_of_b"], False,
                                nothing, nothing, world["dir"], max_windows=6, hom_size=SIZE, log=lambda *a: None)
    assert res["windows_mapped"] == 3
    assert res["accepted"], res
    assert res["median"] < 1.0 and res["gain"] > 0.5 and res["promote"]
    assert res["flip"] is False                                           # orientation chosen from held-out error


def test_video_holdout_rejects_a_mismatched_orientation_or_garbage(world):
    truth = world["truth"]
    stretches = [(70, 100), (200, 100), (430, 100)]
    masked = truth.copy()
    for s, L in stretches:
        masked[s:s + L] = np.nan
    wrong = np.full(NB, -1, dtype=np.int64)                              # alignment says video shows a different moment
    wrong[:300] = np.arange(300) + 250
    wrong[320:] = np.arange(320, NB) - 200
    nothing = np.full((NT, 2), np.nan)
    res = vidpass.video_holdout(world["path"], world["hom"], world["ok"], truth, masked, stretches, wrong, False,
                                nothing, nothing, world["dir"], max_windows=6, hom_size=SIZE, log=lambda *a: None)
    assert not res["promote"]


def test_window_selection_prioritises_long_gaps_and_counts_rejections(world):
    ball = world["truth"].copy()
    ball[100:200] = np.nan                                               # gap inside clean video
    ball[330:430] = np.nan                                               # gap over the cut
    blocks = ((1, 0, 599),)
    wins, rej = vidpass.select_windows(NT, ball, blocks, world["t_of_b"], world["ok"],
                                       restarts=[(500, 0.0, True)], turnovers=[(300, 0.1)], min_gap=25)
    kinds = {w["kind"] for w in wins}
    assert "gap" in kinds
    assert sum(rej.values()) >= 1                                        # the gap over the cut was rejected, with a reason
    assert all(w["b1"] - w["b0"] >= vidpass.MIN_WIN for w in wins)


def _orchestrate(world, tmp_path, monkeypatch, size_ok=True):
    import json
    import shutil
    from types import SimpleNamespace

    import pandas as pd
    from games import GAME, asset_paths
    from poss import fullrun, pipeline

    root = str(tmp_path)
    bp = asset_paths(GAME, root, "broadcast")
    import os
    os.makedirs(os.path.dirname(bp["video"]), exist_ok=True)
    shutil.copy(world["path"], bp["video"])
    H = world["hom"]["H"].reshape(NB, 9)
    df = pd.DataFrame(H, columns=["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"])
    df["reprojection_error"] = 0.3
    df["valid"] = 1
    df.insert(0, "frame", np.arange(NB))
    df.to_csv(bp["homography"], index=False)
    cache = os.path.join(root, "cache")
    os.makedirs(cache)
    stretches = [(70, 100), (200, 100), (430, 100)]
    np.savez_compressed(os.path.join(cache, "holdout_arrays.npz"), t_of_b=world["t_of_b"], flip=False,
                        bc_pos=np.full((NT, 2), np.nan, dtype=np.float32), interp=np.full((NT, 2), np.nan, dtype=np.float32))
    with open(os.path.join(cache, "holdout_ball.json"), "w") as fh:
        json.dump({"stretches": stretches}, fh)
    truth = world["truth"]
    data = {"ball": truth.copy()}
    ball_out = truth.copy()
    ball_out[100:190] = np.nan                                           # a gap the video pass should fill
    data_out = {"ball": ball_out}
    monkeypatch.setattr(pipeline, "prepare_halves", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "label_halves", lambda *a, **k: (np.zeros(NT, dtype=np.int8), {}))
    monkeypatch.setattr(fullrun.vidpass.homog, "ERR_OK", 2.0)
    inv = [{"source": "broadcast", "asset": "video", "size_ok": size_ok, "where": "data_root"}]
    state = {"broadcast": {"map": {"t_of_b": world["t_of_b"]}}, "cache_dir": cache}
    args = SimpleNamespace(no_video_pass=False, video_budget_min=5.0, video_windows=10)
    rep = []
    res, lines = fullrun.run_video_pass(root, data, data_out, {}, None, ((1, 0, NT - 1),), NT, [], state, inv, args, rep,
                                        lambda *a: None)
    return res, lines, rep, truth, ball_out


def test_orchestration_fills_the_gap_when_the_detector_is_verified(world, tmp_path, monkeypatch):
    res, lines, rep, truth, ball_out = _orchestrate(world, tmp_path, monkeypatch)
    assert res and res["hold"]["promote"], lines
    ball = res["ball"]["ball"]
    filled = np.isfinite(ball[100:190, 0])
    assert filled.mean() > 0.6
    err = np.hypot(*((ball[100:190][filled] - truth[100:190][filled]) * PITCH).T)
    assert np.median(err) < 1.0
    assert np.allclose(ball[:100], truth[:100])                          # observed frames never touched
    assert any(r["asset"] == "broadcast/video" and r["processed_ok"] for r in rep)


def test_orchestration_skips_an_incomplete_download(world, tmp_path, monkeypatch):
    res, lines, rep, _, _ = _orchestrate(world, tmp_path, monkeypatch, size_ok=False)
    assert res is None and "not complete" in lines[0]
    assert any(r["asset"] == "broadcast/video" and not r["processed_ok"] and r["blocker"] for r in rep)
