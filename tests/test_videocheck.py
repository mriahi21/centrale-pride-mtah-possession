import numpy as np
import pandas as pd

import poss  # noqa: F401
from poss import videocheck

N = 60000


def _smooth(x, w=51):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def _world(seed=0, extra=0.25, shift=0, camera_noise=0.35):
    """Synthetic tracking activity, stoppages, ball-out-of-bounds, a homography with a panning stretch, and a video cue."""
    rng = np.random.default_rng(seed)
    stop = np.zeros(N, dtype=bool)
    oob = np.zeros(N, dtype=bool)
    starts = []
    for s in range(2000, N - 2000, 1500):
        s = s + int(rng.integers(-300, 300))
        stop[s:s + 320] = True
        oob[s:s + 120] = True
        starts.append(s)
    act = _smooth(np.where(stop, 0.45, 1.0) + rng.normal(0, 0.25, N), 75)
    H = np.tile(np.array([[1920.0, 0, 0], [0, 1080.0, 0], [0, 0, 1.0]]), (N, 1, 1))
    pan = np.zeros(N)
    tx = np.cumsum(np.where((np.arange(N) >= 20000) & (np.arange(N) < 28000), 1.2, 0.0))   # fast pan: 6 px per 5 frames
    H[:, 0, 2] = tx
    pan[20000:28000] = 1.0
    video = 0.25 * act - extra * stop + rng.normal(0, 0.03, N) + camera_noise * _smooth(pan, 25) * rng.random(N)
    video = _smooth(video, 25)
    video = np.roll(video, shift)
    hom = {"H": H, "err": np.full(N, 0.25, dtype=np.float32), "valid": np.ones(N, dtype=bool)}
    return {"act": act, "stop": stop, "oob": oob, "hom": hom, "video": video, "starts": np.array(starts), "pan": pan}


def _verify(w, video=None):
    g = np.ones(N, dtype=bool)
    ref = {"oob": w["oob"], "dead": w["stop"]}
    return videocheck.verify_video(w["video"] if video is None else video, w["act"], w["hom"], w["hom"]["valid"], g, ref)


def test_camera_motion_flags_the_panning_stretch():
    w = _world()
    pan, zoom = videocheck.camera_motion(w["hom"])
    still = videocheck.camera_still(w["hom"]["valid"], pan, zoom)
    assert still[:19000].all() and not still[21000:27000].any()
    assert 0.7 < still.mean() < 0.9


def test_a_real_cue_passes_and_is_gated_off_where_the_camera_pans():
    w = _world()
    res = _verify(w)
    assert res["verdict"], res["lines"]
    assert np.isnan(res["gated"][21000:27000]).all()
    assert np.isfinite(res["gated"][:19000]).all()


def test_pure_noise_cue_fails_negative_controls():
    w = _world()
    rng = np.random.default_rng(5)
    res = _verify(w, _smooth(rng.normal(0.2, 0.05, N), 25))
    assert not res["verdict"]


def test_clock_offset_is_detected():
    w = _world(shift=60)                      # video runs 2.4 s off the tracking clock
    res = _verify(w)
    assert not res["checks"]["clock"] and not res["verdict"]


def test_cue_that_only_repeats_tracking_activity_adds_no_incremental_evidence():
    w = _world(extra=0.0)                     # video energy = f(activity) only: nothing the tracking does not already say
    res = _verify(w)
    assert not res["checks"]["incremental"] and not res["verdict"]


def test_whistles_on_stoppage_starts_pass_and_random_ones_fail():
    w = _world()
    good = videocheck.verify_whistle(list(w["starts"]), w["act"], np.ones(N, bool), w["starts"], N)
    assert good["verdict"], good["lines"]
    rng = np.random.default_rng(2)
    rand = videocheck.verify_whistle(list(rng.integers(500, N - 500, 40)), w["act"], np.ones(N, bool), w["starts"], N)
    assert not rand["verdict"]
