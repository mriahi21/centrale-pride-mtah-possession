"""Pitch motion energy from the tactical video (optional cue; independent of tracking errors).

For every `stride`-th frame: grayscale at low resolution, absolute difference to the previous sampled frame,
fraction of pixels inside the pitch mask whose change exceeds a threshold. Stoppages show much less motion
than live play. The mask comes from the homography (pitch rectangle projected into the image), so the crowd /
benches / scoreboard are excluded. The result is a per-frame series on the video timeline; no pixels are kept.
"""
import numpy as np
import pandas as pd

import poss  # noqa: F401

SIZE = (480, 270)          # (w, h) analysis resolution
STRIDE = 5                 # 5 fps
DIFF_THRESH = 18


def pitch_mask(poly, full_size=(1920, 1080), size=SIZE, shrink=0.97):
    import cv2
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    if poly is None:
        mask[:] = 1
        return mask
    pts = poly.copy()
    c = pts.mean(0)
    pts = c + (pts - c) * shrink
    pts = pts * np.array([size[0] / full_size[0], size[1] / full_size[1]])
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def video_activity(path, mask, stride=STRIDE, size=SIZE, thresh=DIFF_THRESH, max_frames=None, log=print):
    """(frame_index [m], activity [m]) for sampled frames. Decodes the whole file sequentially."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: " + path)
    area = float(mask.sum()) or 1.0
    idx, act = [], []
    prev = None
    i = 0
    while True:
        if max_frames is not None and i >= max_frames:
            break
        if i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            act.append(float(((d > thresh) & (mask > 0)).sum()) / area)
            idx.append(i)
        prev = g
        i += 1
        if i % 20000 == 0:
            log("[video] decoded {} frames".format(i))
    cap.release()
    return np.array(idx, dtype=np.int64), np.array(act, dtype=np.float32)


def to_frame_series(idx, act, n_frames, smooth_s=2.0, fps=25):
    """Interpolate sampled activity to every frame and smooth with a rolling median."""
    if len(idx) < 2:
        return np.full(n_frames, np.nan, dtype=np.float32)
    s = np.interp(np.arange(n_frames), idx, act, left=np.nan, right=np.nan)
    win = int(smooth_s * fps) | 1
    return pd.Series(s).rolling(win, center=True, min_periods=1).median().to_numpy(dtype=np.float32)


def lag_correlation(a, b, max_lag=50, step=5):
    """Pearson correlation of two equally long series at lags -max_lag..max_lag (frames). Returns (best_lag, r_best, r0).
    Used to verify that the video's motion energy is aligned with the tracking's activity (alignment check)."""
    m = np.isfinite(a) & np.isfinite(b)
    best, rb, r0 = 0, -2.0, np.nan
    for lag in range(-max_lag, max_lag + 1, step):
        if lag >= 0:
            x, y, mm = a[lag:], b[:len(b) - lag], m[lag:] & m[:len(m) - lag]
        else:
            x, y, mm = a[:lag], b[-lag:], m[:lag] & m[-lag:]
        if mm.sum() < 1000:
            continue
        r = float(np.corrcoef(x[mm], y[mm])[0, 1])
        if lag == 0:
            r0 = r
        if r > rb:
            best, rb = lag, r
    return best, rb, r0
