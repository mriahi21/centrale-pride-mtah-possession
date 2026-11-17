"""Homography loading, per-frame reliability, and a tracking-vs-image consistency check."""
import numpy as np
import pandas as pd

import poss  # noqa: F401

ERR_OK = 2.0          # reprojection error below which a matrix is trusted (median in play is ~0.24)
IMG_W, IMG_H = 1920, 1080


def load(path):
    """{H [F,3,3] (NaN where invalid), err [F], valid [F] bool}. Rows are video frames (frame column checked)."""
    df = pd.read_csv(path)
    frames = df["frame"].to_numpy()
    if not np.array_equal(frames, np.arange(len(df))):
        raise ValueError("homography 'frame' column is not 0..N-1 in order")
    H = df[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return {"H": H, "err": df["reprojection_error"].to_numpy(dtype=np.float32),
            "valid": df["valid"].to_numpy().astype(bool)}


def reliable(hom, err_ok=ERR_OK):
    return hom["valid"] & np.isfinite(hom["err"]) & (hom["err"] <= err_ok)


def project(H, x, y):
    """Normalised pitch (x, y) -> pixels with one 3x3 matrix; arrays broadcast."""
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w


def pitch_polygon(hom, frame):
    """Pixel corners of the pitch rectangle at one frame (4x2), or None if the matrix is invalid."""
    H = hom["H"][frame]
    if not np.isfinite(H).all():
        return None
    xs, ys = project(H, np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]))
    return np.stack([xs, ys], axis=1)


def consistency(hom, players_norm, sample=500, margin=100):
    """Fraction of tracked players whose projected pixel lies inside the image (+margin), over sampled reliable
    frames. High = the homography and the tracking agree on the same pitch/camera geometry."""
    ok = np.flatnonzero(reliable(hom))
    if not len(ok):
        return {"frames_checked": 0, "inside_frac": float("nan")}
    pick = ok[np.linspace(0, len(ok) - 1, min(sample, len(ok))).astype(int)]
    inside, total = 0, 0
    for f in pick:
        p = players_norm[f]
        m = np.isfinite(p[:, 0])
        if not m.any():
            continue
        px, py = project(hom["H"][f], p[m, 0], p[m, 1])
        inside += int(np.sum((px > -margin) & (px < IMG_W + margin) & (py > -margin) & (py < IMG_H + margin)))
        total += int(m.sum())
    return {"frames_checked": int(len(pick)), "inside_frac": inside / total if total else float("nan")}
