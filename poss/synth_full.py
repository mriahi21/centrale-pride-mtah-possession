"""Full-size SYNTHETIC bundle (tactical + broadcast + homography + low-res video + audio) for smoke-testing the
full-bundle workflow. Contains no match data. Used by scripts/smoke_full_synthetic.py.
"""
import os
import subprocess
import wave

import numpy as np

import poss  # noqa: F401
from games import SOURCES, asset_paths
from poss import media, synth

L, W = synth.L, synth.W


def write_homography(path, n, size=(64, 36)):
    import pandas as pd
    H = [size[0], 0, 0, 0, size[1], 0, 0, 0, 1.0]
    rows = np.tile(np.array(H + [0.25, 1.0]), (n, 1))
    df = pd.DataFrame(rows, columns=["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
                                     "reprojection_error", "valid"])
    df.insert(0, "frame", np.arange(n))
    df["valid"] = df["valid"].astype(int)
    df.to_csv(path, index=False)


def write_broadcast(sim, paths, n_bc, seed=0):
    rng = np.random.default_rng(seed)
    truth = sim["ball_clean"] / np.array([L, W])
    parts = []
    for half, a, b in sim["blocks"]:
        cuts = np.sort(rng.integers(a + 2000, b - 2000, 4))
        edges = [a] + list(cuts) + [b + 1]
        for s, e in zip(edges[:-1], edges[1:]):
            parts.append((s, e - int(rng.integers(100, 400))))        # drop a bit at each cut
    parts.append((20000, 22500))                                      # replays of earlier moments
    parts.append((60000, 62000))
    seq = np.concatenate([np.arange(s, e) for s, e in parts])
    seq = seq[:n_bc] if len(seq) >= n_bc else np.concatenate([seq, np.arange(100000, 100000 + n_bc - len(seq))])
    bc = truth[seq].copy()
    bc[np.isnan(bc[:, 0])] = np.nan
    miss = np.zeros(len(bc), dtype=bool)
    while miss.mean() < 0.5:
        s = int(rng.integers(0, len(bc)))
        miss[s:s + int(rng.integers(10, 150))] = True
    bc[miss] = np.nan
    bc += rng.normal(0, 0.3 / L, bc.shape)
    groups = ";".join(["NaN,NaN,NaN"] * synth.N_SLOTS)
    with open(paths["atd_raw"], "w") as fh:
        for f in range(len(bc)):
            ball = "NaN,NaN,NaN" if np.isnan(bc[f, 0]) else "{:.5f},{:.5f},0.0".format(bc[f, 0], bc[f, 1])
            fh.write("{}:{}:{}\n".format(f + 1, groups, ball))
    return seq


def write_video_with_audio(sim, path, size=(64, 36), seed=0):
    import cv2
    n = sim["n_frames"]
    tmp = path + ".silent.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    P = sim["players"] / np.array([L, W])
    for f in range(n):
        img = np.full((size[1], size[0], 3), 60, dtype=np.uint8)
        pts = P[f][np.isfinite(P[f][:, 0])]
        for x, y in pts:
            cv2.circle(img, (int(np.clip(x * size[0], 0, size[0] - 1)), int(np.clip(y * size[1], 0, size[1] - 1))), 1,
                       (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    sr = 16000
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.03, int(n / 25 * sr)).astype(np.float32)
    truth = sim["truth"]
    starts = np.flatnonzero((truth[1:] == "D") & (truth[:-1] != "D") & (truth[:-1] != "X")) + 1
    t = np.arange(sr) / sr
    for s in starts:
        i = int(s / 25 * sr)
        x[i:i + sr] += (0.4 * np.sin(2 * np.pi * 3600 * t))[:len(x[i:i + sr])].astype(np.float32)
    wav = path + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run([media.ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-shortest", path], check=True)
    os.remove(tmp)
    os.remove(wav)


def build(root, seed=0, with_video=True):
    blocks = ((1, 11185, 90651), (2, 110808, 190197))
    sim = synth.simulate(n_frames=199319, blocks=blocks, seed=seed, second_half_pre_dead=2934)
    synth.write_bundle(sim, root)
    tp = asset_paths(data_dir=root)
    write_homography(tp["homography"], 199319)
    bp = asset_paths(data_dir=root, source="broadcast")
    os.makedirs(os.path.dirname(bp["atd_raw"]), exist_ok=True)
    write_broadcast(sim, bp, SOURCES["broadcast"]["n_video_frames"], seed)
    import shutil
    shutil.copy(tp["atd_meta"], bp["atd_meta"])
    write_homography(bp["homography"], SOURCES["broadcast"]["n_video_frames"])
    if with_video:
        write_video_with_audio(sim, tp["video"], seed=seed)
    return sim
