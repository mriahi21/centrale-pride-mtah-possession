"""Automatic whistle detection from the video's audio track (optional supporting cue).

A referee's whistle is a loud, narrow-band, sustained tone around 2-4.5 kHz; crowd noise is broadband. Score per
20 ms hop = (band energy share) x (peakiness in band); events = sustained robust-z excursions >= 0.24 s. No
labels, no training. If the video has no audio track the cue is simply absent.
"""
import subprocess

import numpy as np

import poss  # noqa: F401
from poss.media import ffmpeg_exe

SR = 16000
N_FFT, HOP = 1024, 320            # 64 ms window, 20 ms hop
BAND_HZ = (2500.0, 4500.0)
MIN_HOPS = 12                      # 0.24 s
MERGE_S = 0.5
Z_THRESH = 7.0


def decode_audio(path, sr=SR, max_seconds=None):
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "s16le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def whistle_scores(x, sr=SR):
    """Per-hop whistle score (>= 0) over the whole signal, computed in blocks to bound memory."""
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    n_hops = max(0, (len(x) - N_FFT) // HOP + 1)
    out = np.zeros(n_hops, dtype=np.float32)
    block = 4000
    for s in range(0, n_hops, block):
        e = min(n_hops, s + block)
        idx = (np.arange(s, e)[:, None] * HOP) + np.arange(N_FFT)[None, :]
        P = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        tot = P.sum(1) + 1e-9
        pb = P[:, band]
        share = pb.sum(1) / tot
        peaky = pb.max(1) / (pb.mean(1) + 1e-12)
        out[s:e] = share * np.minimum(peaky, 60.0) * (tot > 1e-3 * N_FFT)
    return out


def whistle_events(x, sr=SR):
    """[(start_s, end_s, strength)] sustained whistle-like events (robust z-score vs the whole recording)."""
    s = whistle_scores(x, sr)
    if len(s) < 100:
        return []
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-6
    z = (s - med) / (1.4826 * mad)
    on = z > Z_THRESH
    ev = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= MIN_HOPS:
                ev.append([i * HOP / sr, (j * HOP + N_FFT) / sr, float(z[i:j].mean())])
            i = j
        else:
            i += 1
    merged = []
    for a, b, st in ev:
        if merged and a - merged[-1][1] < MERGE_S:
            merged[-1][1] = b
            merged[-1][2] = max(merged[-1][2], st)
        else:
            merged.append([a, b, st])
    return [tuple(m) for m in merged]


def whistle_cue(n_frames, events, fps=25, offset_s=0.0, lead_s=1.0, lag_s=4.0):
    """0..1 array on the video timeline: high from `lead_s` before to `lag_s` after each whistle's start
    (a whistle stops or restarts play; the exact meaning is left to the corroborated stoppage logic)."""
    cue = np.zeros(n_frames, dtype=np.float32)
    for a, b, st in events:
        f0 = int(round((a + offset_s - lead_s) * fps))
        f1 = int(round((a + offset_s + lag_s) * fps))
        cue[max(f0, 0):max(min(f1, n_frames), 0)] = 1.0
    return cue
