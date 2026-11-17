"""Fast, NaN-safe reader for the raw ATD text file (FIFA/EPTS export).

Line format (see starter/baseline.py): ``<frame_id>:<slot>;<slot>;...:<ball>`` where each group is
``x,y,speed`` (normalised 0-1 coordinates, speed in m/s) and a missing observation is NaN.
``video_frame = frame_id - 1``.

Track slots are tracklets, NOT player identities -- nothing here (or downstream) treats a slot
index as a person. A slot only carries a team label.
"""
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np

import poss  # noqa: F401  (sys.path)


def team_of_slots(meta_path):
    """int8 array over slots: 0 = home, 1 = away (first <Team> in the metadata is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def _floats(text):
    """Parse a comma separated numeric string (NaN allowed) -> float32 array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.fromstring(text, dtype=np.float32, sep=",")


def _parse_players_slow(text, n_slots):
    out = np.full((n_slots, 3), np.nan, dtype=np.float32)
    for i, group in enumerate(text.split(";")):
        if i >= n_slots:
            break
        vals = [float(v) if v.strip().lower() != "nan" else np.nan for v in group.split(",") if v.strip()]
        if len(vals) >= 2:
            out[i, :min(3, len(vals))] = vals[:3]
    return out


def read_atd(raw_path, n_frames):
    """Return dict(players [F,S,2], pspeed [F,S], ball [F,2], bspeed [F]) on the video timeline.

    Missing frames/observations are NaN. Coordinates stay normalised (0-1).
    """
    with open(raw_path) as handle:
        first = handle.readline().split(":")
        n_slots = len(first[1].split(";"))
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    pspeed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    bspeed = np.full(n_frames, np.nan, dtype=np.float32)

    n_read = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1
            if not 0 <= frame < n_frames:
                continue
            arr = _floats(parts[1].replace(";", ","))
            if arr.size == n_slots * 3:
                arr = arr.reshape(n_slots, 3)
            else:
                arr = _parse_players_slow(parts[1], n_slots)
            players[frame] = arr[:, :2]
            pspeed[frame] = arr[:, 2]
            b = _floats(parts[2].strip(";"))
            if b.size >= 2:
                ball[frame] = b[:2]
                if b.size >= 3:
                    bspeed[frame] = b[2]
            n_read += 1
    return {"players": players, "pspeed": pspeed, "ball": ball, "bspeed": bspeed,
            "n_slots": n_slots, "n_read": n_read}


def check_coverage(data, n_frames):
    """Light sanity summary printed by the CLI (no per-frame data)."""
    ball_seen = float(np.mean(np.isfinite(data["ball"][:, 0])))
    return {"records": data["n_read"], "expected": n_frames, "ball_seen_frac": ball_seen,
            "slots": data["n_slots"]}
