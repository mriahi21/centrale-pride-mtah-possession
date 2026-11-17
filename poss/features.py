"""Per-half features from the (uncorrected) ATD. Everything is in metres and on the video timeline.

Design notes
- A missing ball is *missing*, never "dead". Missing frames carry confidence 0 downstream.
- Slots are tracklets, not identities: only team membership and positions are used, collectively.
- Each half is processed independently (the caller slices by half).
"""
import warnings

import numpy as np
import pandas as pd

import poss  # noqa: F401
from games import GAMES, GAME

FPS = 25
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]

# Ball tracking cleaning / gap handling
V_BALL_MAX = 45.0          # m/s: faster than any real ball -> detection glitch
MAX_INTERP_GAP = 50        # frames (2 s): bridge only short occlusions
CONF_OBSERVED, CONF_INTERP_SHORT, CONF_INTERP_LONG = 1.0, 0.7, 0.45


def to_metres(players, ball):
    scale = np.array([PITCH_L, PITCH_W], dtype=np.float32)
    return players * scale, ball * scale


def remove_ball_outliers(ball_m, vmax=V_BALL_MAX, fps=FPS, reset_after=12):
    """NaN-out ball detections that imply an impossible speed from the last accepted detection.

    The allowance grows with the gap since the last accepted point. After `reset_after`
    consecutive rejections the tracker is assumed to have genuinely re-acquired elsewhere.
    """
    out = ball_m.copy()
    obs = np.flatnonzero(np.isfinite(out[:, 0]))
    last_i, last_p, rejected = None, None, 0
    for i in obs:
        p = out[i]
        if last_i is None:
            last_i, last_p = i, p
            continue
        gap = i - last_i
        if np.hypot(*(p - last_p)) > vmax / fps * gap + 1.0:
            rejected += 1
            if rejected >= reset_after:
                last_i, last_p, rejected = i, p, 0
            else:
                out[i] = np.nan
        else:
            last_i, last_p, rejected = i, p, 0
    return out


def interpolate_ball(ball_m, max_gap=MAX_INTERP_GAP):
    """Linear interpolation across short gaps. Returns (ball, conf, observed_mask)."""
    n = len(ball_m)
    observed = np.isfinite(ball_m[:, 0])
    ball = ball_m.copy()
    conf = np.where(observed, CONF_OBSERVED, 0.0).astype(np.float32)
    idx = np.flatnonzero(observed)
    if len(idx) < 2:
        return ball, conf, observed
    gaps = np.diff(idx)
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[k], idx[k + 1]
        t = (np.arange(a + 1, b) - a) / float(b - a)
        ball[a + 1:b, 0] = ball[a, 0] + t * (ball[b, 0] - ball[a, 0])
        ball[a + 1:b, 1] = ball[a, 1] + t * (ball[b, 1] - ball[a, 1])
        conf[a + 1:b] = CONF_INTERP_SHORT if (b - a) <= 26 else CONF_INTERP_LONG
    return ball, conf, observed


def ball_speed(ball, half_win=3, fps=FPS):
    """Central-difference speed (m/s) over +/- half_win frames; NaN where either end is missing."""
    n = len(ball)
    v = np.full(n, np.nan, dtype=np.float32)
    if n <= 2 * half_win:
        return v
    d = ball[2 * half_win:] - ball[:-2 * half_win]
    v[half_win:n - half_win] = np.hypot(d[:, 0], d[:, 1]) * fps / (2 * half_win)
    return v


def nearest_team_distances(players_m, ball_m, team_of_slot):
    """(dH, dA) metres from the ball to the nearest tracked player of each team (inf if none)."""
    d = np.hypot(players_m[:, :, 0] - ball_m[:, None, 0], players_m[:, :, 1] - ball_m[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    home = team_of_slot == 0
    return d[:, home].min(axis=1), d[:, ~home].min(axis=1)


def player_activity(pspeed, players_m=None, window=51):
    """Rolling-median of the median tracked-player speed (m/s): a ball-free signal of 'is play live'.

    Uses the ATD speed column when it looks like m/s; otherwise falls back to position differences.
    """
    with np.errstate(all="ignore"):
        ok = np.isfinite(pspeed)
        sp = pspeed
        med_all = np.nanmedian(pspeed) if ok.any() else np.nan
        if (not np.isfinite(med_all) or not (0.2 <= med_all <= 8.0)) and players_m is not None:
            d = players_m[5:] - players_m[:-5]
            sp = np.zeros_like(pspeed)
            sp[:-5] = np.hypot(d[..., 0], d[..., 1]) * FPS / 5.0
            sp[np.isnan(players_m[..., 0])] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame_med = np.nanmedian(sp, axis=1)
    s = pd.Series(frame_med).interpolate(limit=75, limit_direction="both")
    return s.rolling(window, center=True, min_periods=1).median().to_numpy(dtype=np.float32)


def attack_direction(players_m, team_of_slot, kickoff_rel, horizon=250):
    """+1 if a team attacks towards +x in this half, -1 otherwise. Returns (home_dir, away_dir).

    At kickoff each team stands in its own half, so the team with the lower mean x defends the left
    goal and attacks to +x. Falls back to the whole-half mean if the kickoff window has no players.
    """
    def mean_x(sl):
        h = players_m[sl][:, team_of_slot == 0, 0]
        a = players_m[sl][:, team_of_slot == 1, 0]
        if not (np.isfinite(h).any() and np.isfinite(a).any()):
            return None
        return np.nanmean(h), np.nanmean(a)

    res = mean_x(slice(max(kickoff_rel, 0), kickoff_rel + horizon)) or mean_x(slice(None))
    if res is None:
        return 1, -1
    return (1, -1) if res[0] < res[1] else (-1, 1)


def out_of_bounds(ball_m, margin=0.4):
    """True where the (observed) ball is outside the touchlines/goal lines by more than `margin` m."""
    x, y = ball_m[:, 0], ball_m[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > PITCH_L + margin) | (y < -margin) | (y > PITCH_W + margin)


def compute_half_features(players, pspeed, ball, team_of_slot, kickoff_rel=0):
    """All features for one half. Inputs are normalised arrays already sliced to the half."""
    players_m, ball_m = to_metres(players, ball)
    ball_clean = remove_ball_outliers(ball_m)
    ball_i, conf, observed = interpolate_ball(ball_clean)
    dH, dA = nearest_team_distances(players_m, ball_i, team_of_slot)
    return {
        "n": len(ball),
        "ball": ball_i, "ball_conf": conf, "ball_obs": observed,
        "ball_speed": ball_speed(ball_i),
        "dH": dH.astype(np.float32), "dA": dA.astype(np.float32),
        "activity": player_activity(pspeed, players_m),
        "oob": out_of_bounds(ball_clean),
        "attack_dir": attack_direction(players_m, team_of_slot, kickoff_rel),
        "n_players": np.isfinite(players_m[:, :, 0]).sum(axis=1).astype(np.int16),
    }
