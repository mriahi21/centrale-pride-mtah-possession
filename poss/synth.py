"""Synthetic match simulator -- used ONLY to test code wiring. It contains no match data.

It writes files in the same formats as the real bundle (ATD txt/xml, Smart Tagging XML) so the whole
pipeline can be exercised end-to-end, and returns the simulated truth. Scores on synthetic data say
nothing about performance on the real match.
"""
import os

import numpy as np

import poss  # noqa: F401

L, W = 105.3, 68.0
FPS = 25
N_SLOTS = 46
HOME_SLOTS = list(range(0, 11))
AWAY_SLOTS = list(range(23, 34))


def _walk(rng, n, k, sigma, rho=0.995):
    x = np.zeros((n, k, 2))
    eps = rng.normal(0, 1, (n, k, 2)) * sigma
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def sim_block(rng, n, home_attack=1, pre_dead=0):
    """Simulate one half. Returns players [n,22,2] m (0-10 home, 11-21 away), ball [n,2] m, label [n] (0 H,1 A,2 D)."""
    base = np.zeros((22, 2))
    base[:11, 0] = rng.uniform(12, 62, 11)
    base[11:, 0] = rng.uniform(43, 93, 11)
    base[:, 1] = rng.uniform(8, 60, 22)
    if home_attack < 0:
        base[:, 0] = L - base[:, 0]
    label = np.full(n, 2, dtype=np.int8)
    ball = np.zeros((n, 2))
    pos = base[None].repeat(n, 0) + _walk(rng, n, 22, 0.09, 0.992) * 6
    sigma_dead = np.zeros(n, dtype=bool)

    t = pre_dead
    ball[:t] = [L / 2, W / 2]
    team = int(rng.integers(0, 2))
    first = True
    while t < n:
        # alive segment
        dur = int(rng.uniform(6, 22) * FPS)
        end = min(n, t + dur)
        mates = list(range(11)) if team == 0 else list(range(11, 22))
        opps = list(range(11, 22)) if team == 0 else list(range(11))
        carrier = int(rng.choice(mates))
        if t == pre_dead:
            bpos = pos[t, carrier] + np.array([0.0, 0.0])
        else:
            bpos = pos[t, carrier].copy()
        cur = t
        while cur < end:
            hold = int(rng.uniform(1.2, 3.5) * FPS)
            h_end = min(end, cur + hold)
            ball[cur:h_end] = pos[cur:h_end, carrier] + 0.5
            label[cur:h_end] = team
            cur = h_end
            if cur >= end:
                break
            target = int(rng.choice([m for m in mates if m != carrier]))
            tgt = pos[min(n - 1, cur + 20), target]
            start_b = ball[cur - 1].copy()
            fl = max(5, int(np.hypot(*(tgt - start_b)) / 14.0 * FPS))
            f_end = min(end, cur + fl)
            for k in range(cur, f_end):
                a = (k - cur + 1) / float(fl)
                ball[k] = start_b + a * (tgt - start_b)
            label[cur:f_end] = team
            cur = f_end
            carrier = target
        t = end
        if t >= n:
            break
        if rng.random() < 0.55:  # turnover: opponent takes over directly
            team = 1 - team
            # pull an opponent onto the ball: opponent players near the final ball point
            o = int(rng.choice(opps))
            pos[t:t + 60, o] = ball[t - 1] + 0.6
        else:  # stoppage
            d = int(rng.uniform(4, 14) * FPS)
            d_end = min(n, t + d)
            spot = ball[t - 1].copy()
            if rng.random() < 0.5:
                spot[1] = -1.0 if spot[1] < W / 2 else W + 1.0
            ball[t:d_end] = spot
            label[t:d_end] = 2
            sigma_dead[t:d_end] = True
            # slow the players down during the stoppage
            seg = pos[t:d_end]
            pos[t:d_end] = seg[0:1] + (seg - seg[0:1]) * 0.15
            t = d_end
            team = int(rng.integers(0, 2))
            if t < n:
                # taker stands on the ball at the restart
                o = int(rng.choice(list(range(11)) if team == 0 else list(range(11, 22))))
                pos[t:t + 40, o] = ball[t - 1] + 0.7
    label[:pre_dead] = 2
    return pos, ball, label, sigma_dead


def simulate(n_frames=24000, blocks=((1, 2000, 11000), (2, 13000, 22000)), seed=0,
             ball_missing=0.41, second_half_pre_dead=300):
    rng = np.random.default_rng(seed)
    players = np.full((n_frames, N_SLOTS, 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    truth = np.full(n_frames, "X", dtype="<U1")
    restarts = []
    for half, a, b in blocks:
        n = b - a + 1
        pre = 0 if half == 1 else second_half_pre_dead
        pos, bl, lab, _ = sim_block(rng, n, 1 if half == 1 else -1, pre)
        players[a:b + 1, HOME_SLOTS] = pos[:, :11]
        players[a:b + 1, AWAY_SLOTS] = pos[:, 11:]
        ball[a:b + 1] = bl
        truth[a:b + 1] = np.array(["H", "A", "D"])[lab]
        # restarts = every D->alive transition
        d = np.flatnonzero((lab[1:] != 2) & (lab[:-1] == 2)) + 1
        restarts += [(a + int(i), int(lab[i])) for i in d]
    # tracker noise
    players += rng.normal(0, 0.12, players.shape)
    ball += rng.normal(0, 0.1, ball.shape)
    ball_clean = ball.copy()
    # missing ball (bursts) and missing players
    miss = np.zeros(n_frames, dtype=bool)
    while miss.mean() < ball_missing:
        s = int(rng.integers(0, n_frames))
        miss[s:s + int(rng.integers(10, 150))] = True
    ball[miss] = np.nan
    for s in HOME_SLOTS + AWAY_SLOTS:
        for _ in range(20):
            a0 = int(rng.integers(0, n_frames))
            players[a0:a0 + int(rng.integers(5, 100)), s] = np.nan
    # speeds from positions (m/s)
    sp = np.full((n_frames, N_SLOTS), np.nan)
    d = players[2:] - players[:-2]
    sp[1:-1] = np.hypot(d[..., 0], d[..., 1]) * FPS / 2.0
    return {"players": players, "ball": ball, "ball_clean": ball_clean, "speed": sp, "truth": truth,
            "restarts": restarts, "blocks": list(blocks), "n_frames": n_frames}


def write_bundle(sim, data_dir, with_tags=True):
    """Write tactical/BAR-ATH_tactical_{atd.txt,atd.xml,smart_tagging.xml} under data_dir."""
    from games import asset_paths
    folder = os.path.join(data_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    paths = asset_paths(data_dir=data_dir)
    n = sim["n_frames"]
    P, B, S = sim["players"], sim["ball"], sim["speed"]
    with open(paths["atd_raw"], "w") as fh:
        for f in range(n):
            groups = []
            for s in range(N_SLOTS):
                if np.isnan(P[f, s, 0]):
                    groups.append("NaN,NaN,NaN")
                else:
                    sp = 0.0 if np.isnan(S[f, s]) else S[f, s]
                    groups.append("{:.5f},{:.5f},{:.3f}".format(P[f, s, 0] / L, P[f, s, 1] / W, sp))
            ball = "NaN,NaN,NaN" if np.isnan(B[f, 0]) else "{:.5f},{:.5f},0.0".format(B[f, 0] / L, B[f, 1] / W)
            fh.write("{}:{}:{}\n".format(f + 1, ";".join(groups), ball))
    players_xml = "".join('<Player id="P{0}" teamId="{1}"/>'.format(i, "HOME" if i < 23 else "AWAY")
                          for i in range(N_SLOTS))
    with open(paths["atd_meta"], "w") as fh:
        fh.write('<?xml version="1.0"?><Metadata><Teams><Team id="HOME"><Name>Barcelona</Name></Team>'
                 '<Team id="AWAY"><Name>Athletic Club</Name></Team></Teams><Players>{}</Players>'
                 "</Metadata>".format(players_xml))
    if with_tags:
        inst = []

        def add(code, a, b, team, half, typ=None):
            lab = '<label><group>Team</group><text>{}</text></label>'.format(team)
            lab += '<label><group>Half</group><text>{}</text></label>'.format(half)
            if typ:
                lab += '<label><group>Type</group><text>{}</text></label>'.format(typ)
            inst.append("<instance><code>{}</code><start>{:.2f}</start><end>{:.2f}</end>{}</instance>"
                        .format(code, a / FPS, b / FPS, lab))

        for fr, team_code in sim["restarts"]:
            half = "1st Half" if fr < sim["blocks"][0][2] + 1 else "2nd Half"
            name = "Barcelona" if team_code == 0 else "Atletic Club"
            add("SET PIECES", fr - 50, fr + 75, name, half, "Corner Kick")
        with open(paths["smart_tagging"], "w") as fh:
            fh.write('<?xml version="1.0"?><file><ALL_INSTANCES>{}</ALL_INSTANCES></file>'.format("".join(inst)))
    return paths
