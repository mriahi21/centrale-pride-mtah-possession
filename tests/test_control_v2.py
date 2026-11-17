import numpy as np

import poss  # noqa: F401
from poss import control, model


def _ft(n, dH, dA, v=3.0, conf=1.0):
    ball = np.tile(np.array([[50.0, 30.0]]), (n, 1)) + np.arange(n)[:, None] * 0.05
    return {"n": n, "ball": ball, "ball_conf": np.full(n, conf, dtype=np.float32),
            "ball_speed": np.full(n, v, dtype=np.float32), "dH": np.asarray(dH, dtype=np.float32),
            "dA": np.asarray(dA, dtype=np.float32)}


def test_interpolation_gating_long_or_implausible_gaps_are_neutral():
    n = 300
    ft = _ft(n, np.full(n, 1.0), np.full(n, 9.0))
    ft["ball_conf"][100:140] = 0.7            # 40-frame "interpolated" run: too long for ownership
    ft["ball_conf"][200:210] = 0.7            # short but implausible endpoint jump
    ft["ball"][210:] += [60.0, 0.0]
    w = control.frame_weights(ft)
    assert (w[100:140] == 0).all() and (w[200:210] == 0).all()
    ft2 = _ft(n, np.full(n, 1.0), np.full(n, 9.0))
    ft2["ball_conf"][100:110] = 0.7           # short plausible gap
    w2 = control.frame_weights(ft2)
    assert (w2[100:110] == control.W_INTERP).all() and w2[50] == 1.0
    assert w2[98] < 1.0                       # uncertainty around the occlusion edge


def test_isolated_touch_does_not_flip_but_sustained_contact_does():
    n = 600
    dH, dA = np.full(n, 1.0), np.full(n, 9.0)      # home holds the ball
    dH[300:306], dA[300:306] = 9.0, 0.8            # 6-frame opposing touch
    dH[450:], dA[450:] = 9.0, 0.8                  # real turnover at 450
    ft = _ft(n, dH, dA)
    u, aux = control.control_evidence(ft)
    codes = model.decode_alive(u, np.ones(n, dtype=bool), lam=control.dwell_lambda([u]))
    assert (codes[290:330] == model.H).all()       # touch ignored
    assert (codes[470:] == model.A).all()          # turnover recognised
    assert 440 <= int(np.flatnonzero(codes == model.A)[0]) <= 462   # prompt boundary


def test_pass_in_flight_past_a_defender_is_not_a_turnover():
    n = 400
    dH, dA = np.full(n, 1.0), np.full(n, 9.0)
    dH[150:200], dA[150:200] = 8.0, 1.0            # ball flying fast past an away player
    ft = _ft(n, dH, dA, v=14.0)
    ft["ball_speed"][:150] = 3.0
    ft["ball_speed"][200:] = 3.0
    u, _ = control.control_evidence(ft)
    codes = model.decode_alive(u, np.ones(n, dtype=bool), lam=control.dwell_lambda([u]))
    assert (codes == model.H).all()


def test_lambda_is_independent_of_dead_mask_and_bounded():
    rng = np.random.default_rng(0)
    u = [rng.normal(0, 0.6, 5000).astype(np.float32)]
    lam = control.dwell_lambda(u)
    assert control.LAMBDA_MIN <= lam <= control.LAMBDA_MAX
    assert control.dwell_lambda(u) == lam          # depends on evidence only, never on a possession count


def test_restart_prior_from_first_sustained_contact():
    n = 300
    aux = {"sust_h": np.zeros(n, dtype=bool), "sust_a": np.zeros(n, dtype=bool)}
    aux["sust_a"][120:160] = True
    assert control.restart_priors(aux, [(100, 300)]) == [-control.K_RESTART]
    assert control.restart_priors(aux, [(200, 300)]) == [0.0]
