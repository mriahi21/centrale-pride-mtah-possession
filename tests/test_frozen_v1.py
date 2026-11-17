"""Regression guard: the frozen v1 pipeline (tag `frozen-v1`, commit e471aec) must keep producing the
same labels on a fixed synthetic bundle. New work goes behind other profiles; v1 must never drift."""
import hashlib

import poss  # noqa: F401
from poss import pipeline, synth, tags as tg

GOLDEN_SHA256 = "a430fac52e28d913ca4657c1772305a9c9d98b7eca50e5fca368ea19dbbd77de"


def test_v1_profile_unchanged_on_fixed_synthetic(tmp_path):
    n, blocks = 24000, ((1, 1000, 11999), (2, 13000, 23499))
    sim = synth.simulate(n_frames=n, blocks=blocks, seed=11, second_half_pre_dead=0)
    p = synth.write_bundle(sim, str(tmp_path))
    t = tg.load_tags(str(tmp_path), log=lambda *a: None)
    kw = {} if not hasattr(pipeline, "PROFILES") else {"profile": "v1"}
    c, _ = pipeline.generate_labels(p["atd_raw"], p["atd_meta"], n_frames=n, blocks=list(blocks),
                                    kickoffs={1: 1000, 2: 13000}, tags=t, log=lambda *a: None, **kw)
    assert hashlib.sha256(c.tobytes()).hexdigest() == GOLDEN_SHA256
