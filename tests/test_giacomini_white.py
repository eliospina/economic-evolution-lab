"""Tests for the Giacomini-White predictive-ability test."""

import numpy as np

from eel.evaluation import giacomini_white as gw


def test_no_difference_is_well_sized():
    """Two equally-good forecasters: the unconditional GW rejects ~ nominally."""
    reps, n, rej = 400, 150, 0
    for r in range(reps):
        rng = np.random.default_rng(r)
        y = rng.normal(size=n)
        f1 = y + rng.normal(scale=1.0, size=n)
        f2 = y + rng.normal(scale=1.0, size=n)        # same expected accuracy
        if gw.giacomini_white(y, f1, f2, conditional=False)["uncond_p_two_sided"] < 0.05:
            rej += 1
    assert rej / reps < 0.10, f"GW size inflated: {rej/reps:.3f}"


def test_detects_a_better_forecaster():
    """f2 is genuinely closer to y -> GW favours the competitor, one-sided p small."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(size=n)
    f1 = y + rng.normal(scale=1.0, size=n)
    f2 = y + rng.normal(scale=0.3, size=n)            # much better
    res = gw.giacomini_white(y, f1, f2, conditional=True)
    assert res["favored"] == "competitor"
    assert res["uncond_p_one_sided"] < 0.01
