"""
Tests for the Clark-West implementation: size control and power, on constructed
cases with known answers.
"""

import numpy as np

from eel.evaluation import clark_west as cw


def test_size_control_under_the_null():
    """Monte Carlo size check: when the larger model adds only noise (the null),
    the one-sided 5% rejection rate must be near nominal, not inflated.

    A single draw is too noisy to judge size (5% reject by chance); we check the
    rejection RATE over many replications instead."""
    reps, n, rejections = 500, 200, 0
    for r in range(reps):
        rng = np.random.default_rng(r)
        y = rng.normal(size=n)
        f1 = np.zeros(n)
        f2 = f1 + rng.normal(scale=0.5, size=n)  # pure noise, no signal
        if cw.clark_west(y, f1, f2)["pvalue_one_sided"] < 0.05:
            rejections += 1
    rate = rejections / reps
    assert rate < 0.10, f"size inflated: rejection rate {rate:.3f} >> 0.05"


def test_genuine_signal_is_significant():
    """f2 captures real signal that f1 (restricted) misses: CW rejects."""
    rng = np.random.default_rng(1)
    n = 400
    signal = rng.normal(size=n)
    y = signal + rng.normal(scale=0.3, size=n)
    f1 = np.zeros(n)                              # restricted misses the signal
    f2 = signal                                  # larger captures it
    res = cw.clark_west(y, f1, f2)
    assert res["pvalue_one_sided"] < 0.01
    assert res["favored"] == "learning"
    assert res["rmse_learning"] < res["rmse_re"]


def test_power_warning_for_short_holdout():
    assert cw.power_warning(48) is not None
    assert cw.power_warning(80) is None


def test_hac_se_reduces_to_iid_at_lag0():
    z = np.random.default_rng(2).normal(size=100)
    se0 = cw._hac_se(z, 0)
    assert abs(se0 - np.std(z) / np.sqrt(len(z))) < 1e-12
