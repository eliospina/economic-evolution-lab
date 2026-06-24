"""
Tests for the Phase-2b adaptation machinery, including the size-control finding
that motivates Monte-Carlo calibration of the confirmatory Clark-West.
"""

import numpy as np

from eel.evaluation import adaptation as ad


def _dgp():
    """A stable fixed-coefficient VAR(1) null (no adaptation)."""
    A = np.array([[0.90, 0.00, 0.00],
                  [0.05, 0.80, 0.00],
                  [0.00, 0.10, 0.85]])
    return {"c": np.zeros(3), "A": A, "chol": 0.1 * np.eye(3)}


def test_regressors_lag_structure():
    obs = np.arange(15).reshape(5, 3).astype(float)
    Z1 = ad.regressors(obs, 1)
    assert np.allclose(Z1[2], [1, 6, 7, 8])           # [1, obs_2]
    Z2 = ad.regressors(obs, 2)
    assert np.allclose(Z2[2], [1, 6, 7, 8, 3, 4, 5])  # [1, obs_2, obs_1]
    assert np.isnan(Z2[0]).any()                      # not enough history


def test_fixed_gain_is_frozen():
    """gain=0 -> beliefs never move: forecasts equal the training-OLS applied."""
    rng = np.random.default_rng(0)
    obs = ad.simulate_var1(_dgp(), 120, rng)
    train_end = 80
    f = ad.var_forecast(obs, train_end, 0.0, lags=1)
    # independent training OLS, frozen
    Z = ad.regressors(obs, 1)
    tr = np.arange(0, train_end - 1)
    B, *_ = np.linalg.lstsq(Z[tr], obs[tr + 1, :2], rcond=None)
    for t in [train_end, 100, 118]:
        assert np.allclose(f[t], B.T @ Z[t], atol=1e-10)


def test_constant_gain_differs_from_fixed():
    rng = np.random.default_rng(1)
    obs = ad.simulate_var1(_dgp(), 120, rng)
    f0 = ad.var_forecast(obs, 80, 0.0, 1)
    fc = ad.var_forecast(obs, 80, 0.05, 1)
    o = np.arange(80, 119)
    assert np.max(np.abs(f0[o] - fc[o])) > 1e-3


def test_confirmatory_cw_is_oversized_under_no_adaptation():
    """THE size-control finding: the nested CW for fixed-vs-constant-gain
    over-rejects massively under a fixed-coefficient (no-adaptation) null, which
    is why the experiment Monte-Carlo-calibrates instead of trusting asymptotics."""
    res = ad.size_control(_dgp(), reps=200, n_train=90, n_holdout=120,
                          gain=0.04, lags=1, target=1, seed=0)
    assert res["rejection_rate"] > 0.20, \
        f"expected gross oversizing, got {res['rejection_rate']:.3f}"


def test_mc_pvalue_is_calibrated():
    """MC p-value of a draw FROM the null is ~ Uniform: rarely below 0.05."""
    null = np.random.default_rng(0).normal(size=2000)
    ps = [ad.mc_pvalue(null[i], np.delete(null, i)) for i in range(200)]
    assert 0.0 <= min(ps) and max(ps) <= 1.0
    assert np.mean(np.array(ps) < 0.05) < 0.12        # ~ nominal


def test_rolling_forecasts_shapes_and_causality():
    rng = np.random.default_rng(2)
    obs = ad.simulate_var1(_dgp(), 160, rng)
    f_ols, f_cg = ad.rolling_forecasts(obs, window=60, gain=0.04, lags=1)
    assert f_ols.shape == (159, 2)
    valid = ~np.isnan(f_ols[:, 1])
    assert valid.sum() > 50
    # early origins (before a full window) are NaN
    assert np.isnan(f_ols[10, 1])
