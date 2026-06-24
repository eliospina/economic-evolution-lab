"""
MANDATORY: information-set symmetry between the RE and learning forecasters.

Both models forecast from the same observables {x, pi, i} up to t. RE recovers the
latent state via s_t = F^{-1} obs_t; learning uses a VAR PLM in observables and
needs no shocks at all. The learning model must NOT get privileged access to the
true shocks — that would rig the Clark-West comparison in its favour.

These tests pin that down structurally:
  * RE's inversion is correct and refuses a near-singular F (Note 1);
  * neither model looks ahead — forecasts at origin t depend only on obs up to t;
  * learning depends on observables alone (no F / R / shock channel);
  * the corrected design is NON-degenerate (learning != RE), guarding against a
    regression to the F^{-1}-recovered-shock PLM.
"""

import numpy as np
import pytest

from eel.models.macro import nk_dsge
from eel.models.macro.nk_dsge import NKParams
from eel.evaluation import nk_forecast as nf


def _obs(seed=0, n=120):
    return np.random.default_rng(seed).normal(size=(n, 3)) * 0.5


def test_F_is_invertible():
    """Note 1: the reduced-form map must be invertible for the exercise to be valid."""
    F, _ = nk_dsge.re_reduced_form(NKParams())
    assert np.isfinite(np.linalg.cond(F))
    assert np.linalg.cond(F) < 1e6


def test_inversion_round_trips_true_shocks():
    """obs = F s  =>  recover_shocks(obs, F) == s. RE's state recovery is correct."""
    F, _ = nk_dsge.re_reduced_form(NKParams())
    s_true = np.random.default_rng(0).normal(size=(200, 3))
    s_rec = nf.recover_shocks(s_true @ F.T, F)
    assert np.allclose(s_rec, s_true, atol=1e-10)


def test_recover_shocks_rejects_singular_F():
    """Note 1: a near-singular reduced form is refused, not silently inverted."""
    F_bad = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 1.0, 0.0]])         # rank 2, singular
    with pytest.raises(ValueError):
        nf.recover_shocks(np.ones((10, 3)), F_bad)


def test_neither_model_looks_ahead():
    """Forecasts at origin t use only observables up to t: perturbing obs strictly
    after t leaves f[t] unchanged for BOTH models (the shared information set)."""
    F, R = nk_dsge.re_reduced_form(NKParams())
    obs = _obs(5)
    f_re1, f_al1 = nf.forecast_rational(obs, F, R), nf.forecast_learning(obs, 0.04, 60)
    obs2 = obs.copy()
    obs2[100:] += 1.0                            # change only the future
    f_re2, f_al2 = nf.forecast_rational(obs2, F, R), nf.forecast_learning(obs2, 0.04, 60)
    assert np.allclose(f_re1[:100], f_re2[:100])
    assert np.allclose(f_al1[:100], f_al2[:100])


def test_learning_uses_only_observables():
    """forecast_learning takes ONLY obs (no F/R/shock arg): deterministic in obs."""
    obs = _obs(7)
    assert np.allclose(nf.forecast_learning(obs, 0.04, 60),
                       nf.forecast_learning(obs, 0.04, 60))


def test_design_is_non_degenerate():
    """Regression guard: learning forecasts genuinely differ from RE (the bug the
    observable-VAR PLM fixes — the recovered-shock PLM made them identical)."""
    F, R = nk_dsge.re_reduced_form(NKParams())
    obs = _obs(9)
    f_re = nf.forecast_rational(obs, F, R)
    f_al = nf.forecast_learning(obs, 0.04, 60)
    assert np.max(np.abs(f_re - f_al)) > 1e-3
