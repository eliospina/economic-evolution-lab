"""
Correctness tests for the New Keynesian three-equation DSGE.

Two kinds of checks:
  1. Determinacy — Blanchard-Kahn holds under the Taylor principle (phi_pi > 1)
     and FAILS when it is violated (phi_pi < 1). That the model reproduces the
     Taylor principle is meaningful economic validation, not just "it ran".
  2. Qualitative impulse responses match a textbook NK model.
"""

import numpy as np

from eel.models.macro import nk_dsge
from eel.models.macro.nk_dsge import NKParams, RationalExpectations
from eel.solve import perturbation


def test_blanchard_kahn_determinacy():
    """Taylor principle satisfied -> unique stable solution (3 stable roots)."""
    sol, _ = nk_dsge.build(NKParams())
    assert sol.n_stable == len(sol.state_names) == 3
    assert sol.blanchard_kahn


def test_taylor_principle_violation_breaks_determinacy():
    """phi_pi < 1 -> indeterminacy (too many stable roots), so BK fails."""
    sol, _ = nk_dsge.build(NKParams(phi_pi=0.9))
    assert not sol.blanchard_kahn
    assert sol.n_stable > len(sol.state_names)   # indeterminacy, not no-solution


def test_kappa_value():
    """NKPC slope matches the standard composite at the baseline calibration."""
    p = NKParams()
    assert abs(p.kappa - 0.1717) < 1e-3


def test_irf_monetary_shock():
    """Contractionary monetary shock (+v): output gap and inflation both fall."""
    sol, _ = nk_dsge.build(NKParams())
    irf = perturbation.impulse_response(sol, horizon=20, shock_col=2)  # v
    assert irf["x"][0] < 0,  "output gap should fall after +v"
    assert irf["pi"][0] < 0, "inflation should fall after +v"


def test_irf_cost_push_shock():
    """Cost-push (+u): inflation up, output gap down — the policy trade-off."""
    sol, _ = nk_dsge.build(NKParams())
    irf = perturbation.impulse_response(sol, horizon=20, shock_col=1)  # u
    assert irf["pi"][0] > 0, "inflation should rise after +u"
    assert irf["x"][0] < 0,  "output gap should fall after +u"
    assert irf["i"][0] > 0,  "policy rate should rise (phi_pi > 1)"


def test_irf_demand_shock():
    """Demand (+g): output gap, inflation and the policy rate all rise."""
    sol, _ = nk_dsge.build(NKParams())
    irf = perturbation.impulse_response(sol, horizon=20, shock_col=0)  # g
    for var in ["x", "pi", "i"]:
        assert irf[var][0] > 0, f"{var} should rise after +g"


def test_expectation_module_is_swappable():
    """Passing the RE module explicitly reproduces the default — the seam works."""
    sol_default, _ = nk_dsge.build(NKParams())
    sol_explicit, _ = nk_dsge.build(NKParams(), expectations=RationalExpectations())
    assert np.allclose(sol_default.P, sol_explicit.P)
    assert np.allclose(sol_default.F, sol_explicit.F)
