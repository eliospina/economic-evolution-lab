"""
Correctness tests for the RBC model and the perturbation engine.

The anchor is an *exact* check: with full depreciation (delta=1), inelastic
labour and log utility, the RBC collapses to the Brock-Mirman model whose
solution is exactly log-linear,

    K_{t+1} = alpha * beta * Y_t  =>  k_hat_{t+1} = alpha*k_hat_t + z_hat_t.

A first-order solver must reproduce  dP[k,k]=alpha  and  dP[k,z]=1  to machine
precision. If the QZ sorting or the Klein recovery were wrong, this fails.
"""

import numpy as np

from eel.models.macro import rbc
from eel.solve import perturbation


def test_brock_mirman_exact():
    """delta=1, fixed labour -> exact closed form. The hard correctness check."""
    p = rbc.RBCParams(delta=1.0, rho=0.9)
    sol, ss, _ = rbc.build(p, endogenous_labour=False)
    assert sol.blanchard_kahn
    ki, zi = sol.state_names.index("k"), sol.state_names.index("z")
    # k_{t+1} response: alpha to capital, 1 to productivity.
    assert abs(sol.P[ki, ki] - p.alpha) < 1e-8
    assert abs(sol.P[ki, zi] - 1.0) < 1e-8
    # productivity row is just the AR(1).
    assert abs(sol.P[zi, zi] - p.rho) < 1e-8
    assert abs(sol.P[zi, ki]) < 1e-10


def test_blanchard_kahn_baseline():
    """Baseline endogenous-labour model has a unique stable solution."""
    sol, _, _ = rbc.build(rbc.RBCParams())
    assert sol.n_stable == len(sol.state_names) == 2
    assert sol.blanchard_kahn


def test_steady_state_identities():
    """The analytic steady state satisfies the model's own equations."""
    p = rbc.RBCParams()
    ss, psi = rbc.steady_state(p, endogenous_labour=True)
    a, b, d = p.alpha, p.beta, p.delta
    Y, C, I, K, N, Z = ss["y"], ss["c"], ss["i"], ss["k"], ss["n"], ss["z"]
    assert abs(Y - (C + I)) < 1e-10                       # resource
    assert abs(I - d * K) < 1e-10                         # capital law at SS
    assert abs(a * Y / K - (1 / b - 1 + d)) < 1e-10       # Euler at SS
    assert abs(Y - Z * K**a * N**(1 - a)) < 1e-10         # production
    assert abs(psi * C / (1 - N) - (1 - a) * Y / N) < 1e-10  # labour FOC
    assert abs(N - p.target_n) < 1e-10                    # psi hit the target


def test_irf_signs():
    """A positive productivity shock raises Y, C, I and hours on impact."""
    sol, _, _ = rbc.build(rbc.RBCParams())
    irf = perturbation.impulse_response(sol, horizon=20, size=1.0)
    for name in ["y", "c", "i", "n"]:
        assert irf[name][0] > 0, f"{name} should rise on impact"
    # Capital is predetermined: it cannot jump at t=0, builds up after.
    assert abs(irf["k"][0]) < 1e-12
    assert irf["k"][1] > 0


def test_business_cycle_moments():
    """Classic RBC second moments: sd_C < sd_Y < sd_I, all procyclical."""
    sol, _, _ = rbc.build(rbc.RBCParams())
    m = perturbation.moments(sol)
    sd = m["std"]
    assert sd["c"] < sd["y"] < sd["i"], "investment most volatile, consumption least"
    for name in ["c", "i", "n"]:
        assert m["corr"][(name, "y")] > 0.5, f"{name} should be procyclical"
    assert m["autocorr"]["y"] > 0.6, "output is persistent"
