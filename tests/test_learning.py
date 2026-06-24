"""
Validation of the adaptive-learning implementation.

The anchor (Phase-2 analogue of the RBC's Brock-Mirman check): under the Taylor
principle the NK MSV solution is E-stable, so decreasing-gain (RLS) learning must
converge to the rational-expectations beliefs. If the T-map or the update were
wrong, beliefs would not converge to F.
"""

import numpy as np

from eel.expectations import learning
from eel.expectations.learning import re_beliefs
from eel.models.macro.nk_dsge import NKParams


def test_estability_convergence_to_re():
    """RLS learning converges to the RE fixed point (E-stability).

    The defining property is convergence, so we check the distance to RE SHRINKS
    along the horizon (the ~1/sqrt(t) signature) and is small late on — not an
    arbitrary precision at a single T, which 1/t learning reaches only slowly.
    """
    p = NKParams()
    phi_re = re_beliefs(p)
    path = learning.simulate_learning(p, T=60000, gain="rls", seed=1)

    def err(a, b):
        return np.max(np.abs(path[a:b].mean(axis=0) - phi_re))

    e_early, e_mid, e_late = err(8000, 12000), err(28000, 32000), err(56000, 60000)
    assert e_late < e_mid < e_early, \
        f"not converging: {e_early:.3f} -> {e_mid:.3f} -> {e_late:.3f}"
    assert e_late < 0.08, f"late beliefs not close to RE: {e_late:.3f}"


def test_constant_gain_stays_near_re():
    """Constant-gain learning fluctuates around RE (does not diverge)."""
    p = NKParams()
    path = learning.simulate_learning(p, T=10000, gain=0.04, seed=2)
    phi_mean = path[2000:].mean(axis=0)         # after burn-in
    phi_re = re_beliefs(p)
    assert np.max(np.abs(phi_mean - phi_re)) < 0.10
