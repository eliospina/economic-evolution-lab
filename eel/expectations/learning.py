"""
Adaptive learning (Evans-Honkapohja) for the NK model — Phase 2.

The Perceived Law of Motion (PLM) has the minimal-state-variable form: agents
believe the endogenous variables are linear in the exogenous states s = [g,u,v],

    [x_t; pi_t] = a + b s_t      (beliefs phi = (a, b), a 4x2 matrix [const; b]).

Knowing the shock processes (E_t s_{t+1} = R s_t), expectations are

    E_t[x_{t+1}, pi_{t+1}] = phi' [1, R s_t].

Substituting these into the structural block (via nk_dsge.tmap_functions) gives the
Actual Law of Motion. Beliefs update by recursive least squares; the gain may be
decreasing (1/t, RLS — converges a.s. to the RE fixed point under E-stability) or
constant (tracks structural change — the empirically relevant variant).

INFORMATION SYMMETRY (mandatory design rule)
---------------------------------------------
`Learner` receives only a regressor vector. It has NO channel to the "true"
shocks. In the empirical exercise the driver feeds it shocks recovered from
observables (s_t = F^{-1} obs_t) — the SAME inversion the rational-expectations
forecaster uses. Same information set, different expectation mechanism. The
synthetic `simulate_learning` below is the ONE place true shocks are used, and
only because there RE is not a competitor — it is a self-referential DGP used to
validate convergence (E-stability), the Phase-2 analogue of the Brock-Mirman check.
"""

import numpy as np

from eel.models.macro import nk_dsge
from eel.models.macro.nk_dsge import NKParams


class Learner:
    """Constant- or decreasing-gain recursive least squares over [1, s].

    Beliefs `Phi` are 4x2: rows [const, g, u, v], columns [x, pi]. The learner is
    deliberately agnostic about what its regressors mean — that is what makes the
    information set, not the mechanism, the only thing the empirical test varies.
    """

    def __init__(self, Phi0=None, R0=None):
        self.Phi = np.zeros((4, 2)) if Phi0 is None else np.array(Phi0, float)
        self.Rmat = np.eye(4) if R0 is None else np.array(R0, float)

    def forecast(self, s, R):
        """Shock-PLM forecast: E_t[x_{t+1}, pi_{t+1}] = Phi' [1, R s]. (E-stability sim.)"""
        z1 = np.concatenate([[1.0], R @ s])
        ey = self.Phi.T @ z1
        return ey[0], ey[1]

    def predict(self, w):
        """VAR-PLM forecast: Phi' [1, w], where w are observable regressors.

        Used by the empirical exercise: the PLM is a VAR in observables, so the
        one-step forecast maps current observables directly to next-period
        targets — no shock recovery, no R transform.
        """
        return self.Phi.T @ np.concatenate([[1.0], w])

    def update(self, s, y, gain):
        """One RLS step with step size `gain`, regressor z=[1,s], target y=[x,pi]."""
        z = np.concatenate([[1.0], s])
        self.Rmat = self.Rmat + gain * (np.outer(z, z) - self.Rmat)
        err = y - self.Phi.T @ z
        self.Phi = self.Phi + gain * np.linalg.solve(self.Rmat, np.outer(z, err))


def re_beliefs(p: NKParams = None):
    """RE fixed-point beliefs: b = F rows for (x, pi); constants = 0."""
    F, _ = nk_dsge.re_reduced_form(p)
    Phi = np.zeros((4, 2))
    Phi[1:, 0] = F[0, :]    # x  on (g, u, v)
    Phi[1:, 1] = F[1, :]    # pi on (g, u, v)
    return Phi


def simulate_learning(p: NKParams = None, T=20000, gain="rls", seed=0,
                      Phi0=None, t0=100):
    """Self-referential simulation under learning (synthetic; uses true shocks).

    Returns the belief path (T, 4, 2). With gain='rls' and the Taylor principle,
    beliefs converge to `re_beliefs(p)` — the E-stability validation.

    The RLS step size is 1/(t + t0) with t0 > 1, NOT 1/(t+1): a gain of 1 at the
    first step would collapse the moment matrix R to a rank-1 (singular) outer
    product. t0 keeps R positive-definite while preserving the asymptotic ~1/t
    schedule that delivers convergence.
    """
    p = p or NKParams()
    R = nk_dsge.shock_persistence(p)
    tmap = nk_dsge.tmap_functions(p)
    rhos = np.array([p.rho_g, p.rho_u, p.rho_v])
    sigmas = np.array([p.sigma_g, p.sigma_u, p.sigma_v])

    # Initialise the moment matrix to the regressors' true second moments
    # diag(1, var_g, var_u, var_v). With identity instead, the wildly different
    # scales (constant O(1) vs shocks O(1e-2)) leave the slopes under-updated
    # until the 1/t gain has already shrunk — and RLS never catches up.
    var = sigmas**2 / (1.0 - rhos**2)
    R0 = np.diag(np.concatenate([[1.0], var]))
    learner = Learner(Phi0=Phi0, R0=R0)
    rng = np.random.default_rng(seed)
    s = np.zeros(3)
    path = np.empty((T, 4, 2))
    for t in range(T):
        s = rhos * s + rng.normal(0.0, 1.0, 3) * sigmas      # true shocks (DGP)
        ex, epi = learner.forecast(s, R)                      # beliefs phi_{t-1}
        x, pi, _ = tmap(ex, epi, s[0], s[1], s[2])            # realised via ALM
        gt = 1.0 / (t + t0) if gain == "rls" else float(gain)
        learner.update(s, np.array([x, pi]), gt)
        path[t] = learner.Phi
    return path
