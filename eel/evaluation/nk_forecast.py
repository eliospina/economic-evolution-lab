"""
One-step-ahead forecasting of the NK observables under RE vs adaptive learning,
on a SHARED information set.

The mandatory design rule (information symmetry): both models see only the
observables {x, pi, i} up to t. Only the expectation mechanism differs:

    RE       : E_t obs_{t+1} = F R F^{-1} obs_t          (theory-restricted)
               recovers the latent state via the inversion s_t = F^{-1} obs_t.
    learning : a VAR(1) PLM in observables — E_t[x,pi]_{t+1} = Phi_{t-1}' [1, obs_t],
               with Phi updated by constant-gain RLS. This is the standard
               empirical adaptive-learning specification (Milani 2007;
               Slobodyan-Wouters 2012): the agents' subjective forecast.

Why a VAR PLM and NOT a PLM over the F^{-1}-recovered shocks: with s_t = F^{-1}
obs_t the relation obs_t = F s_t is an IDENTITY, so a PLM regressing observables
on the recovered shocks fits perfectly, beliefs stay frozen at RE, and learning
becomes identical to RE by construction (a degenerate comparison). The VAR-in-
observables PLM has genuine residuals, so beliefs move and the comparison is real.

`forecast_learning` takes ONLY observables — no F, no R, no shocks. It cannot use
privileged information; the asymmetry the design rule guards against is impossible
by construction. RE is nested (the VAR with coefficients fixed at F R F^{-1}), so
Clark-West is exactly valid.

Forecasts are indexed by origin t: f[t] predicts obs[t+1], for t = 0 .. T-2.
"""

import numpy as np

from eel.models.macro import nk_dsge
from eel.models.macro.nk_dsge import NKParams
from eel.expectations.learning import Learner


def recover_shocks(obs, F, cond_tol=1e8):
    """s_t = F^{-1} obs_t for every row — RE's state recovery.

    Note 1 (documented): the exercise is only valid if F is invertible (the shock
    representation is fundamental). We check the condition number and refuse a
    near-singular map rather than return garbage shocks.
    """
    F = np.asarray(F, float)
    cond = np.linalg.cond(F)
    if not np.isfinite(cond) or cond > cond_tol:
        raise ValueError(
            f"reduced-form map F is near-singular (cond={cond:.2e}); shock "
            f"recovery is not fundamental and the empirical test is invalid.")
    Finv = np.linalg.inv(F)
    return np.asarray(obs, float) @ Finv.T


def _ar1(z):
    """OLS AR(1) coefficient through the origin (series are mean-zero by design)."""
    z0, z1 = z[:-1], z[1:]
    return float(z1 @ z0 / (z0 @ z0))


def estimate_shock_persistence(obs_train, p: NKParams = None, iters=8):
    """Estimate the shock persistences rho on the TRAINING window only.

    Without this, RE forecasts the highly persistent data with the calibrated
    rho=0.5 and mean-reverts far too fast — a straw-man baseline. Recovered shocks
    depend on F, which depends on rho, so we iterate to a fixed point. Structural
    parameters stay frozen (only rho is estimated), so the comparison still
    isolates the expectation mechanism. rho is clipped below 1 for stationarity.
    """
    from dataclasses import replace
    p = p or NKParams()
    for _ in range(iters):
        F, _ = nk_dsge.re_reduced_form(p)
        s = recover_shocks(obs_train, F)
        rg, ru, rv = (min(max(_ar1(s[:, k]), 0.0), 0.999) for k in range(3))
        p = replace(p, rho_g=rg, rho_u=ru, rho_v=rv)
    return p


def forecast_rational(obs, F, R):
    """RE one-step forecasts of (x, pi). f[t] = (F R F^{-1} obs_t)[:2]."""
    s = recover_shocks(obs, F)
    pred_obs = (s @ R.T) @ F.T                  # E_t obs_{t+1} = F R F^{-1} obs_t
    return pred_obs[:-1, :2]


def forecast_learning(obs, gain, train_end, Phi0=None, R0=None):
    """Adaptive-learning one-step forecasts of (x, pi) from observables ONLY.

    PLM: [x_{t+1}, pi_{t+1}] = Phi' [1, x_t, pi_t, i_t]. Beliefs are initialised by
    OLS on the training window and updated by constant-gain RLS walking forward.

    Takes only `obs` — no F, no R, no shocks. That is the structural guarantee of
    information symmetry: there is no channel through which learning could see the
    true shocks or anything RE doesn't also see.
    """
    obs = np.asarray(obs, float)
    T = len(obs)
    y = obs[:, :2]                              # targets x, pi
    # regressor at origin t is obs_t; Learner prepends the constant.
    if Phi0 is None:                            # OLS init on training pairs only
        Z = np.column_stack([np.ones(train_end - 1), obs[:train_end - 1]])
        Phi0, *_ = np.linalg.lstsq(Z, y[1:train_end], rcond=None)
        R0 = (Z.T @ Z) / (train_end - 1)
    learner = Learner(Phi0=Phi0, R0=R0)

    fc = np.empty((T - 1, 2))
    for t in range(T - 1):
        fc[t] = learner.predict(obs[t])         # phi_{t-1}: predetermined
        learner.update(obs[t], y[t + 1], gain)  # absorb (obs_t -> obs_{t+1})
    return fc


def forecast_var_fixed(obs, train_end):
    """Fixed-coefficient VAR(1) in observables: OLS on training, FROZEN on holdout.

    This is exactly the adaptive learner with gain=0 (same training-OLS init, no
    updating). It is the third baseline that decomposes the learning win:

        RE structural  -> VAR fixed     : the cost of the structural restrictions
                                          (a reduced form beats the NK cross-eq
                                          restrictions) — NOT adaptation.
        VAR fixed       -> VAR adaptive : the value of ADAPTATION alone, since the
                                          only difference is gain>0 vs gain=0.
    """
    return forecast_learning(obs, gain=0.0, train_end=train_end)


def run_comparison(obs, p: NKParams = None, train_end=None, holdout=None,
                   gain=0.04, hac_lags=0):
    """Three-baseline comparison (RE, fixed VAR, adaptive VAR) with Clark-West.

    Per target returns RMSEs for all three and two nested CW tests:
      cw_restrictions: RE (restricted)  vs fixed VAR (free reduced form)
      cw_adaptation:   fixed VAR (gain=0) vs adaptive VAR (gain>0)  <- the thesis
    `obs` is a (T,3) array of demeaned observables [x, pi, i].
    """
    from eel.evaluation import clark_west as cw

    p = p or NKParams()
    obs = np.asarray(obs, float)
    F, R = nk_dsge.re_reduced_form(p)
    T = len(obs)
    if train_end is None:
        train_end = T // 2
    if holdout is None:
        holdout = np.arange(train_end + 1, T)
    holdout = np.asarray(holdout)

    f_re = forecast_rational(obs, F, R)              # structural (restricted)
    f_fixed = forecast_var_fixed(obs, train_end)     # free reduced form, frozen
    f_adapt = forecast_learning(obs, gain, train_end)  # free reduced form, adaptive

    origins = holdout - 1
    results = {"cond_F": float(np.linalg.cond(F)), "gain": gain,
               "n_holdout": int(len(holdout)),
               "rho": (p.rho_g, p.rho_u, p.rho_v), "targets": {}}

    def rmse(f, j):
        return float(np.sqrt(np.mean((obs[holdout, j] - f[origins, j])**2)))

    for j, name in enumerate(["x", "pi"]):
        actual = obs[holdout, j]
        results["targets"][name] = {
            "rmse": {"re": rmse(f_re, j), "var_fixed": rmse(f_fixed, j),
                     "var_adaptive": rmse(f_adapt, j)},
            # RE (restricted) vs fixed VAR (larger): cost of structural restrictions
            "cw_restrictions": cw.clark_west(actual, f_re[origins, j],
                                             f_fixed[origins, j], hac_lags),
            # fixed VAR (restricted, gain=0) vs adaptive VAR (larger): adaptation
            "cw_adaptation": cw.clark_west(actual, f_fixed[origins, j],
                                           f_adapt[origins, j], hac_lags),
        }
    return results
