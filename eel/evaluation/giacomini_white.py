"""
Giacomini-White (2006) test of (conditional) predictive ability.

Unlike Diebold-Mariano (recursive/expanding scheme) or Clark-West (nested,
recursive), GW is built for the FIXED-LENGTH ROLLING window scheme: the finite
estimation window keeps parameter-estimation error from vanishing, so a standard
test of the loss differential is valid even for nested models. That is exactly the
setup of the rolling window-OLS vs constant-gain comparison.

Loss differential of one-step squared-error losses:
    d_t = (y_t - f1_t)^2 - (y_t - f2_t)^2        d_t > 0 favours model 2.

Unconditional GW (test function h_t = 1): t-ratio of mean(d) with HAC SE ~ N(0,1).
Conditional GW (h_t = [1, d_{t-1}]): Wald statistic n * gbar' Omega^{-1} gbar ~
chi^2_q, testing E[h_t d_t] = 0 — i.e. whether ANY function of the past predicts
which model will forecast better.

Reference: Giacomini, R. and H. White (2006), "Tests of Conditional Predictive
Ability", Econometrica 74(6), 1545-1578.
"""

import numpy as np
from scipy.stats import norm, chi2


def _hac(Z, lags):
    """Newey-West HAC covariance of the mean of rows of Z (n, q)."""
    Z = np.atleast_2d(Z)
    if Z.shape[0] < Z.shape[1] and Z.ndim == 2 and Z.shape[0] == 1:
        Z = Z.T
    n = Z.shape[0]
    Zc = Z - Z.mean(axis=0)
    S = (Zc.T @ Zc) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        G = (Zc[k:].T @ Zc[:-k]) / n
        S = S + w * (G + G.T)
    return S / n


def giacomini_white(y, f1, f2, conditional=True, hac_lags=0):
    """GW test. f1 = baseline, f2 = competitor; d>0 favours f2.

    Returns the unconditional one- and two-sided result and, if `conditional`,
    the conditional Wald statistic with instruments [1, d_{t-1}].
    """
    y, f1, f2 = map(lambda a: np.asarray(a, float), (y, f1, f2))
    d = (y - f1) ** 2 - (y - f2) ** 2
    n = len(d)

    # unconditional (h = 1)
    se = float(np.sqrt(_hac(d.reshape(-1, 1), hac_lags)[0, 0]))
    stat_u = d.mean() / se if se > 0 else np.nan
    out = {
        "test": "giacomini_white_2006",
        "n": int(n),
        "mean_loss_diff": float(d.mean()),
        "favored": "competitor" if d.mean() > 0 else "baseline",
        "uncond_stat": float(stat_u),
        "uncond_p_one_sided": float(1.0 - norm.cdf(stat_u)),   # H1: f2 better
        "uncond_p_two_sided": float(2.0 * (1.0 - norm.cdf(abs(stat_u)))),
    }

    if conditional and n > 3:
        h = np.column_stack([np.ones(n - 1), d[:-1]])          # [1, d_{t-1}]
        hd = h * d[1:, None]                                    # (n-1, 2)
        gbar = hd.mean(axis=0)
        Omega = _hac(hd, hac_lags)
        wald = float((n - 1) * gbar @ np.linalg.solve(Omega, gbar))
        out["cond_wald"] = wald
        out["cond_p"] = float(1.0 - chi2.cdf(wald, df=2))
    return out
