"""
Phase 2b — isolating the value of ADAPTATION in NK forecasting.

Three benchmarks, all VAR(p) PLMs in observables, differing only in how beliefs
evolve through the holdout (all initialised by OLS on the training window):

    gain = 0.0   FIXED            frozen training coefficients      (no adaptation)
    gain = 'rls' DECREASING       1/t gain -> beliefs settle         (adaptation that fades)
    gain = g>0   CONSTANT-GAIN    persistent discounting             (persistent adaptation)

The CONFIRMATORY contrast is FIXED vs CONSTANT-GAIN ("adapt vs never adapt"),
tested with the nested Clark-West on a single long split.

A separate, DIFFERENT contrast lives in the rolling fixed-window Giacomini-White
exercise: window-OLS (equal weight) vs constant-gain (discounting) — "within a
fixed memory, discount vs weight equally". These are two distinct notions of
adaptation; the reporting must label them as such, not conflate them.

Both forecasters re-use eel.expectations.learning.Learner (raw regressor API).
This module is independent of the Phase-2 nk_forecast.py (whose committed results
must not move).
"""

import numpy as np

from eel.expectations.learning import Learner


def regressors(obs, lags):
    """VAR(lags) regressor matrix: row t = [1, obs_t, obs_{t-1}, ..., obs_{t-lags+1}].

    Rows t < lags-1 are NaN (insufficient history). obs is (T, k)."""
    obs = np.asarray(obs, float)
    T, k = obs.shape
    Z = np.full((T, 1 + lags * k), np.nan)
    Z[:, 0] = 1.0
    for L in range(lags):
        Z[L:, 1 + L * k: 1 + (L + 1) * k] = obs[:T - L]   # row t gets obs[t-L]
    return Z


def var_forecast(obs, train_end, gain, lags=1):
    """One-step forecasts of (x, pi) from a VAR(lags) PLM.

    Beliefs are initialised by OLS on the training window, then evolve through the
    HOLDOUT only (so training data is not double-counted):
      gain=0.0    -> frozen (fixed)
      gain='rls'  -> decreasing 1/(t+2) (continues the sample count; fades fast for
                     a long training window)
      gain=g>0    -> constant gain (persistent adaptation)
    Returns an (T-1, 2) array indexed by origin t (f[t] predicts obs[t+1]); entries
    outside the usable range are NaN.
    """
    obs = np.asarray(obs, float)
    T, k = obs.shape
    Z = regressors(obs, lags)
    Y = obs[:, :2]
    start = lags - 1

    tr = np.arange(start, train_end - 1)                 # training origins
    Phi0, *_ = np.linalg.lstsq(Z[tr], Y[tr + 1], rcond=None)
    R0 = (Z[tr].T @ Z[tr]) / len(tr)
    learner = Learner(Phi0=Phi0, R0=R0)

    fc = np.full((T - 1, 2), np.nan)
    for t in tr:                                         # frozen pre-holdout (unused)
        fc[t] = Phi0.T @ Z[t]
    for t in range(train_end - 1, T - 1):               # holdout walk
        fc[t] = learner.predict_raw(Z[t])
        g = 1.0 / (t + 2) if gain == "rls" else float(gain)
        learner.update_raw(Z[t], Y[t + 1], g)
    return fc


def rolling_forecasts(obs, window, gain, lags=1):
    """Fixed-length rolling-window forecasts for the Giacomini-White exercise.

    At each origin t with a full window [t-window+1, t], fit on the window's pairs:
      f_ols : window OLS, equal weight (no discounting)
      f_cg  : initialise at the window OLS, run constant gain through the window
              (discounting), then forecast t+1
    Both use the SAME fixed-length window -> GW's fixed-window asymptotics hold.
    Returns (f_ols, f_cg), each (T-1, 2) indexed by origin, NaN where unavailable.
    """
    obs = np.asarray(obs, float)
    T, k = obs.shape
    Z = regressors(obs, lags)
    Y = obs[:, :2]
    p = 1 + lags * k

    f_ols = np.full((T - 1, 2), np.nan)
    f_cg = np.full((T - 1, 2), np.nan)
    for t in range(window, T - 1):
        lo = max(t - window + 1 + (lags - 1), lags - 1)
        idx = np.arange(lo, t)                            # fitting origins s, target Y[s+1]<=Y[t]
        if len(idx) < p + 2:
            continue
        B, *_ = np.linalg.lstsq(Z[idx], Y[idx + 1], rcond=None)
        f_ols[t] = B.T @ Z[t]
        learner = Learner(Phi0=B, R0=(Z[idx].T @ Z[idx]) / len(idx))
        for s in idx:
            learner.update_raw(Z[s], Y[s + 1], float(gain))
        f_cg[t] = learner.predict_raw(Z[t])
    return f_ols, f_cg


# --------------------------------------------------------------------------
# Size control of the confirmatory test (MANDATORY gate).
# --------------------------------------------------------------------------

def fit_var1(obs):
    """Fit a fixed-coefficient VAR(1): obs_{t+1} = c + A obs_t + eps. Returns the
    DGP (c, A, chol(Sigma)) used to simulate a NO-ADAPTATION null."""
    obs = np.asarray(obs, float)
    X = np.column_stack([np.ones(len(obs) - 1), obs[:-1]])
    B, *_ = np.linalg.lstsq(X, obs[1:], rcond=None)       # (4, 3)
    resid = obs[1:] - X @ B
    Sigma = np.cov(resid.T)
    return {"c": B[0], "A": B[1:].T, "chol": np.linalg.cholesky(Sigma)}


def simulate_var1(dgp, n, rng, burn=200):
    """Simulate a fixed-coefficient VAR(1) path of length n (no time variation)."""
    c, A, L = dgp["c"], dgp["A"], dgp["chol"]
    k = len(c)
    s = np.zeros(k)
    out = np.empty((n, k))
    for t in range(-burn, n):
        s = c + A @ s + L @ rng.standard_normal(k)
        if t >= 0:
            out[t] = s
    return out


def size_control(dgp, reps, n_train, n_holdout, gain=0.04, lags=1, target=1,
                 alpha=0.05, seed=0):
    """Run the confirmatory test on synthetic FIXED-coefficient (no-adaptation)
    data and return the empirical rejection rate + the null CW-stat distribution.

    If the rate is near `alpha`, the nested Clark-West is correctly sized for this
    fixed-vs-constant-gain setup and a real-data positive is trustworthy. If it is
    inflated, use the returned null distribution for a Monte-Carlo-calibrated p.
    """
    from eel.evaluation import clark_west as cw
    rng = np.random.default_rng(seed)
    n = n_train + n_holdout
    stats = np.empty(reps)
    rej = 0
    for r in range(reps):
        obs = simulate_var1(dgp, n, rng)
        f_fix = var_forecast(obs, n_train, 0.0, lags)
        f_cg = var_forecast(obs, n_train, gain, lags)
        origins = np.arange(n_train - 1, n - 1)
        res = cw.clark_west(obs[origins + 1, target],
                            f_fix[origins, target], f_cg[origins, target])
        stats[r] = res["statistic"]
        rej += res["pvalue_one_sided"] < alpha
    return {"rejection_rate": rej / reps, "null_stats": stats,
            "nominal": alpha, "reps": reps}


def mc_pvalue(stat, null_stats):
    """Monte-Carlo one-sided p-value: P(null stat >= observed)."""
    null_stats = np.asarray(null_stats, float)
    return float((1 + np.sum(null_stats >= stat)) / (1 + len(null_stats)))


def gw_size_control(dgp, reps, n, window, gain=0.04, lags=1, target=1,
                    alpha=0.05, seed=1):
    """Size control for the rolling Giacomini-White contrast (window-OLS vs
    constant-gain) on fixed-coefficient null data. Returns rejection rate + null
    distributions for BOTH the unconditional stat and the conditional Wald, so
    each GW p can be MC-calibrated (the conditional chi^2_2 reference is badly
    mis-sized here for the same non-vanishing-estimation-error reason as CW)."""
    from eel.evaluation.giacomini_white import giacomini_white
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    cond = np.empty(reps)
    rej = 0
    for r in range(reps):
        obs = simulate_var1(dgp, n, rng)
        f_ols, f_cg = rolling_forecasts(obs, window, gain, lags)
        valid = np.where(~np.isnan(f_ols[:, target]) & ~np.isnan(f_cg[:, target]))[0]
        g = giacomini_white(obs[valid + 1, target], f_ols[valid, target],
                            f_cg[valid, target], conditional=True)
        stats[r] = g["uncond_stat"]
        cond[r] = g["cond_wald"]
        rej += g["uncond_p_one_sided"] < alpha
    return {"rejection_rate": rej / reps, "null_stats": stats,
            "null_cond_wald": cond}
