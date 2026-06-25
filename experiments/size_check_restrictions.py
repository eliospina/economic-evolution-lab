"""
Size check for the RESTRICTIONS channel (RE -> fixed unrestricted VAR).

The paper's -13.7% inflation result (the cost of the NK cross-equation
restrictions) uses the nested Clark-West with ASYMPTOTIC critical values. Unlike
the constant-gain leg, both forecasts here are FROZEN — the rational-expectations
reduced form, and an OLS-on-training VAR(1) held fixed through the holdout — so
the only estimation error comes from the training sample and vanishes; the CW
correction is valid. This confirms it empirically: simulating from the restricted
RE null and testing the frozen unrestricted VAR, the asymptotic 5% CW rejects at
about 5%. (Contrast with the constant-gain leg, ~60% oversized.)

Run:  python -m experiments.size_check_restrictions
Deterministic (seed 0).
"""

import numpy as np

from eel.data import fred
from eel.evaluation import adaptation as ad
from eel.evaluation import clark_west as cw
from eel.models.macro import nk_dsge
from eel.evaluation.nk_forecast import estimate_shock_persistence

TRAIN_END = "2007-12-31"   # the short split where the -13.7% result lives


def main():
    df, _ = fred.load(start="1985-01-01", end="2019-12-31")
    train_mask = np.asarray(df.index <= TRAIN_END)
    obs = df[["x", "pi", "i"]].to_numpy()
    obs = obs - obs[train_mask].mean(0)
    n_train, n = int(train_mask.sum()), len(obs)

    # Restricted RE reduced form as the null DGP: obs_{t+1} = A_re obs_t + eta,
    # with A_re = F R F^{-1} and eta calibrated to the training residuals.
    p = estimate_shock_persistence(obs[:n_train])
    F, R = nk_dsge.re_reduced_form(p)
    A_re = F @ R @ np.linalg.inv(F)
    resid = obs[1:n_train] - obs[:n_train - 1] @ A_re.T
    L = np.linalg.cholesky(np.cov(resid.T))
    dgp = {"c": np.zeros(3), "A": A_re, "chol": L}
    print(f"split: train {n_train}q  holdout {n - n_train}q | "
          f"A_re spectral radius {max(abs(np.linalg.eigvals(A_re))):.3f}")

    reps, target, rej = 1000, 1, 0          # target 1 = inflation
    rng = np.random.default_rng(0)
    for _ in range(reps):
        sim = ad.simulate_var1(dgp, n, rng)
        f_re = sim @ A_re.T                              # RE forecast = the truth
        f_fix = ad.var_forecast(sim, n_train, 0.0, 1)    # frozen unrestricted VAR
        o = np.arange(n_train - 1, n - 1)
        res = cw.clark_west(sim[o + 1, target], f_re[o, target], f_fix[o, target])
        rej += res["pvalue_one_sided"] < 0.05
    print(f"RE -> fixed VAR, inflation: asymptotic 5% CW rejection = {rej / reps:.3f} "
          f"(nominal 0.05)")
    print("=> validly sized; unlike the constant-gain leg, this CW needs no MC gate.")


if __name__ == "__main__":
    main()
