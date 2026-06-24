"""
Figure: why the nested Clark-West must be size-calibrated.

Under a fixed-coefficient (no-adaptation) null, the fixed-vs-constant-gain CW
statistic is NOT distributed N(0,1): constant gain has non-vanishing estimation
error, so the statistic is systematically positive and the asymptotic 5% test
rejects ~60% of the time. The figure contrasts the simulated null distribution
with the assumed N(0,1), the asymptotic 5% threshold, and the observed statistic.

Regenerate:  python -m experiments.fig_size_gate   ->  results/size_gate.png
Deterministic (seed 0).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from eel.data import fred
from eel.evaluation import adaptation as ad
from eel.evaluation import clark_west as cw

TRAIN_END = "1984-12-31"


def main():
    df, _ = fred.load(start="1960-01-01", end="2019-12-31")
    train_mask = np.asarray(df.index <= TRAIN_END)
    obs = df[["x", "pi", "i"]].to_numpy()
    obs = obs - obs[train_mask].mean(0)
    n_train, n = int(train_mask.sum()), len(obs)
    dgp = ad.fit_var1(obs)

    sc = ad.size_control(dgp, reps=1000, n_train=n_train, n_holdout=n - n_train,
                         gain=0.04, lags=1, target=1, seed=0)
    null = sc["null_stats"]
    rate = sc["rejection_rate"]

    f_fix = ad.var_forecast(obs, n_train, 0.0, 1)
    f_cg = ad.var_forecast(obs, n_train, 0.04, 1)
    o = np.arange(n_train - 1, n - 1)
    obs_stat = cw.clark_west(obs[o + 1, 1], f_fix[o, 1], f_cg[o, 1])["statistic"]
    mc_p = ad.mc_pvalue(obs_stat, null)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(null, bins=45, density=True, color="#9bb8c9", edgecolor="white",
            linewidth=0.4, label="simulated null (fixed-coef DGP, no adaptation)")
    xs = np.linspace(-4, max(null.max(), obs_stat) + 0.5, 400)
    ax.plot(xs, norm.pdf(xs), color="#222222", lw=1.6,
            label="assumed asymptotic null  N(0,1)")
    ax.axvline(1.645, color="#444444", ls=":", lw=1.3)
    ax.text(1.70, ax.get_ylim()[1] * 0.40, "asymptotic 5%\nthreshold (1.645)",
            fontsize=8.5, color="#444444", rotation=90, va="center")
    ax.axvline(obs_stat, color="#b5651d", lw=2.0)
    ax.text(obs_stat, ax.get_ylim()[1] * 0.62,
            f" observed = {obs_stat:.2f}", fontsize=9, color="#b5651d")

    above = null[null >= obs_stat]
    ax.text(0.02, 0.97,
            f"null mean = {null.mean():.2f}  (not 0),  sd = {null.std():.2f}\n"
            f"asymptotic test rejects {rate:.0%} of the time at nominal 5%\n"
            f"asymptotic p = {1-norm.cdf(obs_stat):.3f}  (invalid)\n"
            f"MC-calibrated p = {mc_p:.3f}  (= share of null ≥ observed)",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", fc="#faf8f4", ec="#cccccc"))

    ax.set_title("The nested Clark-West is not N(0,1) for fixed-vs-constant-gain:\n"
                 "size calibration is required", fontsize=11)
    ax.set_xlabel("Clark-West statistic")
    ax.set_ylabel("density")
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/size_gate.png", dpi=130, bbox_inches="tight")
    print(f"saved results/size_gate.png  "
          f"(null mean {null.mean():.2f}, reject {rate:.0%}, obs {obs_stat:.2f}, MC p {mc_p:.3f})")


if __name__ == "__main__":
    main()
