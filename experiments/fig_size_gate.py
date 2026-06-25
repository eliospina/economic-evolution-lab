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
from eel import figstyle as S

TRAIN_END = "1984-12-31"


def main():
    S.apply()

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
    nm, nsd = null.mean(), null.std()

    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    xmax = max(null.max(), obs_stat) + 0.6

    # simulated null (no-adaptation DGP)
    ax.hist(null, bins=48, density=True, color="#d9d9d9", edgecolor="#9e9e9e",
            linewidth=0.4, zorder=2,
            label="simulated null  (no-adaptation DGP)")

    # the assumed asymptotic null
    xs = np.linspace(-4, xmax, 500)
    ax.plot(xs, norm.pdf(xs), color=S.INK, lw=1.8, zorder=4,
            label=r"assumed null   $N(0,1)$")
    tail = xs >= 1.645
    ax.fill_between(xs[tail], norm.pdf(xs[tail]), color=S.GRAY, alpha=0.18, zorder=1)
    ax.axvline(1.645, color=S.GRAY, ls=(0, (2, 2)), lw=1.0, zorder=3)
    ax.text(1.645, 0.255, "5% crit. (1.645) ", color=S.GRAY, fontsize=8.5,
            rotation=90, va="center", ha="right")

    ax.axvline(nm, color=S.GRAY, ls=(0, (4, 2)), lw=1.0, zorder=3)
    ax.axvline(obs_stat, color=S.BRICK, lw=1.8, zorder=5)
    ax.text(obs_stat + 0.1, 0.30, f"observed {obs_stat:.2f}", color=S.BRICK,
            fontsize=9.5, va="center")

    card = (f"rejects {rate:.0%} at the 5% level   (nominal 5%)\n"
            f"null mean {nm:.2f},  sd {nsd:.2f}    (N(0,1): 0, 1)\n"
            f"asymptotic p = {1 - norm.cdf(obs_stat):.3f}   invalid\n"
            f"Monte-Carlo p = {mc_p:.3f}   size-corrected")
    ax.text(0.975, 0.74, card, transform=ax.transAxes, va="top", ha="right",
            family="Menlo", fontsize=8.6, color=S.INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=S.LINE, lw=0.8))

    ax.set_xlabel("Clark–West statistic")
    ax.set_ylabel("density")
    ax.set_xlim(-4, xmax)
    ax.set_title("Null distribution of the fixed-vs-constant-gain Clark–West statistic")
    ax.legend(loc="upper left")

    fig.savefig("results/size_gate.png", dpi=300)
    print(f"saved results/size_gate.png  "
          f"(null mean {nm:.2f}, reject {rate:.0%}, obs {obs_stat:.2f}, MC p {mc_p:.3f})")


if __name__ == "__main__":
    main()
