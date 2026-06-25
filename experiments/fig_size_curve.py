"""
Figure: the nested Clark-West size distortion as a function of holdout length.

Rejection rate of the nominal one-sided 5% test (frozen vs. constant-gain) under
a fixed-coefficient null, for several constant gains, as the holdout length n
grows. The distortion does not shrink with n -- it rises -- the signature of a
non-vanishing-estimation-error bias amplified by the sample size. A correctly
sized test would track the 5% line, and the curves are nearly invariant to the
gain. This is the econometric-idiom statement of the same point a size table
would make.

Regenerate:  python -m experiments.fig_size_curve   ->  results/size_curve.png
Deterministic (seed 0). ~1 min.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eel.data import fred
from eel.evaluation import adaptation as ad
from eel import figstyle as S

TRAIN_END = "1984-12-31"
NS = np.array([30, 60, 90, 120, 150, 180, 210, 240])
GAINS = [0.02, 0.04, 0.08]
REPS = 400


def main():
    S.apply()
    df, _ = fred.load(start="1960-01-01", end="2019-12-31")
    train_mask = np.asarray(df.index <= TRAIN_END)
    obs = df[["x", "pi", "i"]].to_numpy()
    obs = obs - obs[train_mask].mean(0)
    n_train = int(train_mask.sum())
    dgp = ad.fit_var1(obs)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    markers = ["o", "s", "^"]
    for k, g in enumerate(GAINS):
        rates = [ad.size_control(dgp, REPS, n_train, int(nh), gain=g, lags=1,
                                 target=1, seed=0)["rejection_rate"] for nh in NS]
        ax.plot(NS, rates, color=S.PALETTE[k + 1], ls=S.STYLES[k], lw=1.6,
                marker=markers[k], ms=5, label=f"gain = {g:.2f}")
        print(f"g={g:.2f}: " + " ".join(f"{r:.2f}" for r in rates))

    ax.axhline(0.05, color=S.GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(NS[0], 0.065, "nominal 5%", color=S.GRAY, ha="left", va="bottom",
            fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(NS[0] - 5, NS[-1] + 5)
    ax.set_xlabel("holdout length  n  (quarters)")
    ax.set_ylabel("rejection rate at the 5% level")
    ax.set_title("Size of the nested Clark–West under a no-adaptation null")
    ax.legend(loc="lower right", title="constant gain")

    fig.savefig("results/size_curve.png", dpi=300)
    print("\nsaved results/size_curve.png")


if __name__ == "__main__":
    main()
