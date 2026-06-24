"""
Phase-2 experiment: decomposing the RE vs adaptive-learning forecast comparison
into THREE baselines, one-step-ahead out-of-sample on FRED data, with Clark-West.

  RE structural   ->  fixed VAR(1)    : cost of the structural restrictions
  fixed VAR(1)    ->  adaptive VAR(1) : value of ADAPTATION (gain>0 vs gain=0)

This separates "the NK is misspecified, any reduced form beats it" (jump a) from
"time-varying beliefs help" (jump b). Both forecast from the same observables
{x, pi, i} up to t; rho is estimated on the training window only.

The conclusions are FRAGILE — see `nk_phase2_findings.md`. This runner therefore
reports RMSE magnitudes (not just p-values), a split-sensitivity sweep, and the
caveats, so the result is not over-read.

Run:  python -m experiments.nk_re_vs_learning
"""

import numpy as np

from eel.data import fred
from eel.models.macro.nk_dsge import NKParams
from eel.evaluation import nk_forecast as nf

TRAIN_END = "2007-12-31"


def _run_split(df, train_end_date, gain=0.04):
    """Run the three-baseline comparison for a given train/holdout split."""
    dev = fred.demean_on_training(df, train_end_date)
    obs = dev[["x", "pi", "i"]].to_numpy()
    train_end = int((df.index <= train_end_date).sum())
    holdout = np.where(df.index > train_end_date)[0]
    holdout = holdout[holdout >= 1]
    if len(holdout) < 12 or train_end < 30:
        return None
    p = nf.estimate_shock_persistence(obs[:train_end])
    return nf.run_comparison(obs, p, train_end=train_end, holdout=holdout, gain=gain)


def _adp(res, name):
    """(p-value, RMSE gap fixed->adaptive) for a target."""
    t = res["targets"][name]
    gap = t["rmse"]["var_fixed"] - t["rmse"]["var_adaptive"]
    return t["cw_adaptation"]["pvalue_one_sided"], gap


def main():
    df, vintage = fred.load(start="1985-01-01", end="2019-12-31")
    print(f"data: {vintage};  {len(df)} quarters")

    res = _run_split(df, TRAIN_END)
    n = res["n_holdout"]
    print(f"base split: train ..{TRAIN_END[:7]}  holdout 2008-01..2019  (n={n})")
    print(f"shock persistence rho estimated on training = "
          f"({res['rho'][0]:.2f}, {res['rho'][1]:.2f}, {res['rho'][2]:.2f})")

    print("\n" + "=" * 74)
    print(f"THREE-BASELINE DECOMPOSITION  (fair RE, gain=0.04, n={n})")
    print("=" * 74)
    print(f"  {'target':<7}{'RE struct':>11}{'VAR fixed':>11}{'VAR adapt':>11}"
          f"{'  CW restr':>11}{'  CW adapt':>11}")
    for name in ["x", "pi"]:
        t = res["targets"][name]
        r = t["rmse"]
        print(f"  {name:<7}{r['re']:>11.4f}{r['var_fixed']:>11.4f}"
              f"{r['var_adaptive']:>11.4f}"
              f"{t['cw_restrictions']['pvalue_one_sided']:>11.3f}"
              f"{t['cw_adaptation']['pvalue_one_sided']:>11.3f}")

    print("\n  decomposition with MAGNITUDES (statistical significance is not size):")
    for name in ["x", "pi"]:
        r = res["targets"][name]["rmse"]
        d_restr = r["re"] - r["var_fixed"]
        d_adapt = r["var_fixed"] - r["var_adaptive"]
        print(f"    {name}:  restrictions {d_restr:+.4f} "
              f"({100*d_restr/r['re']:+.1f}% of RE)    "
              f"adaptation {d_adapt:+.4f} "
              f"({100*d_adapt/r['var_fixed']:+.2f}% of fixed)")

    # Split sensitivity — the adversarial verification's main finding: the
    # adaptation verdict flips with whether the 2008 crisis is in train or holdout.
    print("\n" + "=" * 74)
    print("SPLIT SENSITIVITY  —  CW(fix->adp) p [RMSE gain sign]   (the verdict flips)")
    print("=" * 74)
    print(f"  {'train ends':<12}{'n_hold':>7}{'inflation adapt':>20}{'gap adapt':>18}")
    for d in ["2004-12-31", "2007-12-31", "2009-12-31", "2011-12-31"]:
        s = _run_split(df, d)
        if s is None:
            continue
        pp, pg = _adp(s, "pi")
        xp, xg = _adp(s, "x")
        f = lambda p, g: f"p={p:.3f} [{'+' if g > 0 else '-'}]"
        print(f"  {d[:7]:<12}{s['n_holdout']:>7}{f(pp, pg):>20}{f(xp, xg):>18}")

    print("\n" + "-" * 74)
    print("VERIFIED reading (adversarially checked; see nk_phase2_findings.md):")
    print("* SOLID: relaxing the NK structural restrictions explains essentially")
    print("  all the forecast gain over RE — inflation -13.7% RMSE (p<0.001). (jump a)")
    print("* UNRESOLVED: belief adaptation (jump b) is small and FRAGILE — its sign")
    print("  and significance flip with the split and with PLM richness (VAR(2)),")
    print("  the 'significant' gap result is a CW artifact (worse RMSE), and n=48")
    print("  is underpowered. No robust evidence adaptation helps, here.")


if __name__ == "__main__":
    main()
