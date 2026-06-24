"""
Clark-West (2007) test of equal predictive accuracy for NESTED models.

When model 2 nests model 1 (here: adaptive learning nests rational expectations,
since RE is the gain -> 0 limit), the naive MSPE comparison is biased toward the
small model — the larger model's extra estimation noise inflates its MSPE even
when it is true. Clark-West adjusts for exactly that noise.

For one-step-ahead forecasts f1 (restricted = RE) and f2 (larger = learning) of a
target y:

    e1 = y - f1,  e2 = y - f2
    c_hat = e1^2 - [ e2^2 - (f1 - f2)^2 ]

The statistic is the t-ratio of the mean of c_hat (HAC standard error), compared
one-sided to N(0,1). c_hat > 0 (significantly) favours the larger model.

INTERPRETATION CAVEATS (read before quoting a number)
------------------------------------------------------
* Both models here are MISSPECIFIED (frozen calibration). Clark-West compares
  RELATIVE accuracy only. A rejection supports "learning forecasts better than
  RE", NOT "learning is the true / correct model".
* The holdout is short (~48 quarters). The test has LIMITED POWER at that length.
  A marginal p-value (~0.05-0.10) is weak evidence, not a strong result, and
  should be reported as such. `power_warning()` flags small samples.
* Diebold-Mariano is reported only as a descriptive secondary: it is INVALID
  under nesting (degenerate null), so it is not the basis for any claim.
"""

import numpy as np
from scipy.stats import norm


def _hac_se(z, lags):
    """Newey-West HAC standard error of the mean of z."""
    z = np.asarray(z, float)
    n = len(z)
    zc = z - z.mean()
    gamma0 = np.dot(zc, zc) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)            # Bartlett weight
        gk = np.dot(zc[k:], zc[:-k]) / n
        var += 2.0 * w * gk
    return np.sqrt(var / n)


def power_warning(n):
    """Honesty guard: flag short holdouts where the CW test has little power."""
    if n < 60:
        return (f"holdout n={n} quarters: the CW test has LIMITED POWER here; "
                f"treat a marginal p-value as weak evidence, not a strong result.")
    return None


def clark_west(y, f1, f2, hac_lags=0):
    """Clark-West adjusted-MSPE test. f1 = restricted (RE), f2 = larger (learning).

    Returns a dict with the statistic, one-sided p-value (H1: f2 better), each
    model's RMSE, the sample size, and a power warning for short holdouts.
    """
    y, f1, f2 = map(lambda a: np.asarray(a, float), (y, f1, f2))
    e1, e2 = y - f1, y - f2
    c_hat = e1**2 - (e2**2 - (f1 - f2)**2)
    mean = c_hat.mean()
    se = _hac_se(c_hat, hac_lags)
    stat = mean / se if se > 0 else np.nan
    return {
        "test": "clark_west_2007",
        "statistic": stat,
        "pvalue_one_sided": float(1.0 - norm.cdf(stat)),
        "favored": "learning" if mean > 0 else "rational",
        "rmse_re": float(np.sqrt(np.mean(e1**2))),
        "rmse_learning": float(np.sqrt(np.mean(e2**2))),
        "mean_adjusted": float(mean),
        "n": int(len(y)),
        "power_warning": power_warning(len(y)),
    }


def diebold_mariano(y, f1, f2, hac_lags=0):
    """Diebold-Mariano (DESCRIPTIVE ONLY — invalid under nesting). Two-sided."""
    y, f1, f2 = map(lambda a: np.asarray(a, float), (y, f1, f2))
    d = (y - f1)**2 - (y - f2)**2
    se = _hac_se(d, hac_lags)
    stat = d.mean() / se if se > 0 else np.nan
    return {
        "test": "diebold_mariano_DESCRIPTIVE_invalid_under_nesting",
        "statistic": float(stat),
        "pvalue_two_sided": float(2.0 * (1.0 - norm.cdf(abs(stat)))),
        "n": int(len(y)),
    }
