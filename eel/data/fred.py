"""
FRED data for the NK out-of-sample exercise (Phase 2).

Series (public, no API key via pandas_datareader):
    GDPC1     real GDP (quarterly)
    GDPPOT    real potential GDP, CBO (quarterly)
    PCEPILFE  core PCE price index (monthly -> quarterly)
    FEDFUNDS  effective federal funds rate (monthly -> quarterly)

Constructed observables (all in quarterly percent, model units):
    x   = 100 * (ln GDPC1 - ln GDPPOT)         output gap (CBO)
    pi  = 100 * dln(PCEPILFE)                   quarterly core PCE inflation
    i   = FEDFUNDS / 4                          quarterly nominal rate

ZLB CAVEAT (note): over 2009-2015 FEDFUNDS is pinned near zero and is a poor
measure of the policy stance; the Wu-Xia shadow rate (Atlanta Fed, separate
download) is the robustness option. Not fetched here — baseline uses FEDFUNDS.

VINTAGE: FRED revises history, so results are only reproducible against a frozen
snapshot. `load(...)` caches the constructed series to a dated CSV under data/
and reloads it unless refresh=True.
"""

import os
import datetime as dt
import numpy as np
import pandas as pd

SERIES = ["GDPC1", "GDPPOT", "PCEPILFE", "FEDFUNDS"]


def _fetch(start, end):
    import pandas_datareader.data as web
    raw = {s: web.DataReader(s, "fred", start, end) for s in SERIES}
    q = {}
    # quarterly start-of-period index for alignment
    q["GDPC1"] = raw["GDPC1"]["GDPC1"].resample("QS").last()
    q["GDPPOT"] = raw["GDPPOT"]["GDPPOT"].resample("QS").last()
    q["PCEPILFE"] = raw["PCEPILFE"]["PCEPILFE"].resample("QS").mean()
    q["FEDFUNDS"] = raw["FEDFUNDS"]["FEDFUNDS"].resample("QS").mean()
    df = pd.DataFrame(q).dropna()
    out = pd.DataFrame(index=df.index)
    out["x"] = 100.0 * (np.log(df["GDPC1"]) - np.log(df["GDPPOT"]))
    out["pi"] = 100.0 * (np.log(df["PCEPILFE"]) - np.log(df["PCEPILFE"].shift(1)))
    out["i"] = df["FEDFUNDS"] / 4.0
    return out.dropna()


def load(start="1985-01-01", end="2019-12-31", cache_dir="data", refresh=False):
    """Load (or fetch + cache) the constructed quarterly observables.

    Returns (DataFrame[x,pi,i], vintage_str). Cached to a dated CSV so the exact
    numbers are reproducible even after FRED revises the underlying series.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"fred_nk_{start}_{end}.csv")
    if os.path.exists(cache) and not refresh:
        with open(cache) as f:
            vintage = f.readline().strip().lstrip("# ")
        df = pd.read_csv(cache, index_col=0, parse_dates=True, skiprows=1)
        return df, vintage

    df = _fetch(pd.Timestamp(start), pd.Timestamp(end))
    vintage = f"FRED pull {dt.date.today().isoformat()} ({start}..{end})"
    with open(cache, "w") as f:
        f.write(f"# {vintage}\n")
        df.to_csv(f)
    return df, vintage


def demean_on_training(df, train_end_date):
    """Map to model deviations by subtracting the TRAINING-window mean (only).

    The same means are applied to the whole sample so both models share the
    transform; nothing is computed on the holdout.
    """
    train = df.loc[:train_end_date]
    return df - train.mean()
