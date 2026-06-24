# NK Phase 2 — RE vs adaptive learning: findings

One-step-ahead, out-of-sample forecast comparison on US quarterly FRED data
(1985–2019; train → 2007Q4 = 91q, holdout 2008Q1–2019Q4 = 48q). Observables
`{x = output gap, π = core-PCE inflation, i = fed funds/4}`, demeaned on training;
shock persistences `ρ` estimated on training only.

## Method: three baselines that decompose "learning beats RE"

| Baseline | What it is |
|---|---|
| **RE structural** | NK 3-equation reduced form `F R F⁻¹ obs_t` (theory cross-equation restrictions; ρ estimated on training) |
| **VAR fixed** | VAR(1) in observables, OLS on training, **frozen** on holdout (= adaptive learner at gain 0) |
| **VAR adaptive** | same VAR(1), beliefs updated by constant-gain RLS (gain 0.04) |

Two nested Clark-West (2007) tests isolate the two channels:

- **RE → VAR fixed** = *cost of the structural restrictions* (does any free reduced form beat the NK?). **Not adaptation.**
- **VAR fixed → VAR adaptive** = *value of adaptation alone* (the only difference is gain > 0 vs 0, identical OLS init). **This is the thesis.**

## Result at the base split (gain 0.04, n = 48)

| target | RE struct | VAR fixed | VAR adapt | CW(RE→fix) | CW(fix→adp) |
|---|---|---|---|---|---|
| x  | 0.6161 | 0.6550 | 0.6521 | 0.346 | 0.009 |
| π  | 0.1695 | 0.1463 | 0.1475 | 0.000 | 0.253 |

Decomposition (RMSE): inflation **restrictions −13.7%** (large), **adaptation −0.84%** (wrong sign);
output gap restrictions +6.3% wrong sign (RE best), adaptation +0.45% (trivial).

## What survives adversarial verification (5 independent lenses, all ran code)

**SOLID — jump (a), structural restrictions.** Relaxing the NK cross-equation
restrictions explains essentially the entire forecast gain over RE for inflation
(−13.7% RMSE, CW p < 0.001), robust to gain, HAC lags, demeaning, and multiplicity.
The statistical machinery and the code are correct: the fixed VAR is provably an
independent OLS-then-frozen VAR (max diff 0.0), the nesting is genuine, there is no
look-ahead, and all p-values reproduce by hand.

**UNRESOLVED / FRAGILE — jump (b), adaptation.** The data do **not** support a
robust claim either way:

1. **Split fragility (decisive).** The adaptation verdict flips with where the 2008
   crisis falls. CW(fix→adp) p-value [sign]:

   | train ends | n | inflation adapt | gap adapt |
   |---|---|---|---|
   | 2004Q4 | 60 | 0.094 [+] | 0.024 [−] |
   | 2007Q4 | 48 | 0.253 [−] | 0.009 [+] |
   | 2009Q4 | 40 | 0.012 [+] | 0.798 [−] |
   | 2011Q4 | 32 | 0.011 [+] | 0.798 [−] |

   With the crisis in the holdout, inflation adaptation *helps significantly*; with
   it in training, it doesn't — and the output-gap result swaps the opposite way.

2. **PLM dependence.** The inflation "no adaptation" result is specific to a VAR(1)
   PLM. Under a VAR(2) PLM, inflation adaptation becomes significant (CW p = 0.029).

3. **CW artifact at the base split.** The output-gap "significant" adaptation
   (p = 0.009) is **not** a real gain: adaptive RMSE is +0.45% *worse* than fixed,
   loses in 26/48 quarters, and ~62% of the CW signal is 3–4 crisis quarters
   (2009Q2 alone ≈ 33%). Clark-West's noise correction can flag "significant" even
   when the larger model forecasts worse.

4. **Power.** n = 48 (and 32–60 across splits) is underpowered; an insignificant CW
   is a failure-to-reject, not proof of no effect.

## Bottom line

The honest answer to "is the win in jump (a) or (b)?":

> **Jump (a) — relaxing the structural restrictions — is real, large, and robust,
> and accounts for essentially all of the forecast improvement over RE (inflation
> −13.7% RMSE). Jump (b) — belief adaptation — is too small and too fragile to
> claim in either direction at n = 48: its sign and significance flip with the
> sample split and the PLM order, and where it reads "significant" it is a
> Clark-West artifact masking a worse RMSE.**

So the original "learning beats RE on inflation" was structural misspecification,
not adaptation — and the corrected exercise cannot establish that adaptation helps
on this sample. A larger holdout (or a multi-country panel) and a richer PLM would
be needed to resolve jump (b). Both models remain misspecified throughout; every
claim here is about *relative* forecast accuracy.
