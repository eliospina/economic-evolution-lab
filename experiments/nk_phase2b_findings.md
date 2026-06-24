# NK Phase 2b — does belief ADAPTATION help? (resolving "jump b")

Goal: with maximal power and no specification-mining, test whether belief
adaptation (constant-gain learning) improves NK out-of-sample forecasts —
isolating it from "the NK is misspecified, any reduced form beats it" (Phase 2a).

Long split: train **1960Q1–1984Q4** (99q), holdout **1985Q1–2019Q4** (140q, 3× the
Phase-2a power). The Great-Inflation→Great-Moderation regime change sits at the
train/holdout boundary — the setting most favourable to adaptive tracking.

All findings below were **adversarially verified** (5 independent lenses, all ran
code; high confidence). Every number reproduces to machine precision.

## The headline is methodological: the naive test is ~60% oversized

The nested Clark-West for **fixed (frozen) vs constant-gain** is **not correctly
sized**. Constant gain has *non-vanishing* estimation error, so CW's adjustment
term `(f1−f2)²` never dies and the statistic is systematically positive under the
null. On fixed-coefficient (no-adaptation) synthetic data the asymptotic one-sided
5% test **rejects 60% of the time** (null CW mean +2.36, sd 2.1).

Therefore every CW/GW p-value here is **Monte-Carlo-calibrated** against a
fixed-coefficient null. Verified valid: feeding null draws through the MC-calibrated
pipeline rejects at **0.047** (target 0.05) with ~Uniform p-values. The calibration
is a genuine size correction, not a fudge.

> The mandatory size gate was decisive. It converted a fake **p = 0.000** into an
> honest non-result — **twice** (the confirmatory CW, and the GW conditional Wald).

## Confirmatory result (pre-registered): adaptation is NOT significant

VAR(1) PLM, inflation, gain 0.04, long split:

| | RMSE | vs fixed |
|---|---|---|
| fixed (no adaptation) | 0.1742 | — |
| constant-gain | 0.1635 | **+6.17%** |

CW stat 4.476 → asymptotic p = 0.000 (**invalid**) → **MC-calibrated p ≈ 0.15–0.17**
(stable across seeds, gains 0.02–0.08, reps, and alternative null DGPs 0.10–0.27;
never crosses 0.05). The observed CW stat sits only ~1 null-sd above the null mean.

**Reading:** even with 3× power and the most favourable split, the size-correct
test **fails to reject the no-adaptation null**. The +6.2% RMSE edge is *not
statistically distinguishable from estimation noise*. This is a failure-to-reject
at n=140, **not** proof that adaptation cannot help.

## The only marginal signal is a DIFFERENT, economically trivial contrast

Rolling fixed-window Giacomini-White compares **window-OLS vs constant-gain** —
"within a fixed memory, *discount* vs *weight equally*", a narrower notion of
adaptation than "adapt vs never". Inflation, R=80:

- RMSE 0.1993 → 0.1963 = **+1.5%** (economically trivial; 0.003 in inflation units).
- GW well-sized (null rejection 0.003, conservative); unconditional **MC p ≈ 0.02–0.04** — marginally significant.
- Conditional Wald: raw χ²₂ p = 0.000 is **invalid** (~99% oversized under the null); **MC-calibrated p = 0.39** — *not* significant. The striking conditional "signal" was a pure calibration artifact.

This contrast is deliberately quarantined from the confirmatory one and must not be
read as "adaptation works".

## Where is the (insignificant) edge? Not the Volcker transition

Share of cumulative squared-error reduction (inflation), by regime:

| period | n | RMSE fix | RMSE cg | share |
|---|---|---|---|---|
| Volcker transition 1985–1992 | 32 | 0.2136 | 0.2148 | **−3%** |
| Great Moderation 1993–2007 | 60 | 0.1518 | 0.1416 | 36% |
| Crisis + ZLB 2008–2019 | 48 | 0.1711 | 0.1487 | **68%** |

The edge is **not** in the regime change adaptation was supposed to help with — it
is **−3%** there. It concentrates in the crisis/ZLB. **Caveat:** the total is
insignificant and the shares are outlier-driven (2009Q2 ≈ 14% alone), so read the
localization as suggestive, not a stable distribution.

## Exploratory (Holm-corrected, MC-calibrated)

| spec | RMSE% | MC p | Holm | verdict |
|---|---|---|---|---|
| VAR(1) output gap | +27.6% | 0.024 | 0.072 | **borderline / MC-fragile** |
| VAR(1) inflation, gain 0.10 | +3.8% | 0.250 | 0.500 | no |
| VAR(2) inflation | +2.8% | 0.424 | 0.424 | no |

The output gap shows a **large** magnitude with a Holm p straddling 0.05 across
seeds — borderline, not a clean null; worth a dedicated, pre-registered follow-up
rather than dismissal. Inflation specs are clearly non-significant.

## Bottom line on jump (b)

> **There is no robust evidence that belief adaptation improves NK inflation
> forecasting on this data — even under conditions designed to favour it.** The
> apparent +6% gain vanishes once the 60%-oversized test is correctly sized
> (MC p ≈ 0.16). The only size-correct "significant" signal is a different,
> economically trivial (+1.5%) discounting contrast. The output-gap result is
> borderline and deserves a separate study. Both models remain misspecified
> throughout; all claims are about *relative* forecast accuracy.

The durable contribution is the **size-control discipline**: a constant-gain
learner breaks the standard nested-forecast asymptotics, and naive CW/GW p-values
are wildly anti-conservative. Reporting size-gated, MC-calibrated p-values beside
RMSE magnitudes is the honest way to evaluate adaptive-learning forecasts — and it
is what turned two `p = 0.000`s into non-results here.
