"""
New Keynesian three-equation DSGE — baseline with rational expectations.

All variables are deviations from a zero-inflation steady state:
    x   output gap        pi  inflation        i   nominal interest rate
    g   demand shock      u   cost-push shock  v   monetary policy shock

Structural block (Galí 2008; Clarida-Galí-Gertler 1999)
    IS      x_t  = E_t x_{t+1} - (1/sigma)(i_t - E_t pi_{t+1}) + g_t
    NKPC    pi_t = beta E_t pi_{t+1} + kappa x_t + u_t
    Taylor  i_t  = phi_pi pi_t + phi_x x_t + v_t
    shocks  g,u,v each AR(1)
with  kappa = lambda (sigma + varphi),  lambda = (1-theta)(1-beta theta)/theta.

--------------------------------------------------------------------------
EXPECTATIONS ARE A SWAPPABLE MODULE — NOT BAKED INTO THE EQUATIONS.
--------------------------------------------------------------------------
The structural equations are written with explicit expectation *placeholders*
`E[x+]` and `E[pi+]` (symbols Ex, Epi). An ExpectationModule decides how those
placeholders are closed:

    RationalExpectations  (this file): closes E[w+] -> the next-period symbol,
        so the rational fixed point is solved by the perturbation engine.

    AdaptiveLearning      (Phase 2, planned below): forms E[w+] from a perceived
        law of motion + beliefs updated by recursive least squares, used in a
        forward simulation. The structural equations are NOT rewritten — only a
        different module is plugged in.

==========================================================================
PHASE 2 — adaptive learning (PLANNED, NOT IMPLEMENTED HERE)
==========================================================================
Replace RE with Evans-Honkapohja adaptive learning:
  * Agents hold a Perceived Law of Motion (PLM), e.g. linear in the shocks/
    states, and form E[x+], E[pi+] from it.
  * Beliefs update by recursive least squares (RLS) or constant-gain learning;
    the Actual Law of Motion (ALM) follows from substituting expectations into
    the structural block. Study E-stability / convergence to the RE solution.

Out-of-sample evaluation on real data (design notes):
  * Source: FRED (public, reproducible) via `fredapi`.
        output gap : GDPC1 vs GDPPOT (CBO potential)
        inflation  : PCEPILFE (core PCE) or GDPDEF
        nominal rate: FEDFUNDS; Wu-Xia shadow rate for the ZLB (2009-2015)
  * Compare adaptive learning vs RE one-step-ahead forecasts of pi and x.
  * STATISTICAL TEST: the two models are NESTED — RE is the limiting case of
    learning (as the gain -> 0 / beliefs converge). Under nesting the
    Diebold-Mariano test is invalid (degenerate/undersized null). Use the
    Clark-West (2007) adjusted-MSPE test for nested models; DM may be reported
    only as a secondary, non-nested-style descriptive.
  * GAIN DISCIPLINE: the learning gain is fixed a priori (e.g. constant gain
    ~0.02-0.05) OR estimated ONLY on the training window. It is NEVER tuned on
    the holdout — doing so would leak the test set and invalidate the comparison.
"""

from dataclasses import dataclass
import numpy as np
import sympy as sp

from eel.solve import perturbation


@dataclass
class NKParams:
    beta: float = 0.99       # discount factor
    sigma: float = 1.0       # inverse intertemporal elasticity of substitution
    varphi: float = 1.0      # inverse Frisch elasticity of labour supply
    theta: float = 0.75      # Calvo price stickiness (avg duration ~4 quarters)
    phi_pi: float = 1.5      # Taylor coefficient on inflation
    phi_x: float = 0.125     # Taylor coefficient on the output gap
    rho_g: float = 0.5       # persistence: demand shock
    rho_u: float = 0.5       # persistence: cost-push shock
    rho_v: float = 0.5       # persistence: monetary policy shock
    sigma_g: float = 0.01    # std: demand innovation
    sigma_u: float = 0.01    # std: cost-push innovation
    sigma_v: float = 0.01    # std: monetary policy innovation

    @property
    def kappa(self):
        lam = (1 - self.theta) * (1 - self.beta * self.theta) / self.theta
        return lam * (self.sigma + self.varphi)


# --------------------------------------------------------------------------
# Swappable expectation modules
# --------------------------------------------------------------------------

class ExpectationModule:
    """Decides how the E_t[·] placeholders in a model are closed.

    A module receives the list of (expectation_symbol, variable_name) pairs and
    the model's next-period symbol map, and returns how to resolve them. This is
    the seam that lets Phase 2 swap rational expectations for adaptive learning
    without touching the structural equations.
    """
    def closure(self, expectations, next_symbol):  # pragma: no cover - interface
        raise NotImplementedError


class RationalExpectations(ExpectationModule):
    """Model-consistent expectations: E_t[w_{t+1}] = the actual next-period w.

    Returns a symbolic substitution so the rational fixed point is solved by the
    generalized-Schur perturbation engine.
    """
    name = "rational"

    def closure(self, expectations, next_symbol):
        return {e_sym: next_symbol[var] for (e_sym, var) in expectations}


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

def build(p: NKParams = None, expectations: ExpectationModule = None):
    """Build the NK block and solve it under the given expectation module."""
    p = p or NKParams()
    expectations = expectations or RationalExpectations()
    s, k, b = p.sigma, p.kappa, p.beta

    # states (predetermined exogenous shocks) and controls (jumps)
    states = ["g", "u", "v"]
    controls = ["x", "pi", "i"]
    g, u, v, x, pi, i = sp.symbols("g u v x pi i")
    gp, up, vp, xp, pip, ip = sp.symbols("gp up vp xp pip ip")
    next_symbol = {"g": gp, "u": up, "v": vp, "x": xp, "pi": pip, "i": ip}

    # explicit expectation placeholders — the only forward terms in the model
    Ex, Epi = sp.symbols("Ex Epi")

    # structural block written with placeholders, NOT next-period symbols
    raw = [
        gp - p.rho_g * g,                              # demand AR(1)
        up - p.rho_u * u,                              # cost-push AR(1)
        vp - p.rho_v * v,                              # monetary AR(1)
        x - Ex + (1 / s) * (i - Epi) - g,              # IS
        pi - b * Epi - k * x - u,                      # NKPC
        i - p.phi_pi * pi - p.phi_x * x - v,           # Taylor rule
    ]

    # close the expectations via the plugged-in module
    sub = expectations.closure([(Ex, "x"), (Epi, "pi")], next_symbol)
    equations = [eq.subs(sub) for eq in raw]

    cur = [g, u, v, x, pi, i]
    nxt = [gp, up, vp, xp, pip, ip]
    ss = {name: 0.0 for name in states + controls}     # zero-inflation steady state
    kinds = {name: "level" for name in states + controls}
    shocks = {"g": p.sigma_g, "u": p.sigma_u, "v": p.sigma_v}

    sol = perturbation.solve(states, controls, cur, nxt, equations, ss, shocks,
                             kinds=kinds)
    return sol, p


# --------------------------------------------------------------------------
# CLI: solve, validate (Blanchard-Kahn + Taylor principle), report IRFs.
# --------------------------------------------------------------------------

def _report():
    sol, p = build()
    print("=" * 64)
    print("New Keynesian 3-equation DSGE — rational expectations")
    print("=" * 64)
    print(f"calibration: beta={p.beta} sigma={p.sigma} varphi={p.varphi} "
          f"theta={p.theta}")
    print(f"             phi_pi={p.phi_pi} phi_x={p.phi_x}  ->  kappa={p.kappa:.4f}")
    print(f"expectations module: {RationalExpectations.name}")

    print("\nrational-expectations diagnosis:")
    print(f"  stable eigenvalues = {sol.n_stable}  (need {len(sol.state_names)})")
    print(f"  Blanchard-Kahn (unique stable solution): {sol.blanchard_kahn}")

    # Taylor principle: phi_pi < 1 should break determinacy.
    sol_bad, _ = build(NKParams(phi_pi=0.9))
    print(f"  determinacy with phi_pi=0.9 (violates Taylor principle): "
          f"{sol_bad.blanchard_kahn}  (expected False)")

    # Impulse responses to each shock.
    shock_cols = {"g": 0, "u": 1, "v": 2}
    labels = {"g": "demand (+g)", "u": "cost-push (+u)", "v": "monetary (+v)"}
    print("\nimpulse responses on impact (t=0):")
    print(f"  {'shock':<16}{'x':>9}{'pi':>9}{'i':>9}")
    irfs = {}
    for sname, col in shock_cols.items():
        irf = perturbation.impulse_response(sol, horizon=20, shock_col=col)
        irfs[sname] = irf
        print(f"  {labels[sname]:<16}"
              f"{irf['x'][0]:>9.4f}{irf['pi'][0]:>9.4f}{irf['i'][0]:>9.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        for ax, sname in zip(axes, ["v", "u", "g"]):
            irf = irfs[sname]
            for var in ["x", "pi", "i"]:
                ax.plot(irf[var] * 100, label=var, linewidth=2)
            ax.axhline(0, color="k", linewidth=0.6)
            ax.set_title(labels[sname])
            ax.set_xlabel("quarters")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("% deviation from steady state")
        axes[0].legend()
        os.makedirs("results", exist_ok=True)
        out = "results/nk_irf.png"
        fig.suptitle("NK 3-equation DSGE — impulse responses")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nimpulse-response figure saved to {out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    _report()
