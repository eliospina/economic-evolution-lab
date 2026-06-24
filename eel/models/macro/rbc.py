"""
Real Business Cycle model — canonical, with endogenous labour supply.

Households
    max E_0 sum_t beta^t [ log C_t + psi * log(1 - N_t) ]
Technology
    Y_t = Z_t K_t^alpha N_t^(1-alpha)
    K_{t+1} = (1 - delta) K_t + I_t
    Y_t = C_t + I_t
    log Z_{t+1} = rho log Z_t + eps_{t+1},   eps ~ N(0, sigma^2)

Equilibrium conditions
    Production        Y_t = Z_t K_t^alpha N_t^(1-alpha)
    Resource          Y_t = C_t + I_t
    Capital           K_{t+1} = (1-delta) K_t + I_t
    Labour (MRS=MPL)  psi C_t / (1 - N_t) = (1-alpha) Y_t / N_t
    Euler             1/C_t = beta E_t[ (1/C_{t+1})(alpha Y_{t+1}/K_{t+1} + 1-delta) ]
    Productivity      log Z_{t+1} = rho log Z_t

The steady state is solved analytically; the dynamics are solved by the
generic first-order perturbation engine in `eel.solve.perturbation`.

Set `endogenous_labour=False` to fix N (labour inelastic). With that flag,
delta=1 and log utility the model has the exact closed form K_{t+1}=alpha*beta*Y_t
(Brock-Mirman), which `tests/test_rbc.py` uses as an exact correctness check.
"""

from dataclasses import dataclass, field
import numpy as np
import sympy as sp

from eel.solve import perturbation


@dataclass
class RBCParams:
    alpha: float = 0.33     # capital share
    beta: float = 0.99      # discount factor (quarterly)
    delta: float = 0.025    # depreciation
    rho: float = 0.95       # persistence of productivity
    sigma: float = 0.007    # std of productivity innovation
    psi: float = None       # leisure weight (calibrated to target_n if None)
    target_n: float = 1.0 / 3.0   # steady-state hours target (used to set psi)


def steady_state(p: RBCParams, endogenous_labour=True):
    """Analytic steady state. Returns (dict of levels, calibrated psi)."""
    a, b, d = p.alpha, p.beta, p.delta
    mpk = 1.0 / b - 1.0 + d          # = alpha * Y/K
    y_over_k = mpk / a
    k_over_y = 1.0 / y_over_k
    c_over_y = 1.0 - d * k_over_y    # since I = delta K
    if c_over_y <= 0:
        raise ValueError("implied C/Y <= 0; check calibration")

    if endogenous_labour:
        # Labour FOC at SS: N = (1-a) / (psi*(C/Y) + (1-a)).
        # Calibrate psi to hit target_n if psi not given.
        if p.psi is None:
            n = p.target_n
            psi = (1.0 - a) * (1.0 - n) / (n * c_over_y)
        else:
            psi = p.psi
            n = (1.0 - a) / (psi * c_over_y + (1.0 - a))
    else:
        psi = 0.0
        n = 1.0

    k_over_n = y_over_k ** (-1.0 / (1.0 - a))   # K/N = (Y/K)^(-1/(1-a))
    K = k_over_n * n
    Y = y_over_k * K
    C = c_over_y * Y
    I = d * K
    Z = 1.0
    ss = {"k": K, "z": Z, "c": C, "n": n, "y": Y, "i": I}
    return ss, psi


def build(p: RBCParams = None, endogenous_labour=True):
    """Build symbols + equilibrium conditions and solve to first order."""
    p = p or RBCParams()
    ss, psi = steady_state(p, endogenous_labour)
    a, b, d, rho = p.alpha, p.beta, p.delta, p.rho

    # current and next-period symbols (positive => logs well-defined)
    k, z, c, n, y, i = sp.symbols("k z c n y i", positive=True)
    kp, zp, cp, np_, yp, ip = sp.symbols("kp zp cp np yp ip", positive=True)

    if endogenous_labour:
        states = ["k", "z"]
        controls = ["c", "n", "y", "i"]
        cur = [k, z, c, n, y, i]
        nxt = [kp, zp, cp, np_, yp, ip]
        equations = [
            y - z * k**a * n**(1 - a),                      # production
            y - c - i,                                      # resource
            kp - (1 - d) * k - i,                           # capital
            psi * c / (1 - n) - (1 - a) * y / n,            # labour FOC
            1 / c - b * (1 / cp) * (a * yp / kp + 1 - d),   # Euler
            sp.log(zp) - rho * sp.log(z),                   # productivity AR(1)
        ]
    else:
        # Labour fixed at N = 1 (its steady-state value here).
        states = ["k", "z"]
        controls = ["c", "y", "i"]
        cur = [k, z, c, y, i]
        nxt = [kp, zp, cp, yp, ip]
        equations = [
            y - z * k**a,                                   # production (N=1)
            y - c - i,                                      # resource
            kp - (1 - d) * k - i,                           # capital
            1 / c - b * (1 / cp) * (a * yp / kp + 1 - d),   # Euler
            sp.log(zp) - rho * sp.log(z),                   # productivity AR(1)
        ]

    sol = perturbation.solve(states, controls, cur, nxt, equations, ss,
                             shocks={"z": p.sigma})
    return sol, ss, psi


# --------------------------------------------------------------------------
# CLI: solve, validate, report moments and impulse responses, save a figure.
# --------------------------------------------------------------------------

def _report():
    p = RBCParams()
    sol, ss, psi = build(p)

    print("=" * 64)
    print("RBC — canonical, endogenous labour")
    print("=" * 64)
    print(f"calibration: alpha={p.alpha} beta={p.beta} delta={p.delta} "
          f"rho={p.rho} sigma={p.sigma}")
    print(f"psi calibrated to N*={p.target_n:.3f}  ->  psi={psi:.4f}")
    print("\nsteady state:")
    for name in ["y", "c", "i", "k", "n"]:
        print(f"  {name.upper():2s} = {ss[name]:.4f}")

    print("\nrational-expectations diagnosis:")
    print(f"  stable eigenvalues = {sol.n_stable}  (need {len(sol.state_names)})")
    print(f"  Blanchard-Kahn satisfied: {sol.blanchard_kahn}")
    mods = np.sort(np.abs(sol.eigenvalues))
    print("  |eigenvalues| =", ", ".join(f"{m:.3f}" for m in mods))

    m = perturbation.moments(sol)
    sy = m["std"]["y"]
    print("\nbusiness-cycle moments (relative to output):")
    print(f"  {'var':<4}{'sd':>8}{'sd/sd_y':>10}{'corr w/Y':>10}{'autocorr':>10}")
    for name in ["y", "c", "i", "n", "k"]:
        print(f"  {name.upper():<4}{m['std'][name]*100:>7.2f}%"
              f"{m['std'][name]/sy:>10.2f}"
              f"{m['corr'][(name, 'y')]:>10.2f}"
              f"{m['autocorr'][name]:>10.2f}")

    # Impulse responses to a 1% productivity shock.
    irf = perturbation.impulse_response(sol, horizon=40, size=1.0)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in ["y", "c", "i", "n"]:
            ax.plot(irf[name] * 100, label=name.upper(), linewidth=2)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title("RBC — impulse responses to a 1% productivity shock")
        ax.set_xlabel("quarters")
        ax.set_ylabel("% deviation from steady state")
        ax.legend()
        ax.grid(alpha=0.3)
        os.makedirs("results", exist_ok=True)
        out = "results/rbc_irf.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nimpulse-response figure saved to {out}")
    except Exception as e:                      # plotting is optional
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    _report()
