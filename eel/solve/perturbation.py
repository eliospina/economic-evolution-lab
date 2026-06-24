"""
First-order perturbation solver for linear rational-expectations models.

A model is declared symbolically (SymPy):

  - `states`   : predetermined variables (e.g. capital, the productivity state)
  - `controls` : forward-looking / jump variables (e.g. consumption, labour)
  - `equations`: the equilibrium conditions  E_t[ f_i(.) ] = 0, written with the
                 current-period symbols and their next-period counterparts
  - `steady_state`: the deterministic steady state of every variable
  - `shocks`   : which state receives an exogenous innovation, and its std

The solver does three things, all of them inspectable:

  1. **Linearises automatically.** SymPy differentiates the equilibrium
     conditions and evaluates the Jacobians at the steady state, producing a
     *log-linear* system  A E_t w_{t+1} = B w_t  (coefficients are elasticities).
     The symbolic formalisation does real work here — it is not decoration.

  2. **Solves** the system with the generalized Schur (QZ) method of
     Klein (2000), returning the policy functions

         x_{t+1} = P x_t + R eps_{t+1}      (states)
         y_t     = F x_t                    (controls)

     and the Blanchard-Kahn diagnosis (existence + uniqueness of a stable path).

  3. **Returns the intermediate objects** (A, B, eigenvalues, P, F) so a human
     — or an AI auditor — can check the math rather than trust a "it ran" print.

Reference: Klein, P. (2000), "Using the generalized Schur form to solve a
multivariate linear rational expectations model", JEDC 24(10), 1405-1423.
"""

from dataclasses import dataclass
import numpy as np
import sympy as sp
from scipy.linalg import ordqz, solve_discrete_lyapunov


@dataclass
class Solution:
    """Result of a first-order solve. All matrices are real."""
    state_names: list      # order of the state vector x
    control_names: list    # order of the control vector y
    P: np.ndarray          # x_{t+1} = P x_t (+ R eps)
    F: np.ndarray          # y_t     = F x_t
    R: np.ndarray          # state loading on the shock(s)
    A: np.ndarray          # linearised system: A E_t w_{t+1} = B w_t
    B: np.ndarray
    eigenvalues: np.ndarray  # dynamic eigenvalues (modulus < 1 => stable)
    n_stable: int
    blanchard_kahn: bool     # n_stable == n_states  -> unique stable solution

    @property
    def names(self):
        return list(self.state_names) + list(self.control_names)


def _linearize(cur, nxt, equations, ss, scales):
    """Return (A, B) with  A E_t w_{t+1} = B w_t.

    For an equation f(w_{t+1}, w_t) = 0, the linear coefficient on the
    deviation of variable i is  scale_i * df/dvar_i  at the steady state.

      - log-deviation  (hat_i = d log var_i):   scale_i = ss_i
        because d/dhat f(var = ss*e^hat) = ss * df/dvar at hat = 0.
      - level deviation (hat_i = var_i - ss_i):  scale_i = 1
        the natural choice when the steady state is 0 (gaps, inflation, rates),
        where logs are undefined.

    Mixing the two per variable is what lets the same engine solve both a
    levels model (RBC) and a deviations model (the NK three-equation block).
    """
    n = len(cur)
    subs = {cur[i]: ss[i] for i in range(n)}
    subs.update({nxt[i]: ss[i] for i in range(n)})

    A = np.zeros((n, n))   # = J_next  (coeff on E_t w_{t+1})
    B = np.zeros((n, n))   # = -J_cur  (coeff on w_t)
    for j, eq in enumerate(equations):
        for i in range(n):
            d_next = sp.diff(eq, nxt[i]).subs(subs)
            d_cur = sp.diff(eq, cur[i]).subs(subs)
            A[j, i] = float(d_next) * scales[i]
            B[j, i] = -float(d_cur) * scales[i]
    return A, B


def solve(states, controls, cur, nxt, equations, steady_state, shocks,
          kinds=None, tol=1e-9):
    """Solve a model to first order.

    Parameters
    ----------
    states, controls : list[str]
        Names, in the order the symbols appear in `cur`/`nxt`. States first.
    cur, nxt : list[sympy.Symbol]
        Current- and next-period symbols, ordered [states..., controls...].
    equations : list[sympy.Expr]
        Equilibrium residuals (== 0). Must number len(cur).
    steady_state : dict[str, float]
        Steady-state level of every variable (keyed by name).
    shocks : dict[str, float]
        {state_name: std}. The named state(s) receive the innovation.
    kinds : dict[str, str] | None
        Per-variable linearisation: 'log' (log-deviation, the default) or
        'level' (level deviation, for gap/rate variables with a zero steady
        state). Defaults to 'log' for every variable.
    """
    names = list(states) + list(controls)
    n = len(names)
    n_s = len(states)
    assert len(cur) == len(nxt) == len(equations) == n, "size mismatch"

    kinds = kinds or {}
    ss = [float(steady_state[name]) for name in names]
    scales = [ss[i] if kinds.get(names[i], "log") == "log" else 1.0
              for i in range(n)]
    A, B = _linearize(cur, nxt, equations, ss, scales)

    # --- Generalized Schur (QZ), stable eigenvalues sorted to the top-left ---
    # Dynamic eigenvalue of the pencil is beta/alpha; stable iff |beta| < |alpha|.
    # This avoids dividing by zero for infinite/zero roots.
    def _stable(alpha, beta):
        return np.abs(beta) < np.abs(alpha)

    AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort=_stable, output="complex")

    with np.errstate(divide="ignore", invalid="ignore"):
        eig = beta / alpha
    n_stable = int(np.sum(np.abs(beta) < np.abs(alpha)))
    bk = (n_stable == n_s)

    if not bk:
        # Return what we have so the caller can diagnose; P/F left as NaN.
        nan_P = np.full((n_s, n_s), np.nan)
        nan_F = np.full((n - n_s, n_s), np.nan)
        return Solution(states, controls, nan_P, nan_F,
                        np.zeros((n_s, len(shocks))), A, B, eig, n_stable, bk)

    # --- Klein recovery ---------------------------------------------------
    # Transform w_t = Z s_t  =>  S s_{t+1} = T s_t  (S=AA, T=BB upper-tri).
    # Bounded solution requires the unstable coords s2_t = 0, hence
    #   k_t = Z11 s1_t,  u_t = Z21 s1_t  =>  u_t = Z21 Z11^{-1} k_t
    #   k_{t+1} = Z11 S11^{-1} T11 Z11^{-1} k_t
    Z11 = Z[:n_s, :n_s]
    Z21 = Z[n_s:, :n_s]
    S11 = AA[:n_s, :n_s]
    T11 = BB[:n_s, :n_s]

    Z11_inv = np.linalg.inv(Z11)
    F = (Z21 @ Z11_inv)
    P = (Z11 @ np.linalg.inv(S11) @ T11 @ Z11_inv)

    # Tiny imaginary parts are numerical noise; assert and drop.
    for M in (P, F):
        if np.max(np.abs(M.imag)) > 1e-7:
            raise RuntimeError("complex policy matrix — solver inconsistency")
    P, F = P.real, F.real

    # Shock loading on the states.
    R = np.zeros((n_s, len(shocks)))
    for col, (sname, std) in enumerate(shocks.items()):
        R[states.index(sname), col] = std

    return Solution(states, controls, P, F, R, A, B, eig, n_stable, bk)


# --------------------------------------------------------------------------
# Post-solution analytics: impulse responses and unconditional moments.
# --------------------------------------------------------------------------

def impulse_response(sol: Solution, horizon=40, shock_col=0, size=1.0):
    """Impulse responses (log-deviations) to a one-time innovation.

    Returns dict name -> array of length `horizon`. `size` is in std units.
    """
    n_s = len(sol.state_names)
    x = sol.R[:, shock_col] * size            # impact on states at t=0
    rows = {name: np.zeros(horizon) for name in sol.names}
    for t in range(horizon):
        y = sol.F @ x
        for i, name in enumerate(sol.state_names):
            rows[name][t] = x[i]
        for i, name in enumerate(sol.control_names):
            rows[name][t] = y[i]
        x = sol.P @ x
    return rows


def moments(sol: Solution):
    """Unconditional second moments via the discrete Lyapunov equation.

    Returns dict with per-variable std, correlation-with-first-control-or-state
    map keyed by name, and first-order autocorrelation.
    """
    n_s = len(sol.state_names)
    Sigma_x = solve_discrete_lyapunov(sol.P, sol.R @ sol.R.T)

    # Stack all variables: v_t = M x_t, with states = identity, controls = F.
    M = np.vstack([np.eye(n_s), sol.F])
    Sigma_v = M @ Sigma_x @ M.T
    cov_lag1 = M @ sol.P @ Sigma_x @ M.T       # Cov(v_t, v_{t-1})

    std = np.sqrt(np.clip(np.diag(Sigma_v), 0, None))
    names = sol.names
    out = {"std": {}, "autocorr": {}, "corr": {}}
    for i, name in enumerate(names):
        out["std"][name] = std[i]
        out["autocorr"][name] = (cov_lag1[i, i] / Sigma_v[i, i]
                                 if Sigma_v[i, i] > 0 else np.nan)
    # correlation matrix
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            denom = std[i] * std[j]
            out["corr"][(ni, nj)] = (Sigma_v[i, j] / denom
                                     if denom > 0 else np.nan)
    return out
