"""
Task: is a New Keynesian equilibrium determinate?

A random quarterly calibration of the three-equation NK model is drawn (with
phi_pi straddling 1, so the answer genuinely varies), and the solver must decide
whether the rational-expectations equilibrium is determinate. The ground truth is
the engine's Blanchard-Kahn verdict — not a lookup.

The task discriminates depth of reasoning. The exact condition for this model is
the *generalised* Taylor principle

    phi_pi + ((1 - beta) / kappa) * phi_x > 1,

not the textbook shorthand phi_pi > 1. A solver that knows only the shorthand is
right most of the time but wrong near the boundary, where the output-gap response
flips the verdict — exactly the kind of gap an eval should expose.
"""

from eel.models.macro import nk_dsge
from eel.models.macro.nk_dsge import NKParams
from eel.econ_eval.core import Instance, Task


class DeterminacyTask(Task):
    name = "nk_determinacy"

    def generate(self, rng):
        p = NKParams(
            beta=0.99,
            sigma=round(float(rng.uniform(0.5, 2.0)), 2),
            varphi=round(float(rng.uniform(0.5, 3.0)), 2),
            theta=round(float(rng.uniform(0.55, 0.85)), 2),
            phi_pi=round(float(rng.uniform(0.75, 1.5)), 2),   # straddles 1
            phi_x=round(float(rng.uniform(0.0, 1.0)), 2),
        )
        sol, _ = nk_dsge.build(p)
        reference = bool(sol.blanchard_kahn)
        prompt = (
            "New Keynesian three-equation model (dynamic IS, NKPC, contemporaneous "
            "Taylor rule). Quarterly calibration: "
            f"beta={p.beta}, sigma={p.sigma}, varphi={p.varphi}, theta={p.theta}, "
            f"phi_pi={p.phi_pi}, phi_x={p.phi_x}. "
            "Is the rational-expectations equilibrium determinate (a unique "
            "non-explosive solution)? Answer True or False."
        )
        return Instance(
            self.name,
            dict(beta=p.beta, sigma=p.sigma, varphi=p.varphi, theta=p.theta,
                 phi_pi=p.phi_pi, phi_x=p.phi_x, kappa=p.kappa),
            prompt, reference)

    def grade(self, instance, candidate):
        return bool(candidate) == bool(instance.reference)


# --- reference solvers, to show the eval discriminates reasoning quality -------

def generalized_taylor_solver(instance):
    """Correct economic reasoning: the generalised Taylor principle."""
    p = instance.params
    return p["phi_pi"] + ((1.0 - p["beta"]) / p["kappa"]) * p["phi_x"] > 1.0


def naive_taylor_solver(instance):
    """Textbook shorthand only (phi_pi > 1) — right most of the time, not always."""
    return instance.params["phi_pi"] > 1.0
