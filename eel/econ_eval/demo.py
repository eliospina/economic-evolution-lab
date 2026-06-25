"""
Demonstration that the eval discriminates reasoning quality.

Three reference solvers are scored on the same instances:
  - generalised Taylor principle  (correct)        -> should be ~100%
  - naive phi_pi > 1 shorthand    (textbook only)  -> high but imperfect
  - coin flip                     (no reasoning)   -> base rate

If the eval had no discriminating power the three would tie. They do not.

Run:  python -m eel.econ_eval.demo
"""

import numpy as np

from eel.econ_eval.core import make_instances, score
from eel.econ_eval.tasks.determinacy import (
    DeterminacyTask, generalized_taylor_solver, naive_taylor_solver)


def main(n=300, seed=0):
    task = DeterminacyTask()
    insts = make_instances(task, n=n, seed=seed)
    base_rate = sum(i.reference for i in insts) / n

    rng = np.random.default_rng(123)
    coin = lambda inst: bool(rng.integers(0, 2))

    print(f"task: {task.name}   instances: {n}   "
          f"(determinate share = {base_rate:.0%})")
    print("-" * 56)
    print(f"  {'solver':<34}{'accuracy':>12}")
    print(f"  {'generalised Taylor principle (correct)':<34}"
          f"{score(task, insts, generalized_taylor_solver):>11.1%}")
    print(f"  {'naive phi_pi > 1 (textbook only)':<34}"
          f"{score(task, insts, naive_taylor_solver):>11.1%}")
    print(f"  {'coin flip (no reasoning)':<34}"
          f"{score(task, insts, coin):>11.1%}")
    print("-" * 56)
    print("The naive solver trails the correct one near the determinacy boundary,")
    print("where the output-gap response decides — the gap the eval is meant to find.")


if __name__ == "__main__":
    main()
