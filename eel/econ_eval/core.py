"""
A small evaluation harness for economic reasoning.

Each task produces *parameterised* instances with a ground-truth answer supplied
by the validated engine, so grading is objective and an answer cannot be looked
up — the solver has to actually reason. A "solver" is any callable
`instance -> answer`; this is model-agnostic (an LLM hook is a later step), so the
same harness scores a heuristic, an oracle, or a language model identically.

The point is falsifiability: a task is only included if the repo can check the
answer programmatically.
"""

from dataclasses import dataclass


@dataclass
class Instance:
    task: str            # task name
    params: dict         # the parameters that define this instance
    prompt: str          # the natural-language question posed to a solver
    reference: object    # ground-truth answer from the validated engine


class Task:
    """A family of parameterised, engine-graded questions."""
    name = "base"

    def generate(self, rng) -> Instance:
        raise NotImplementedError

    def grade(self, instance: Instance, candidate) -> bool:
        raise NotImplementedError


def make_instances(task: Task, n: int, seed: int = 0):
    """Generate n reproducible instances of a task."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return [task.generate(rng) for _ in range(n)]


def score(task: Task, instances, solver) -> float:
    """Fraction of instances a solver answers correctly."""
    correct = sum(bool(task.grade(inst, solver(inst))) for inst in instances)
    return correct / len(instances)
