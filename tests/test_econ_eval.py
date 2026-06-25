"""
Tests for the economic-reasoning eval harness (seed task: NK determinacy).
"""

import numpy as np

from eel.econ_eval.core import make_instances, score
from eel.econ_eval.tasks.determinacy import (
    DeterminacyTask, generalized_taylor_solver, naive_taylor_solver)


def test_instances_are_a_nondegenerate_mix():
    """Generated instances include both determinate and indeterminate cases."""
    insts = make_instances(DeterminacyTask(), n=120, seed=0)
    refs = [i.reference for i in insts]
    assert 0 < sum(refs) < len(refs)


def test_correct_reasoning_matches_the_engine():
    """The generalised Taylor principle reproduces the engine's verdict exactly."""
    task = DeterminacyTask()
    insts = make_instances(task, n=120, seed=0)
    assert score(task, insts, generalized_taylor_solver) == 1.0


def test_eval_discriminates_reasoning_quality():
    """A coin flip scores far below the correct solver; the naive shorthand does
    not beat the correct one — so the eval has real discriminating power."""
    task = DeterminacyTask()
    insts = make_instances(task, n=200, seed=1)
    rng = np.random.default_rng(7)
    coin = lambda inst: bool(rng.integers(0, 2))
    correct = score(task, insts, generalized_taylor_solver)
    assert score(task, insts, coin) < correct - 0.2
    assert score(task, insts, naive_taylor_solver) <= correct


def test_grading_is_objective():
    """Grading compares against the engine reference, not the prompt text."""
    inst = make_instances(DeterminacyTask(), n=1, seed=0)[0]
    assert DeterminacyTask().grade(inst, inst.reference) is True
    assert DeterminacyTask().grade(inst, not inst.reference) is False
