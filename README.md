# Economic Evolution Lab (EEL)

**A formalized, solvable and auditable library of economic models — built so that
a human (or an AI) can inspect the math, not just trust that "it ran."**

Today's economic models (RBC, NK-DSGE, …) were written for a pre-AGI world. EEL
implements them rigorously and reproducibly, then opens two doors: *auditing*
their fragile assumptions, and eventually *mutating* them when autonomous agents
change the rules of the game. The credible foundation comes first.

This repository follows the principles of the parent
[`economia-ai` charter](../economia-ai/CHARTER.md): transparency, reproducibility,
and **rigour before impression** — *being convincingly wrong is harm, not a
detail.*

---

## What's here now

| Layer | Path | Status |
|---|---|---|
| First-order perturbation engine (SymPy linearisation + Klein QZ) | `eel/solve/perturbation.py` | ✅ working |
| RBC — canonical, endogenous labour | `eel/models/macro/rbc.py` | ✅ solved + validated |
| Correctness tests (exact Brock-Mirman, Blanchard-Kahn, moments) | `tests/test_rbc.py` | ✅ passing |
| Calibration | `configs/rbc.yaml` | ✅ |

The RBC is the **template**: declare a model symbolically (steady state +
equilibrium conditions), and the same engine linearises and solves it. NK-DSGE,
Solow and Merton follow the same recipe.

## How it actually solves a model

1. **Formalise** — the equilibrium conditions are written in SymPy. This is not
   decoration: SymPy differentiates them to build the log-linear system.
2. **Solve** — `A E_t w_{t+1} = B w_t` is solved with the generalized Schur (QZ)
   method of Klein (2000), yielding policy functions and a Blanchard-Kahn check.
3. **Validate** — exact closed-form check (delta=1 ⇒ Brock-Mirman), steady-state
   identities, impulse-response signs, and business-cycle moments.

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"        # or: pip install -r requirements.txt

python -m eel.models.macro.rbc # solve, report moments + IRFs, save results/rbc_irf.png
pytest -q                      # run the correctness suite
```

Requires Python 3.9+. (Phase B — the AGI-native ABM + LLM-agent lab — needs 3.11+
and is intentionally not part of the default install.)

## Roadmap

- **A. Model zoo (now).** One model done *end to end* beats ten empty folders.
  RBC ✅ → NK-DSGE → Solow → Merton, each formalised, solved, validated.
- **B. AGI-native lab (next).** ABM (Mesa) + LLM agents (LangGraph) + mutable
  institutions, built *on top of* the validated zoo — never instead of it.

## Method (PhD standard)

Symbolic formalisation → reproducible numerical solution → rigorous validation
(moment matching, IRFs, sensitivity) → agentic audit of fragile assumptions.
Fixed seeds, pinned dependencies, config-driven runs.
