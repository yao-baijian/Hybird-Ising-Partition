# QIS3 — Quantum-Inspired Solver v3

**Location:** `src/qis3/`

QIS3 combines **Simulated Bifurcation (SB)** with **Branch & Bound** and
**Adaptive Perturbation** to find high-quality Ising ground states.

## Three-Stage Algorithm

### Stage 1: Simulated Bifurcation (SB)
Generates an initial population of candidate solutions using the bounded SB
dynamics from `bsb_torch_batch()`. The best solution from the population
seeds the next stages.

### Stage 2: Branch & Bound
Recursively fixes the values of uncertain variables and solves reduced
subproblems:

1. **Variable selection**: pick the most uncertain variable (closest to $x_i = 0$)
2. **Branch**: try both $s_i = +1$ and $s_i = -1$, pruning branches whose
   lower bound exceeds the current best energy
3. **Subproblem solve**: run SB on the remaining free variables

Branch depth is controlled by `branch_depth` (default: 2).

### Stage 3: Adaptive Perturbation
Randomly perturbs the best solution and re-runs SB to escape local optima.
Enabled by default (`adaptive=True`).

## Files

| File | Purpose |
|------|---------|
| `qis3.py` | `QIS3` class with solve, branch & bound, perturbation |
| `utils.py` | QIS3 wrapper for balanced min-cut problems |
| `tests_qis3.py` | Comparison benchmarks: QIS3 vs BSB |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sb_type` | `'bsb'` | SB variant (`'bsb'` or `'dsb'`) |
| `num_iters` | 1000 | SB iterations per solve |
| `dt` | 0.1 | SB time step |
| `branch_depth` | 2 | Branch & Bound recursion depth |
| `popsize` | 10 | Initial SB population size |
| `adaptive` | True | Enable adaptive perturbation restarts |

## Usage

```python
from src.qis3.qis3 import QIS3
import torch

J = torch.randn(100, 100)
J = (J + J.T) / 2

solver = QIS3(J, branch_depth=2, popsize=16,
              num_iters=1000, device='cpu')
best_spins, best_energy = solver.solve()
```

## Benchmarks

`tests_qis3.py` compares QIS3 against standard BSB on Gset graph instances
(G6, G11, G35) across different $\lambda$ values and hyperparameters.
