# SBM — Simulated Bifurcation Machines

**Location:** `src/sbm/`

Simulated Bifurcation (SB) is a physics-inspired Ising solver that models spin
dynamics as a system of coupled, damped oscillators undergoing a bifurcation.

## Physical Model

Each spin variable $s_i \in \{\pm1\}$ is represented by a continuous particle
with position $x_i \in [-1, 1]$ and momentum $y_i$:

$$
\begin{aligned}
\dot{y}_i &= (-1 + \alpha(t)) x_i + \xi (J \vec{x})_i \\
\dot{x}_i &= y_i
\end{aligned}
$$

- $\alpha(t)$ anneals from 0 → 1, driving the bifurcation
- $\xi = 4/\sqrt{N}$ scales the coupling strength
- Boundary condition: $y_i = 0$ when $|x_i| > 1$ (hard clamping)
- At termination: $s_i = \text{sign}(x_i)$

## Files

| File | Purpose |
|------|---------|
| `sbm.py` | SB solvers: `sb`, `bsb_torch`, `bsb_torch_batch`, `bsb_bmincut_batch` |
| `utils.py` | Data loaders for Gset, DIMACS10, QPLIB, TSPLIB formats |

## Key Functions

### `bsb_torch_batch(J, init_x, init_y, num_iters, dt, ...)`
- **Main batch SB solver** with convergence detection
- Input: `J` (n×n coupling matrix), `init_x/y` (batch_size × n initial states)
- Returns: `(energies, solutions, es)`
- Supports `use_compile=True` for `torch.compile` acceleration

### `bsb_bmincut_batch(J, init_x, init_y, num_iters, dt, lambda_balance)`
- Specialized for **balanced min-cut** problems
- Uses modified coupling: $J_{\text{balanced}} = -0.5J - 2\lambda \cdot \mathbf{1}\mathbf{1}^T$
- Returns: `(energies, solutions, cut_values, imbalances)`

### `_bsb_step(x, y, J, alpha_i, xi, dt)`
- Single SB iteration, extracted for `torch.compile`
- Called in a loop by both `bsb_torch_batch` and `bsb_bmincut_batch`

### Legacy Functions
- `sb(sb_type, ...)` — NumPy-based single-trial SB (supports `'bsb'`, `'dsb'`, `'sb'`)
- `bsb_torch(J, ...)` — PyTorch single-trial bounded SB

## Usage

```python
from src.sbm.sbm import bsb_bmincut_batch
import torch

n, batch_size = 100, 40
J = torch.randn(n, n)
J = (J + J.T) / 2  # symmetric

init_x = 2 * torch.rand(batch_size, n) - 1
init_y = 2 * torch.rand(batch_size, n) - 1

energies, solutions, cuts, imb = bsb_bmincut_batch(
    J, init_x, init_y, num_iters=1000, dt=0.1,
    lambda_balance=1.0, use_compile=True,
)
```
