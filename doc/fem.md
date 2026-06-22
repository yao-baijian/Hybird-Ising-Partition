# FEM — Flexible Entropy Minimization Solver

**Location:** `src/fem/` and `src/hyper_solver.py` (`FemCoarsenSolver`)

The FEM solver uses a **mean-field approximation** with entropy-based optimization
to solve QUBO/Ising problems. It minimizes the free energy:

$$F = \mathbb{E}[H] - \frac{1}{\beta} S[p]$$

where $S[p]$ is the Shannon entropy of the marginal probabilities.

## Files

| File | Purpose |
|------|---------|
| `interface.py` | Main `FEM` class: factory methods, solver setup, solve entry point |
| `solver_fem.py` | `Solver` class with entropy-based iterative optimization |
| `problem.py` | Problem definitions (bmincut, maxcut, hypergraph, etc.) |
| `initial_partition.py` | FEM-based k=2 initial partition for coarse graphs |
| `cyclic_expansion.py` | Cyclic Expansion QUBO refinement for k-way partitioning |
| `utils.py` | Graph file parsing utilities |
| `hyper_solver.py` | `FemCoarsenSolver` — FEM/PUBO initial partition on coarsened **hypergraphs** (uses `FEM` internally) |

## Problem Types

| Type | Enum Value | Description |
|------|-----------|-------------|
| Balanced min-cut | `'bmincut'` | Minimize cut edges, penalize imbalance |
| Hypergraph min-cut | `'hyperbmincut'` | Cut minimization on hyperedges |
| Maximum cut | `'maxcut'` | Maximize cut edges (binary) |
| Max k-SAT | `'maxksat'` | Maximum satisfiability |
| Modularity | `'modularity'` | Community detection |
| Vertex cover | `'vertexcover'` | Minimum vertex cover (binary) |
| Custom | `'customize'` | User-defined objective/gradient |

## Solver Algorithm

### Entropy Functions

- **Binary** ($q=2$): $S = -\sum_i [p_i \log p_i + (1-p_i)\log(1-p_i)]$
- **k-way** ($q>2$): $S = -\sum_{i,t} p_i(t) \log p_i(t)$

### Annealing Schedules

| Schedule | Formula | Use Case |
|----------|---------|----------|
| `'inverse'` | $\beta_t = 1/t$ | Default, broad exploration |
| `'lin'` | $\beta_t \in [\beta_{\min}, \beta_{\max}]$ | Linear ramp |
| `'exp'` | $\beta_t \in [e^{\log \beta_{\min}}, e^{\log \beta_{\max}}]$ | Fast early exploration |

### Gradient & Optimization

- **Autograd** (default): calls `free_energy.backward()` on the expectation
- **Manual gradient** (`manual_grad=True`): uses `problem.manual_grad(p)` directly
- **Optimizers**: Adam (default) or RMSprop
- **Acceleration**: `use_compile=True` enables `torch.compile` on the core step

## Cyclic Expansion Refinement

`cyclic_expansion_refine()` iteratively selects pairs of partition blocks,
constructs a QUBO subproblem for swapping vertices between them, and solves
it with FEM. This is the core of the MIER pipeline family.

## Usage

```python
from src.fem import FEM

# Load from file
case = FEM.from_file('bmincut', 'path/to/graph.txt', index_start=1)
case.set_up_solver(num_trials=10, num_steps=1000, q=2, dev='cpu')
config, result = case.solve()

# Or from coupling matrix
case = FEM.from_couplings('bmincut', n, m, J, node_weights=w)
case.set_up_solver(num_trials=10, num_steps=1000, q=4, use_compile=True)
config, result = case.solve()
```
