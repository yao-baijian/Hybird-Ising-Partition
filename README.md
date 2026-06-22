# fem-partition

A Python library for solving graph and hypergraph partition problems using the **QUBO** framework, with multi-level pipelines and hardware-inspired solvers.

## Problem Types

| Type | Description |
|------|-------------|
| **Balanced minimum cut** (normal graph) | Partition graph vertices into `k` equal-weight blocks while minimizing the cut edges |
| **Balanced minimum cut** (hypergraph) | Partition hypergraph vertices into `k` equal-weight blocks while minimizing the cut hyperedges (weighted-node-aware balance) |
| **Max-cut** | Partition graph vertices into two blocks maximizing the cut edges |
| **Max-SAT** | Approximate maximum satisfiability via QUBO encoding |

## Solvers

### Normal Graph Solvers

| Solver | Source | Description |
|--------|--------|-------------|
| **FEM** | `src/fem/` | Mean-field entropy minimization with simulated annealing |
| **SBM** | `src/sbm/` | Simulated Bifurcation — physics-inspired Ising machine solver |
| **KaFFPa / KaHIP** | `src/partition/kaffpa_multiway.py` | External Karlsruhe Fast Flow Partitioner (wrapper) |
| **METIS** | `src/partition/refine.py` | External METIS multi-level partitioner (wrapper) |
| **Cyclic Expansion** | `src/fem/cyclic_expansion.py` | Cyclic Expansion FEM refinement (arXiv 2312.15467v1) |

### Hypergraph Solvers

All in `src/hyper_solver.py`:

| Solver | Role | Description |
|--------|------|-------------|
| **KahyparLikeSolver** | Coarsening | HEM (heavy-edge matching) coarsening with optional LSH pre-coarsening; saves a `hierarchy_stack` for V-Cycle |
| **FemCoarsenSolver** | Initial partition | FEM or PUBO-based initial partition on a coarsened hypergraph |
| **HyperRefineSolver** | Refinement | FM (greedy incremental), MCTS rollouts, evolutionary search, or hybrid combinations |

These three solvers compose into a complete hypergraph pipeline (see below).

## Pipelines

### Normal Graph Pipeline

Multi-level partitioning combining coarsening, initial partitioning, and refinement via solver composition. Methods are registered in `src/method_registry.py` and dispatched by name:

| Method | Family | Init Solver | Refine Solver |
|--------|--------|-------------|---------------|
| `direct_fem` | DI | FEM (on full graph) | — |
| `direct_sbm` | DI | SBM (on full graph) | — |
| `kaffpa` | DML | KaFFPa (native multi-level) | KaFFPa |
| `init_fem_refine_kaffpa` | IECM | FEM (on coarse) | KaFFPa |
| `init_sbm_refine_kaffpa` | IECM | SBM (on coarse) | KaFFPa |
| `init_kaffpa_refine_fem` | MIER | KaFFPa (on coarse) | Cyclic Expansion FEM |
| `coarse_fem_refine_kaffpa` | IECM | FEM (on coarse) | KaFFPa |
| `coarse_kaffpa_refine_fem` | MIER | KaFFPa (on coarse) | Cyclic Expansion FEM |

Pipeline families:
| Family | Meaning | Flow |
|--------|---------|------|
| **DI** | Direct | Solver runs directly on the full graph (no coarsening) |
| **DML** | Direct Multi-Level | Native tool manages its own coarsening + refinement |
| **IECM** | Init + External Coarsen + Refine | Coarsen → FEM/SBM init → External refine |
| **MIER** | Multi-level Init + Ext. Refine | External init on coarse → Cyclic Expansion FEM refine |

### Hypergraph Pipeline

| Stage | Method | Description |
|-------|--------|-------------|
| **Coarsening** | HEM / LSH | Heavy-edge matching directly on hyperedges; optional MinHash/LSH pre-coarsening for large graphs. Intermediate levels are saved in a `hierarchy_stack`. |
| **Initial partition** | Greedy / FEM / PUBO | Initial assignment on the coarsest level — either greedy (built into `KahyparLikeSolver`) or optimization-based (`FemCoarsenSolver`). |
| **V-Cycle uncoarsening** | Iterative projection + refinement | Pop levels off the `hierarchy_stack` one-by-one. At each step: project the current assignment to the next finer level via the saved `remap` array, then refine immediately. |
| **Refinement** | FM / MCTS / Evolution / Hybrid | Greedy incremental FM (flow), Monte-Carlo tree search rollouts, small evolutionary search, or any combination in a configurable `mode_cycle`. |

```python
# Usage:
res = kahypar_solver.coarsen(hyperedges, num_nodes, q)       # coarsen
fem_part = fem_solver.initial_partition(...)                   # init
final = vcycle_uncoarsen(fem_part, res['hierarchy_stack'],     # V-Cycle
                         hyperedges, q, refine_solver)
```

## Acceleration

- **`torch.compile`** support (opt-in) for FEM `Solver.iterate()` and SBM `bsb_torch_batch` step function.

```python
compile_fem = True    # compile FEM Solver.iterate()
compile_sbm = True    # compile SBM bsb_torch_batch step function
```

## Project Structure

```
src/
├── hyper_solver.py      # Hypergraph solver: KahyparLikeSolver,
│                        #   FemCoarsenSolver, HyperRefineSolver, vcycle_uncoarsen
├── solver_base.py       # Solver base classes (Fem, Sbm, Kaffpa, Metis, Cyclic)
├── method_registry.py   # Pipeline method registry + JSON config loading
├── fem/                 # Flexible Entropy Minimization solver
├── partition/           # Multi-level partitioning (coarsen, refine, hyper_utils)
├── sbm/                 # Simulated Bifurcation Machines
├── qis3/                # Quantum-Inspired Solver v3
├── digcim/              # Digital Co-Ising Machine experiments
└── configs/             # Default solver JSON configs (fem, sbm, kaffpa, metis, cyclic)
tests/                   # Test suite and benchmarks
├── test_hyper_bmincut_coarsen.py  # Hypergraph V-Cycle benchmark
├── test_hyper_bmincut.py         # Hypergraph bmincut tests
├── test_bmincut_coarsen.py       # Multi-level coarsening benchmarks
├── test_bmincut.py               # Graph bmincut tests
├── test_bmincut_base.py          # Base test utilities
├── test_bmincut_gpu_boost.py     # GPU acceleration tests
├── plot_results.py               # Result plotting (5 plot types)
├── utils.py                      # Hypergraph I/O utilities
├── config/                       # Working config copies (gitignored)
└── build/                        # Benchmark CSV outputs
benchmarks/
├── bmincut/                      # Balanced min-cut benchmarks
├── maxcut/                       # Max-cut benchmarks (Gset, WK2000)
└── maxsat/                       # Max-SAT benchmarks
doc/                    # Module documentation
config/                 # Working solver configs (copied from src/configs/)
build/                  # Benchmark result CSVs
```

## Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate fem

# 2. Install PyTorch (see https://pytorch.org/)
pip3 install torch torchvision torchaudio

# 3. Optional: external partition tools
pip install pymetis             # METIS wrapper
pip install kahypar             # KaHyPar
pip install kahip               # KaFFPa/KaHIP
```

## Configuration

Each solver has a default JSON config under `src/configs/`. At runtime these are copied to `config/` (gitignored) where you can override them:

```
config/
├── fem.json
├── sbm.json
├── kaffpa.json
├── metis.json
└── cyclic.json
```

Use `method_registry.ensure_configs()` to populate the working directory, or manually edit the JSON files in `config/`.

## Running Tests

Run from project root:

```powershell
python -u tests/test_hyper_bmincut_coarsen.py   # Hypergraph V-Cycle (best-of-N trials)
python -u tests/test_hyper_bmincut.py           # Hypergraph bmincut tests
python -u tests/test_bmincut_coarsen.py         # Multi-level coarsening benchmarks
python -u tests/test_bmincut.py                 # Graph bmincut tests
python -u tests/test_bmincut_gpu_boost.py       # GPU acceleration tests
python -u tests/plot_results.py                 # Plot results (5 plot types)
```

### Hypergraph V-Cycle Test

`test_hyper_bmincut_coarsen.py` runs a configurable benchmark:
- Coarsens once with HEM matching
- Runs `num_runs` outer trials (different seeds) for greedy and FEM initial partitions
- Each trial performs a full V-Cycle uncoarsening with refinement at every hierarchy level
- Reports the best result in pipe-delimited table format:
  ```
  |powersim|4|88|0.047859578229574|140|0.046344235383255|15|
  ```

### Select Methods

Set `partition_methods` in any test file to pick which pipeline to run (see [Normal Graph Pipeline](#normal-graph-pipeline) table above).

## Documentation

See `doc/` for detailed module documentation:
- `doc/fem.md` — FEM solver details
- `doc/partition.md` — Partition pipeline details
- `doc/sbm.md` — Simulated Bifurcation details
- `doc/qis3.md` — Quantum-Inspired Solver v3 details

## References

- FEM framework: mean-field entropy minimization with annealing
- Simulated Bifurcation: Goto et al., Science Advances (2019)
- Cyclic Expansion: arXiv 2312.15467v1
- KaHyPar: Schlag et al., SEA (2016)
- KaFFPa/KaHIP: Sanders & Schulz, ALENEX (2011)
- KaHIP — Karlsruhe High Quality Partitioning (kahip.github.io)
