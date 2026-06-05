# fem-partition

A Python library for solving graph and hypergraph partition problems using the **QUBO** framework.

## Features

### Solvers
- **FEM** (`src/fem/`) — Mean-field entropy-based optimization with simulated annealing
- **SBM / Simulated Bifurcation** (`src/sbm/`) — Physics-inspired Ising machine solver
- **QIS3** (`src/qis3/`) — Quantum-Inspired Solver v3: SB + Branch & Bound + Adaptive Perturbation

### Problem Types
- Balanced minimum cut (normal graph)
- Balanced minimum cut (hypergraph)

### Partition Pipelines
Multi-level partitioning pipelines combining coarsening, FEM/SBM initial partitioning, and refinement via METIS, KaFFPa, KaHyPar, or Cyclic Expansion:

| Pipeline Family | Algorithm | Description |
|----------------|-----------|-------------|
| DI (Direct) | `fem` / `sbm` | FEM or SBM directly on full graph |
| DML (Direct Multi-Level) | `kaffpa` / `metis` / `kahypar` | Native tool multi-level |
| IECM (Init + External Coarsen + Refine) | `metis` / `kaffpa` / `kahypar` | Coarsen → FEM/SBM init → External refine |
| MIER (Multi-level Init + Ext. Refine) | `metis` / `kaffpa` / `kahypar` | External init on coarse → Cyclic Expansion FEM refine |

### Acceleration
- **`torch.compile`** support (opt-in) for FEM `Solver.iterate()` and SBM step functions

## Project Structure

```
src/
├── fem/              # Flexible Entropy Minimization solver
├── partition/        # Multi-level partitioning pipelines
├── sbm/              # Simulated Bifurcation Machines
└── qis3/             # Quantum-Inspired Solver v3
tests/                # Test suite and benchmarks
doc/                  # Module documentation
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

## Running Tests

Run from project root:

```powershell
python -u tests/test_bmincut_coarsen.py     # Multi-level coarsening benchmarks
python -u tests/test_hyper_bmincut.py       # Hypergraph bmincut tests
python -u tests/test_bmincut_gpu_boost.py   # GPU acceleration tests
python -u tests/plot_results.py             # Plot results (5 plot types)
```

### Partition Methods

Set `partition_methods` in any test file:

| Key | Description |
|-----|-------------|
| `'direct_fem'` | FEM directly on full graph |
| `'direct_sbm'` | SBM directly on full graph |
| `'kaffpa'` | KaFFPa multi-level partitioner |
| `'coarse_fem_refine_metis'` | Coarsen → FEM init → METIS refine |
| `'coarse_fem_refine_kaffpa'` | Coarsen → FEM init → KaFFPa refine |
| `'coarse_metis_refine_fem'` | Coarsen → METIS init → Cyclic Expansion FEM refine |
| `'coarse_kaffpa_refine_fem'` | Coarsen → KaFFPa init → Cyclic Expansion FEM refine |

### torch.compile

```python
compile_fem = True    # compile FEM Solver.iterate()
compile_sbm = True    # compile SBM bsb_torch_batch step function
```

## Documentation

See `doc/` for detailed module documentation.

## References

- FEM framework: mean-field entropy minimization with annealing
- Simulated Bifurcation: Goto et al., Science Advances (2019)
- Cyclic Expansion: arXiv 2312.15467v1
