# Partition — Multi-level Graph & Hypergraph Partitioning

**Location:** `src/partition/`

Provides coarsening, refinement, and multi-level pipeline implementations
for both normal graphs and hypergraphs.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `coarsen.py` | Normal-graph matching-based coarsening |
| `refine.py` | FM-style local refinement & PyMetis wrapper |
| `hyper_coarsen.py` | Hypergraph coarsening (KaHyPar-like, LSH matching) |
| `hyper_refine.py` | Hypergraph refinement & balance repair |
| `hyper_utils.py` | Hypergraph utilities (clique expansion, cut evaluation) |
| `kaffpa_multiway.py` | Multi-level KaFFPa-style partitioner |
| `utils.py` | Coarse hyperedge building & PUBO wrapper |
| `tests.py` | Pytest unit tests |

## Normal-Graph Coarsening (`coarsen.py`)

### `coarsen_graph_by_matching(J, ...)`
- Greedy heavy-edge matching
- ~50% reduction per round
- Returns: `(J_coarse, node_weights, groups, original_to_coarse, rounds)`

### `expand_coarse_labels(groups, coarse_labels, n)`
- Projects coarse partition back to original fine graph

## Normal-Graph Refinement (`refine.py`)

### `_fm_refinement(adj, q, part, ...)`
- Fiduccia-Mattheyses algorithm with bucket queue
- Negative-gain moves for hill climbing
- Balance constraints with $\epsilon$ relaxation

### `simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, ...)`
- FM-style refinement with perturbation restarts
- Replaces external KaFFPa when wrapper doesn't support initial partitions

### `call_pymetis_with_part(q, adj, part=None, ...)`
- Calls PyMetis with optional initial partition
- Falls back to `_fm_refinement` if PyMetis doesn't support `part=` parameter

## Hypergraph Coarsening (`hyper_coarsen.py`)

### `coarsen_kahypar_like(hyperedges, num_nodes, target_size, ...)`
- Uses minhash signatures and LSH for hyperedge similarity
- Vertex feature vectors (incident degree, size, weight)
- Heavy-edge matching with feasibility checks

### `coarsen_fem_refine_kahypar(...)`
- Two-level: FEM init on coarse, KaHyPar refine on fine

## Hypergraph Refinement (`hyper_refine.py`)

### `hybrid_refine_partition(...)`
- Greedy refinement with optional KaHyPar calls
- Balance repair (fast and slow strategies)
- Incremental vertex move evaluation

## Multi-level KaFFPa (`kaffpa_multiway.py`)

Implements the full multi-level pipeline used by `kaffpa_kway` and `fem_multilevel_refine`:

1. **Coarsening**: `_he_match_one_round()` — heavy-edge matching, ~50% per round
2. **Initial partition**: greedy growing + FM (`initial_partition_greedy_fm`)
   or FEM (`initial_partition_fem`) or SBM (`initial_partition_sbm`)
3. **Uncoarsening & refinement**: `fm_refine_lookahead()` with boundary tracking
4. **Global polish**: perturbation restarts

### Key Entry Points

| Function | Init | Refine |
|----------|------|--------|
| `kaffpa_multiway_kway()` | Greedy + FM | Look-ahead FM |
| `fem_multilevel_refine()` | FEM QUBO solver | Look-ahead FM |
| `sbm_multilevel_refine()` | SBM solver | Look-ahead FM |

## Hypergraph Utilities (`hyper_utils.py`)

| Function | Description |
|----------|-------------|
| `build_clique_expanded_graph(H, ...)` | Convert hyperedges to clique-expanded sparse graph |
| `evaluate_kahypar_cut_value(assign, H, ...)` | Compute $(\lambda_e - 1) \cdot w_e$ cut metric |
| `greedy_initial_hypergraph_partition(H, w, k, ...)` | Greedy balanced k-way hypergraph partition (uses running tracking for $O(\deg(v))$ cost evaluation) |
| `greedy_refine_hypergraph_incremental(assign, H, ...)` | Local refinement for hypergraph with weighted-node support |

## Hypergraph Solver Pipeline (`src/hyper_solver.py`)

A self-contained hypergraph partitioning pipeline with three composable solver classes and a V-Cycle uncoarsening helper:

| Class / Function | Role | Description |
|------------------|------|-------------|
| `KahyparLikeSolver` | Coarsening | HEM (heavy-edge matching) directly on hyperedges; optional LSH pre-coarsening; saves a `hierarchy_stack` of all intermediate levels |
| `FemCoarsenSolver` | Initial partition | FEM or PUBO-based optimization on the coarsest level; supports clique/star expansion and configurable annealing schedules |
| `HyperRefineSolver` | Refinement | FM (greedy incremental), MCTS rollouts, evolutionary search, or hybrid combinations; weighted-node-aware balance repair |
| `vcycle_uncoarsen()` | Uncoarsening | Iterative projection + refinement through the saved hierarchy stack; extracts per-level node weights from groups; applies final pass on the original hypergraph |

### Pipeline Stages

```
Original hypergraph
    │
    ▼  ┌─────────────────────────────────────┐
    │  │ KahyparLikeSolver.coarsen()          │
    │  │  - HEM matching rounds               │
    │  │  - hierarchy_stack appended per round │
    │  └─────────────────────────────────────┘
    ▼
Coarsest level  ◄── initial partition (greedy / FEM / PUBO)
    │
    │  ┌── V-Cycle ──────────────────────────┐
    │  │  for each level (coarsest → finest): │
    │  │    1. project via remap               │
    │  │    2. HyperRefineSolver.refine()      │
    │  │       (FM / MCTS / evolution / hybrid)│
    │  └─────────────────────────────────────┘
    ▼
Original hypergraph  ◄── final refinement (unit weights)
```

### Weighted Balance Support

All levels of the pipeline (partition summary, balance limits, balance repair, and refinement) accept an optional `node_weights` array. When provided, block sizes are computed as the sum of vertex weights rather than raw counts, enabling correct balance enforcement during V-Cycle uncoarsening where coarse vertices represent clusters of varying sizes.

### Usage

```python
from src.hyper_solver import (
    KahyparLikeSolver, FemCoarsenSolver,
    HyperRefineSolver, vcycle_uncoarsen,
)

# 1. Coarsen (returns hierarchy_stack for V-cycle)
res = kahypar_solver.coarsen(hyperedges, num_nodes, q)

# 2. Initial partition on coarsest level
fem_part = fem_solver.initial_partition(
    res['coarse_hyperedges'], res['coarse_node_weights'], q,
)

# 3. V-Cycle uncoarsening + refinement
final = vcycle_uncoarsen(
    fem_part, res['hierarchy_stack'],
    hyperedges, q, refine_solver,
)
```

## Pipeline Families

```
DI (Direct):      fem / sbm
DML (Direct ML):  kaffpa / metis / kahypar / kahip
IECM (Init + Coarsen + Refine):  init_fem_refine_kaffpa / init_sbm_refine_kaffpa
MIER (ML Init + Ext. Refine):    init_kaffpa_refine_fem
```
