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
| `greedy_initial_hypergraph_partition(H, w, k, ...)` | Greedy balanced k-way hypergraph partition |
| `greedy_refine_hypergraph_incremental(assign, H, ...)` | Local refinement for hypergraph |

## Pipeline Families

```
DI (Direct):      fem / sbm
DML (Direct ML):  kaffpa / metis / kahypar / kahip
IECM (Init + Coarsen + Refine):  metis / kaffpa / kahypar / sbm+kaffpa
MIER (ML Init + Ext. Refine):    metis / kaffpa / kahypar
```
