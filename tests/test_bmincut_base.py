import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from src.fem import FEM
import torch
import time
import numpy as np
import warnings
import os
import csv
from datetime import datetime
from utils import simple_kaffpa, coarsen_graph_by_matching, expand_coarse_labels, call_pymetis_with_part
from src.fem.problem import infer_bmincut
from src.fem.cyclic_expansion import cyclic_expansion_refine, adjacency_from_sparse
from src.fem.initial_partition import fem_initial_partition_kway
from src.fem.utils import read_graph
from src.partition.kaffpa_multiway import kaffpa_multiway_kway, fem_multilevel_refine
from src.method_registry import registry, ensure_configs
from typing import Dict, Tuple, Optional, List, Set

# ── Ensure per-method JSON configs exist in config/ ───────────────────────
ensure_configs()

try:
    import pymetis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    warnings.warn("pymetis is not installed.")

# try:
#     import metis
#     HAS_METIS = True
# except ImportError:
#     HAS_METIS = False
#     warnings.warn("metis is not installed.")

try:
    import kahip
    HAS_KAHIP = True
except ImportError:
    HAS_KAHIP = False
    warnings.warn("kahip is not installed.")

try:
    import kahypar
    HAS_KAHYPAR = True
except ImportError:
    HAS_KAHYPAR = False
    warnings.warn("kahypar is not installed.")
    
# ==========================================
# Select the partition method to run:
# 'direct_fem'                : Original FEM applied directly to normal graph
# 'coarse_fem_refine_metis'   : Multi-level coarsening + FEM coarse opt + METIS fine opt
# 'coarse_fem_refine_kahypar' : Multi-level coarsening + FEM coarse opt + KaHyPar fine opt
# 'coarse_fem_refine_kaffpa'  : Multi-level coarsening + FEM coarse opt + KaFFPa fine opt
# 'coarse_kaffpa_refine_fem'  : Multi-level coarsening + KaFFPa coarse opt + Cyclic Expansion FEM fine opt
# 'metis'                     : PyMetis graph partitioner alone
# 'kahypar'                   : KaHyPar partitioner alone
# 'kaffpa'                    : KaFFPa partitioner alone
# ==========================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
build_dir = 'build'
os.makedirs(build_dir, exist_ok=True)
csv_path = os.path.join(build_dir, f'bmincut_results_best_{timestamp}.csv')

fieldnames = [
    'instance',
    'q',
    'partition_method',
    'coarsen_to',
    'cut_value',
    'imbalance',
    'total_time_s',
    'coarsen_time_s',
    'init_partition_time_s',
    'refine_time_s',
]

col_w = (24, 4, 22, 10, 10, 12, 10)  # instance, q, method, coarsen_to, time, cut, imbalance

def print_header():
    header_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:<{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}} {{:>{col_w[5]}}} {{:>{col_w[6]}}}"
    sep = ' '.join(['-' * w for w in col_w])
    print(header_fmt.format('instance', 'q', 'method', 'coarsen_to', 'time(s)', 'cut', 'imbalance'))
    print(sep)

def print_row(best_row):
    row_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:<{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}.4f}} {{:>{col_w[5]}.1f}} {{:>{col_w[6]}.4f}}"
    print(row_fmt.format(best_row['instance'], best_row['q'], best_row['partition_method'], best_row['coarsen_to'], best_row['total_time_s'], best_row['cut_value'], best_row['imbalance']))

def save_to_csv(best_rows):
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)
    print(f"Saved best results to: {csv_path}")

def direct_fem(case_type, instance, index_start, num_trials, num_steps,
               anneal, dev, q, manual_grad, use_compile=False):
    
    case_bmincut = FEM.from_file(case_type, instance, index_start)
    case_bmincut.set_up_solver(num_trials, num_steps, anneal=anneal, dev=dev, q=q, manual_grad=manual_grad, use_compile=use_compile)
    
    init_start = time.perf_counter()
    config, result = case_bmincut.solve()
    partition_time_s = time.perf_counter() - init_start
    
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    p = config[optimal_inds[0]]
    cut = result.min().item()
    
    J = case_bmincut.problem.coupling_matrix
    
    return p, cut, partition_time_s

def direct_sbm(case_type, instance, index_start, num_trials, num_steps,
               anneal, dev, q, manual_grad, use_compile=False):
    """Direct SBM solver applied to balanced k-way min-cut.

    Uses Simulated Bifurcation (bsb_bmincut_batch) from sbm.py.
    For q > 2, uses recursive bisection: repeatedly bipartitions the
    largest remaining block until k parts are obtained.
    """
    from src.sbm.sbm import bsb_bmincut_batch

    case_bmincut = FEM.from_file(case_type, instance, index_start)
    J = case_bmincut.problem.coupling_matrix
    J = J.to(dev)
    
    n = J.shape[0]
    dt = 0.1

    init_start = time.perf_counter()

    if q == 2:
        # ── Direct SB for bipartition ──
        batch_size = num_trials
        init_x = 2 * torch.rand(batch_size, n, device=dev) - 1
        init_y = 2 * torch.rand(batch_size, n, device=dev) - 1

        _, sol, cut_values, _ = bsb_bmincut_batch(
            J, init_x, init_y, num_steps, dt, lambda_balance=1.0,
            use_compile=use_compile,
        )
        best_idx = torch.argmin(cut_values)
        spins = sol[best_idx]

        p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
        p[spins == 1, 0] = 1.0
        p[spins == -1, 1] = 1.0

    else:
        # ── Recursive bisection for k-way ──
        current_part = torch.zeros(n, dtype=torch.long, device=dev)
        n_parts = 1
        next_label = 1

        while n_parts < q:
            sizes = torch.bincount(current_part, minlength=n_parts)
            largest = int(torch.argmax(sizes).item())
            mask = current_part == largest
            sub_idx = torch.where(mask)[0]

            if len(sub_idx) <= 1:
                break

            # Extract subgraph
            J_sub = J[sub_idx][:, sub_idx]

            sub_n = len(sub_idx)
            batch_size = max(num_trials, 5)
            init_x = 2 * torch.rand(batch_size, sub_n, device=dev) - 1
            init_y = 2 * torch.rand(batch_size, sub_n, device=dev) - 1

            _, sol_sub, cut_sub, _ = bsb_bmincut_batch(
                J_sub, init_x, init_y, max(num_steps // n_parts, 50),
                dt, lambda_balance=1.0, use_compile=use_compile,
            )
            best_sub = int(torch.argmin(cut_sub).item())
            spins_sub = sol_sub[best_sub]  # +1/-1

            current_part[sub_idx[spins_sub == -1]] = next_label
            # spins_sub == +1 stays in current block (label = largest)
            next_label += 1
            n_parts += 1

        # Build one-hot matrix
        p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
        for i in range(n):
            label = int(current_part[i].item())
            p[i, min(label, q - 1)] = 1.0

    partition_time_s = time.perf_counter() - init_start

    _, cut = infer_bmincut(J, p.unsqueeze(0))
    cut = cut.item()

    return p, cut, partition_time_s

def metis_kway(J, q):
    init_start = time.perf_counter()
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    
    n = J.shape[0]
    adjacency_list = [[] for _ in range(n)]
    
    indices = J.indices()
    for idx in range(indices.shape[1]):
        r = int(indices[0, idx])
        c = int(indices[1, idx])
        if r != c:  # no self loops
            adjacency_list[r].append(c)

    edgecuts, parts = pymetis.part_graph(q, adjacency=adjacency_list)
    partition_time_s = time.perf_counter() - init_start
    
    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(parts):
        p[i, p_group] = 1.0
        
    _, cut = infer_bmincut(J, p.unsqueeze(0))
    cut = cut.item()

    return p, cut, partition_time_s

# ── Registry registration helpers ────────────────────────────────────────

def _register_method(name, family, algorithm, run_func, description="", solver_names=None):
    if name not in registry:
        registry.register(name, family, algorithm, description, solver_names=solver_names)
    registry.bind(name, run_func)
    return name


def init_fem_refine_metis(J, q, coarsen_to, num_trials, num_steps, anneal, dev, manual_grad):
    
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    
    # 1. Multi-level coarsening
    coarsen_start = time.perf_counter()
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, coarsen_rounds = coarsen_graph_by_matching(
        J,
        node_weights=torch.ones(n, dtype=torch.float32),
        coarsen_to=coarsen_to,
    )
    coarsen_time_s = time.perf_counter() - coarsen_start
    
    num_coarse_nodes = coarse_graph.shape[0]
    
    # 2. FEM solver on coarse graph
    init_start = time.perf_counter()
    case_bmincut_coarse = FEM.from_couplings(
        'bmincut',
        num_coarse_nodes,
        int(coarse_graph._nnz() // 2),
        coarse_graph,
        node_weights=coarse_node_weights,
    )
    case_bmincut_coarse.set_up_solver(num_trials, num_steps, dev=dev, q=q, anneal=anneal, manual_grad=manual_grad)
    config, result = case_bmincut_coarse.solve()
    
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    best_config = config[optimal_inds[0]]
    coarse_assignment = best_config.argmax(dim=1).cpu().numpy()
    init_partition_time_s = time.perf_counter() - init_start
    
    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
    
    refine_start = time.perf_counter()
    adjacency_list = [[] for _ in range(n)]
    indices = J.indices()
    for idx in range(indices.shape[1]):
        r = int(indices[0, idx])
        c = int(indices[1, idx])
        if r != c:  
            adjacency_list[r].append(c)

    edgecuts, parts = call_pymetis_with_part(q, adjacency_list, part=initial_assignment.tolist())
    refine_time_s = time.perf_counter() - refine_start
    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(parts):
        p[i, p_group] = 1.0
        
    _, cut = infer_bmincut(J, p.unsqueeze(0))
    
    return p, cut.item(), coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds, coarsen_rounds

# Legacy alias
coarse_fem_refine_metis = init_fem_refine_metis
_register_method("init_fem_refine_metis", "IECM", "metis", init_fem_refine_metis,
                  "Coarsen → FEM init → METIS refine", solver_names=["fem", "metis"])


def init_fem_refine_kaffpa(J, q, coarsen_to, num_trials, num_steps, anneal, dev, manual_grad, penalty = 1.0):
    """Multi-level coarsening + FEM initial partition + look-ahead FM refinement.
    
    Uses the same multi-level pipeline as kaffpa_multiway but replaces the
    greedy+FM initial partition on the coarsest graph with the FEM QUBO solver.
    """
    p, cut, tc, ti, tr, coarsen_rounds = fem_multilevel_refine(
        J, q, coarsen_to=coarsen_to,
        epsilon=0.05,
        refine_passes=10,
        fem_trials=num_trials,
        fem_steps=num_steps,
        fem_dev=dev,
        fem_anneal=anneal,
        seed=42,
        verbose=False,
    )
    return p, cut, tc, ti, tr, coarsen_rounds

def init_sbm_refine_kaffpa(J, q, coarsen_to, num_trials, num_steps, anneal, dev, manual_grad):
    """Multi-level coarsening + SBM initial partition + look-ahead FM refinement.

    Uses the same multi-level pipeline as kaffpa_multiway but replaces the
    greedy+FM initial partition on the coarsest graph with the SBM (Simulated
    Bifurcation) solver from sbm.py.
    """
    from src.partition.kaffpa_multiway import sbm_multilevel_refine
    return sbm_multilevel_refine(
        J, q, coarsen_to=coarsen_to,
        epsilon=0.05,
        refine_passes=10,
        sbm_trials=num_trials,
        sbm_steps=num_steps,
        sbm_dev=dev,
        seed=42,
        verbose=False,
    )


# Legacy alias
coarse_sbm_refine_kaffpa = init_sbm_refine_kaffpa

def kaffpa_kway(J, q, coarsen_to, epsilon=0.05, max_coarse_rounds=20, num_init_trials=5, refine_passes=10, seed=42, verbose=False):
    """Multi-level graph partitioning using kaffpa_multiway.
    
    Args:
        J: sparse coupling matrix (n x n)
        q: number of partitions
        coarsen_to: target coarsest graph size
        epsilon: allowed imbalance
        max_coarse_rounds: max coarsening rounds
        num_init_trials: trials for initial partition on coarsest graph
        refine_passes: FM refinement passes per level
        seed: random seed
        verbose: print progress
        
    Returns:
        (p, cut, coarsen_time_s, init_partition_time_s, refine_time_s)
    """
    p, cut, tc, ti, tr, coarsen_rounds = kaffpa_multiway_kway(
        J, q, coarsen_to=coarsen_to,
        epsilon=epsilon,
        max_coarse_rounds=max_coarse_rounds,
        num_init_trials=num_init_trials,
        refine_passes=refine_passes,
        seed=seed,
        verbose=verbose,
    )
    return p, cut, tc, ti, tr, coarsen_rounds

_register_method("kaffpa", "DML", "kaffpa", kaffpa_kway,
                  "Multi-level KaFFPa partitioner (greedy+FM init)")

def kahip_kway(J, q, coarsen_to):
    """Partition a graph using the KaHIP kaffpa function via the kahip Python package.
    
    Args:
        J: sparse coupling matrix (n x n)
        q: number of partitions
        coarsen_to: target coarsening size (unused, kept for interface compatibility)
        
    Returns:
        (p, cut, coarsen_time_s, init_partition_time_s, refine_time_s)
    """
    import kahip

    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]

    # Build CSR using kahip_graph for proper format
    # KaHIP expects integer edge weights, so we round float weights
    g = kahip.kahip_graph()
    g.set_num_nodes(n)

    indices = J.indices()
    values = J.values()
    seen: Set[Tuple[int, int]] = set()
    for idx in range(indices.shape[1]):
        r = int(indices[0, idx])
        c = int(indices[1, idx])
        w = values[idx].item()
        if r != c and (r, c) not in seen:
            g.add_undirected_edge(r, c, int(round(w)))
            seen.add((r, c))
            seen.add((c, r))

    coarsen_start = time.perf_counter()
    coarsen_time_s = time.perf_counter() - coarsen_start

    init_start = time.perf_counter()
    vwgt, xadj, adjcwgt, adjncy = g.get_csr_arrays()

    # Use ECO mode for balanced speed/quality
    edgecut, part = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,
                                  q, 0.03, 0, 0, int(kahip.ECO))
    init_partition_time_s = time.perf_counter() - init_start

    refine_time_s = 0.0

    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i, p_group in enumerate(part):
        p[i, p_group] = 1.0

    _, cut = infer_bmincut(J, p.unsqueeze(0))

    return p, cut.item(), coarsen_time_s, init_partition_time_s, refine_time_s

def init_metis_refine_fem(J, q, coarsen_to, anneal, dev, manual_grad, max_iterations, num_steps_cyclic, max_candidates, num_trials, patience, allow_nonadjacent, verbose=False):
    
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]

    coarsen_start = time.perf_counter()
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
    )
    coarsen_time_s = time.perf_counter() - coarsen_start
    num_coarse_nodes = coarse_graph.shape[0]

    # Build adjacency list for coarse METIS
    c_indices = coarse_graph.indices()
    c_values = coarse_graph.values()
    coarse_adj_list = [[] for _ in range(num_coarse_nodes)]
    for idx in range(c_indices.shape[1]):
        r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
        if r != c:
            coarse_adj_list[r].append(c)

    init_start = time.perf_counter()
    edgecuts, coarse_parts = pymetis.part_graph(num_coarse_nodes and q or q, adjacency=coarse_adj_list)
    coarse_assignment = np.array(coarse_parts)
    init_partition_time_s = time.perf_counter() - init_start

    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

    # Run cyclic expansion FEM refinement on the full graph
    refine_start = time.perf_counter()
    adjacency = adjacency_from_sparse(J)
    refined_assignment = cyclic_expansion_refine(
        adjacency,
        initial_assignment,
        q,
        max_iterations=max_iterations,
        max_candidates=max_candidates,
        num_trials=num_trials,
        num_steps=num_steps_cyclic,
        dev=dev,
        patience=patience,
        verbose=verbose,
        allow_nonadjacent=allow_nonadjacent,
    )
    refine_time_s = time.perf_counter() - refine_start

    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i in range(n):
        p[i, refined_assignment[i]] = 1.0

    _, cut = infer_bmincut(J, p.unsqueeze(0))
    
    return p, cut.item(), coarsen_time_s, init_partition_time_s, refine_time_s

# Legacy alias
coarse_metis_refine_fem = init_metis_refine_fem
_register_method("init_metis_refine_fem", "MIER", "metis", init_metis_refine_fem,
                  "Coarsen → METIS init → Cyclic Expansion FEM refine")


def init_kaffpa_refine_fem(J, q, coarsen_to, anneal, dev, manual_grad, max_iterations, num_steps_cyclic, max_candidates, num_trials, patience, allow_nonadjacent, verbose=False):
    
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]

    # Stage 1: Multi-level coarsening using matching-based coarsening
    coarsen_start = time.perf_counter()
    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
    )
    coarsen_time_s = time.perf_counter() - coarsen_start
    num_coarse_nodes = coarse_graph.shape[0]

    # Build adjacency for coarse graph for KaFFPa
    coarse_adj = [[] for _ in range(num_coarse_nodes)]
    c_indices = coarse_graph.indices()
    c_values = coarse_graph.values()

    for idx in range(c_indices.shape[1]):
        r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
        if r != c:
            coarse_adj[r].append((c, int(c_values[idx].item())))

    c_xadj = [0]
    c_adjncy = []
    c_adjcwgt = []
    for r in range(num_coarse_nodes):
        for c, w in coarse_adj[r]:
            c_adjncy.append(c)
            c_adjcwgt.append(w)
        c_xadj.append(len(c_adjncy))

    c_vwgt = coarse_node_weights.int().cpu().numpy().tolist()

    # Stage 2: Run KaFFPa on coarse graph to get initial q-way partition
    init_start = time.perf_counter()
    edgecut, coarse_assignment = simple_kaffpa(c_vwgt, c_xadj, c_adjcwgt, c_adjncy, q, epsilon=0.05, max_passes=10)
    coarse_assignment = np.array(coarse_assignment)
    init_partition_time_s = time.perf_counter() - init_start

    # Stage 3: Project coarse partition back to original graph
    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

    # Stage 4: Cyclic Expansion QUBO refinement using FEM
    # Convert sparse coupling matrix to adjacency list format
    refine_start = time.perf_counter()
    adjacency = adjacency_from_sparse(J)

    # Run Cyclic Expansion refinement
    refined_assignment = cyclic_expansion_refine(
        adjacency,
        initial_assignment,
        q,
        max_iterations=max_iterations,
        max_candidates=max_candidates,
        num_trials=num_trials,
        num_steps=num_steps_cyclic,
        dev=dev,
        patience=patience,
        verbose=verbose,
        allow_nonadjacent=allow_nonadjacent,
    )
    refine_time_s = time.perf_counter() - refine_start

    # Build output tensor
    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i in range(n):
        p[i, refined_assignment[i]] = 1.0

    _, cut = infer_bmincut(J, p.unsqueeze(0))

    return p, cut.item(), coarsen_time_s, init_partition_time_s, refine_time_s

# ── MethodName is defined in src/partition/method_registry.py ──────────────
from src.method_registry import MethodName


# ── Register all methods in the global registry ──────────────────────────
_register_method("direct_fem", "DI", "fem", direct_fem,
                  "FEM directly on full graph", solver_names=["fem"])
_register_method("direct_sbm", "DI", "sbm", direct_sbm,
                  "SBM directly on full graph", solver_names=["sbm"])
_register_method("init_fem_refine_metis", "IECM", "metis", init_fem_refine_metis,
                  "Coarsen → FEM init → METIS refine", solver_names=["fem", "metis"])
_register_method("init_fem_refine_kaffpa", "IECM", "kaffpa", init_fem_refine_kaffpa,
                  "Coarsen → FEM init → KaFFPa refine", solver_names=["fem", "kaffpa"])
_register_method("init_sbm_refine_kaffpa", "IECM", "sbm,kaffpa", init_sbm_refine_kaffpa,
                  "Coarsen → SBM init → KaFFPa refine", solver_names=["sbm", "kaffpa"])
_register_method("kaffpa", "DML", "kaffpa", kaffpa_kway,
                  "Multi-level KaFFPa partitioner", solver_names=["kaffpa"])
_register_method("init_metis_refine_fem", "MIER", "metis", init_metis_refine_fem,
                  "Coarsen → METIS init → Cyclic Expansion FEM refine",
                  solver_names=["metis", "cyclic"])
_register_method("init_kaffpa_refine_fem", "MIER", "kaffpa", init_kaffpa_refine_fem,
                  "Coarsen → KaFFPa init → Cyclic Expansion FEM refine",
                  solver_names=["kaffpa", "cyclic"])

# ── METHOD_NAME_MAP generated from registry ───────────────────────────────
METHOD_NAME_MAP = {name: registry[name].method_name for name in registry.keys()}



