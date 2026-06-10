import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / 'tests'
for path in (ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.fem import FEM
import torch
import time
import numpy as np
import warnings

from src.hyper_solver import (
    KahyparLikeSolver,
    FemCoarsenSolver,
    HyperRefineSolver,
)
from src.partition.utils import build_coarse_hyperedges, make_q4_pubo_object
from utils import (
    _compute_summary,
    _log,
    _parse_run_label,
    _print_results_table,
    _print_result_row,
    build_clique_expanded_graph,
    coarsen_graph_by_matching,
    expand_coarse_labels,
    greedy_refine_hypergraph_incremental,
    parse_hypergraph_edges,
    PUBOObjective,
)

try:
    import kahypar  # type: ignore[import-not-found]

    HAS_KAHYPAR = True
except ImportError:
    HAS_KAHYPAR = False
    warnings.warn("KaHyPar is not installed. Will fallback to FEM where applicable.")

try:
    import pymetis  # type: ignore[import-not-found]

    HAS_METIS = True
except ImportError:
    HAS_METIS = False


num_trials = 1
num_steps = 200
dev = 'cpu'
instance = '../partition/full_benchmark_set/powersim.mtx.hgr'

# ==========================================
# Select the partition method(s) to run:
# 'direct_fem'             : Original FEM applied directly to the clique-expanded hypergraph
# 'coarsen_fem_refine_kahypar' : QUBO-based matching coarsening (FEM) + KaHyPar on coarse hypergraph
# 'coarsen_kahypar_refine' : Multi-level coarsening + KaHyPar initial guess + Greedy refinement
# 'kahyper_like'           : Self-implemented KaHyPar-like coarsening, split into [HEM] and [LSH] submodes
# 'pubo_direct'            : Full PUBO-based objective directly on hypergraph (Auto Grad + Opt)
# 'pubo_coarsen'           : Coarsening framework + PUBO on the compressed hyperedges
# 'pubo_q4_explicit'       : Coarsening + explicit formulation via expected_hyperbmincut_explicit
# 'pubo_implicit'          : Coarsening + approximate formulation via expected_hyperbmincut
# ==========================================
partition_runs = [
    # 'direct_fem',
    'coarsen_fem_refine_kahypar[fem_as_greedy_init]',
    'coarsen_kahypar_refine',
    'kahyper_like[HEM]',
    # 'pubo_direct',
    # 'pubo_coarsen',
    # 'pubo_q4_explicit',
    # 'pubo_implicit',
]

verbose = True
use_lsh = False
coarsen_to = 100


# ── Solver instances (shared across methods) ─────────────────────────────
# KahyparLikeSolver does coarsening + greedy initial partition.
# FemCoarsenSolver does FEM-based initial partition (no coarsening).

_kahypar_solver = KahyparLikeSolver()
_kahypar_solver.update_params(coarsen_to=coarsen_to, verbose=verbose, use_lsh=use_lsh)

_fem_solver = FemCoarsenSolver()
_fem_solver.update_params(
    num_trials=num_trials, num_steps=num_steps, dev=dev,
)

_kahypar_refine_solver = HyperRefineSolver()
_kahypar_refine_solver.update_params(mode_cycle=('flow', 'mcts', 'evolution'), rounds=3)

_fem_refine_solver = HyperRefineSolver()
_fem_refine_solver.update_params(mode_cycle=('flow',), rounds=1)


def _build_pubo_result(pubo_obj, num_coarse_nodes, q_ways):
    """Helper: run FEM with a custom PUBO objective and return the assignment."""
    dummy_matrix = torch.zeros((num_coarse_nodes, num_coarse_nodes))
    case = FEM()
    case.set_up_problem(
        num_coarse_nodes, 0, 'customize', dummy_matrix, q=q_ways,
        customize_expected_func=pubo_obj.expectation,
        customize_infer_func=pubo_obj.inference,
    )
    case.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)
    configs, results = case.solve()
    return configs[0].argmax(dim=1).cpu().numpy()


def run_partition_method(run_label, hyperedges, clique_graph, num_nodes, q_ways, verbose=False):
    partition_method, submode = _parse_run_label(run_label)
    display_label = run_label
    fem_mode = submode or 'fem_as_hem'
    use_lsh_flag = submode == 'LSH'
    start_time = time.time()
    log = lambda msg: _log(msg, verbose)

    log(f"Loading {instance}...")
    log(f"====== Running {display_label} ======")

    # ── Direct FEM on clique-expanded graph ───────────────────────────────
    if partition_method == 'direct_fem':
        graph_for_fem = clique_graph
        node_weights_for_fem = torch.ones(num_nodes, dtype=torch.float32)

        log("Setting up FEM solver...")
        case_bmincut = FEM.from_couplings(
            'bmincut',
            graph_for_fem.shape[0],
            int(clique_graph._nnz() // 2),
            graph_for_fem,
            node_weights=node_weights_for_fem,
        )
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=True)

        log("Running FEM optimize...")
        config, result = case_bmincut.solve()
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
        best_config = config[optimal_inds[0]]
        final_assignment = best_config.argmax(dim=1).cpu().numpy()

    # ── Direct PUBO on hypergraph ─────────────────────────────────────────
    elif partition_method == 'pubo_direct':
        pubo_obj = PUBOObjective(
            hyperedges,
            [1.0] * len(hyperedges),
            q=q_ways,
            num_nodes=num_nodes,
            node_weights=torch.ones(num_nodes, dtype=torch.float32),
            imbalance_weight=5.0,
            obj_type='cut_net',
            max_degree=5,
        )

        dummy_matrix = torch.zeros((num_nodes, num_nodes))
        case_bmincut = FEM()
        case_bmincut.set_up_problem(
            num_nodes,
            0,
            'customize',
            dummy_matrix,
            q=q_ways,
            customize_expected_func=pubo_obj.expectation,
            customize_infer_func=pubo_obj.inference,
        )
        case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=q_ways, manual_grad=False)

        log("Running PUBO FEM optimize...")
        config, result = case_bmincut.solve()
        best_config = config[2] if len(config) > 2 else config[0]
        final_assignment = best_config.argmax(dim=1).cpu().numpy()

    # ── Multi-level methods (coarsen → initial partition → refine) ────────
    elif partition_method in ['coarsen_fem_refine_kahypar', 'coarsen_kahypar_refine',
                               'kahyper_like', 'pubo_coarsen',
                               'pubo_q4_explicit', 'pubo_implicit']:

        # --- Phase 1: Coarsen via solver (same coarse result for all methods) ---
        _kahypar_solver.set_param('use_lsh', use_lsh_flag)
        res = _kahypar_solver.coarsen(hyperedges, num_nodes, q_ways)
        coarse_hyperedges = res['coarse_hyperedges']
        coarse_node_weights = res['coarse_node_weights']
        coarse_groups = res['coarse_groups']
        original_to_coarse = res['original_to_coarse']
        num_coarse_nodes = len(coarse_groups)
        log(f"Coarsening took: {time.time() - start_time:.4f}s, "
            f"coarse_nodes={num_coarse_nodes}")

        # --- Phase 2: Initial partition on coarse graph ---
        if partition_method in ['coarsen_kahypar_refine', 'kahyper_like']:
            initial_assignment = _kahypar_solver.initial_partition_greedy(
                coarse_hyperedges, coarse_node_weights, q_ways,
            )

        elif partition_method == 'coarsen_fem_refine_kahypar':
            initial_assignment = _fem_solver.initial_partition(
                coarse_hyperedges, coarse_node_weights, q_ways,
            )

        else:  # pubo_coarsen, pubo_q4_explicit, pubo_implicit
            coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = (
                coarsen_graph_by_matching(
                    clique_graph,
                    node_weights=torch.ones(num_nodes, dtype=torch.float32),
                    coarsen_to=coarsen_to,
                )
            )
            num_coarse_nodes = coarse_graph.shape[0]
            coarse_hyperedges = build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes)
            initial_assignment = None  # will be filled by PUBO below

        use_kahypar_refine = False
        if partition_method == 'coarsen_kahypar_refine' and HAS_KAHYPAR:
            log("Using KaHyPar for refinement (greedy initial assignment).")
            use_kahypar_refine = True
        elif partition_method == 'coarsen_kahypar_refine' and not HAS_KAHYPAR:
            log("KaHyPar not installed. Falling back to FEM refinement.")
            partition_method = 'coarsen_fem_refine_kahypar'

        if partition_method in ('pubo_coarsen', 'pubo_q4_explicit', 'pubo_implicit'):
            if partition_method == 'pubo_coarsen':
                log("Using PUBO as the primary solver on the coarsened graph...")
                pubo_obj = PUBOObjective(
                    coarse_hyperedges, [1.0] * len(coarse_hyperedges),
                    q=q_ways, num_nodes=num_coarse_nodes,
                    node_weights=coarse_node_weights, imbalance_weight=5.0,
                    obj_type='cut_net', max_degree=5,
                )
            elif partition_method == 'pubo_q4_explicit':
                log("Using Explicit q=4 PUBO on the coarsened graph...")
                from src.fem.customized_problem.hyper_bmincut import expected_hyperbmincut_explicit
                pubo_obj = make_q4_pubo_object(
                    coarse_hyperedges, coarse_node_weights,
                    expected_hyperbmincut_explicit, num_coarse_nodes, q_ways,
                )
            else:  # pubo_implicit
                log("Using implicit q=4 PUBO on the coarsened graph...")
                from src.fem.customized_problem.hyper_bmincut import expected_hyperbmincut
                pubo_obj = make_q4_pubo_object(
                    coarse_hyperedges, coarse_node_weights,
                    expected_hyperbmincut, num_coarse_nodes, q_ways,
                )

            initial_assignment = _build_pubo_result(pubo_obj, num_coarse_nodes, q_ways)
            log(f"PUBO partitioning took: {time.time() - start_time:.4f} seconds")

        # --- Phase 3: Uncoarsen ---
        log("Step 3: Uncoarsening (Projection) back to original hypergraph...")
        step3_t0 = time.time()
        group_assignment = expand_coarse_labels(coarse_groups, initial_assignment, num_nodes)
        log(f"Step 3: expand_coarse_labels finished in {time.time() - step3_t0:.4f}s")

        # --- Phase 4: Refine ---
        if partition_method in ('coarsen_kahypar_refine', 'coarsen_fem_refine_kahypar') and HAS_KAHYPAR and use_kahypar_refine:
            log("Step 3: Running KaHyPar refinement on the original hypergraph...")
            hyperedges_indices = []
            hyperedges_ptrs = [0]
            for he in hyperedges:
                hyperedges_indices.extend(he)
                hyperedges_ptrs.append(len(hyperedges_indices))

            hg = kahypar.Hypergraph(num_nodes, len(hyperedges), hyperedges_indices,
                                     hyperedges_ptrs, q_ways,
                                     [1] * len(hyperedges), [1] * num_nodes)
            for i in range(num_nodes):
                hg.setNodePart(i, int(group_assignment[i]))

            ctx = kahypar.Context()
            try:
                ctx.loadINIconfiguration('kahypar_config.ini')
            except Exception:
                pass
            ctx.setK(q_ways)
            ctx.setEpsilon(0.05)
            kahypar.improvePartition(hg, ctx)
            final_assignment = np.array([hg.blockID(i) for i in range(num_nodes)], dtype=np.int64)
        else:
            log("Step 3: Running Hybrid Refinement...")
            step3_refine_t0 = time.time()
            if partition_method == 'kahyper_like':
                final_assignment = _kahypar_refine_solver.refine(
                    group_assignment, hyperedges, q_ways, verbose=verbose,
                )
            elif partition_method == 'coarsen_fem_refine_kahypar':
                final_assignment = _fem_refine_solver.refine(
                    group_assignment, hyperedges, q_ways, verbose=verbose,
                    flow_passes=2, skip_exploration_if_good=True,
                )
            else:
                final_assignment = greedy_refine_hypergraph_incremental(
                    group_assignment, hyperedges,
                    [1.0] * len(hyperedges), q=q_ways,
                    max_passes=5, max_imbalance=0.05,
                )
            log(f"Step 3: refinement finished in {time.time() - step3_refine_t0:.4f}s")
    else:
        raise ValueError(f"Unknown partition method: {partition_method}")

    cut_value, max_imbalance = _compute_summary(final_assignment, hyperedges, q_ways)
    elapsed = time.time() - start_time
    return {
        'instance': instance,
        'method': display_label,
        'time_s': elapsed,
        'cut': cut_value,
        'imbalance': max_imbalance,
        'assignment': final_assignment,
    }


hyperedges = parse_hypergraph_edges(instance)
num_nodes = max((max(hyperedge) for hyperedge in hyperedges if hyperedge), default=-1) + 1
clique_graph = build_clique_expanded_graph(hyperedges, num_nodes=num_nodes, normalize_weight=True)

q_ways = 4

_print_results_table([])
for run_label in partition_runs:
    row = run_partition_method(run_label, hyperedges, clique_graph, num_nodes, q_ways, verbose=verbose)
    _print_result_row(row)
