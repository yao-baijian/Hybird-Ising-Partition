"""
Test utility module.

Functions that are also used by partition/ hypergraph code live in
src/partition/hyper_utils.py; normal-graph coarsening/refinement functions
live in src/partition/coarsen.py and src/partition/refine.py.
This module re-exports them for backward compatibility during the
transition, plus provides test-specific utilities.
"""

import time

import numpy as np
import torch

# ── Re-exports from src/partition (backward-compatible aliases) ────────────

from src.partition.coarsen import (
    coarsen_graph_by_matching,
    expand_coarse_labels,
)

from src.partition.refine import (
    simple_kaffpa,
    call_pymetis_with_part,
)

from src.partition.hyper_utils import (
    build_clique_expanded_graph,
    evaluate_kahypar_cut_value,
    greedy_initial_hypergraph_partition,
    greedy_refine_hypergraph_incremental,
)

# ── Test-only utilities (not moved to src/partition) ──────────────────────


def parse_hypergraph_edges(instance_path: str) -> list:
    hyperedges = []
    try:
        with open(instance_path, 'r') as f:
            f.readline()
            for line in f:
                if line.strip():
                    vertices = [int(v) - 1 for v in line.split() if v.strip()]
                    if len(vertices) > 1:
                        hyperedges.append(vertices)
        return hyperedges
    except Exception as e:
        print(f"Error parsing hypergraph: {e}")
        return []


class PUBOObjective:
    """Wraps cut functions for PUBO solvers."""

    def __init__(self, hyperedges, node_weights, cut_func, num_nodes, q, imbalance_weight=1.0):
        self.hyperedges = hyperedges
        self.node_weights = node_weights
        self.cut_func = cut_func
        self.num_nodes = num_nodes
        self.q = q
        self.imbalance_weight = imbalance_weight

    def evaluate(self, assignment):
        cut = self.cut_func(assignment, self.hyperedges)
        counts = np.bincount(assignment, minlength=self.q)
        ideal = self.num_nodes / float(self.q)
        imbalance = np.max(np.abs(counts - ideal) / ideal)
        return cut + self.imbalance_weight * imbalance

    def expected_cut_and_imbalance(self, assignment):
        cut = self.cut_func(assignment, self.hyperedges)
        counts = np.bincount(assignment, minlength=self.q)
        ideal = self.num_nodes / float(self.q)
        imbalance = np.max(np.abs(counts - ideal) / ideal)
        return cut, imbalance


# ── Test display helpers ──────────────────────────────────────────────────


def _log(message, enabled=False):
    if enabled:
        print(message)


def _print_results_table(rows):
    col_w = (30, 28, 10, 12, 10)
    header_fmt = f"{{:<{col_w[0]}}} {{:<{col_w[1]}}} {{:>{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}}"
    sep = ' '.join(['-' * w for w in col_w])
    print(header_fmt.format('instance', 'method', 'time(s)', 'cut', 'imbalance'))
    print(sep)
    for row in rows:
        print(
            header_fmt.format(
                row['instance'],
                row['method'],
                f"{row['time_s']:.4f}",
                f"{row['cut']:.4f}",
                f"{row['imbalance']:.6f}",
            )
        )


def _print_result_row(row):
    col_w = (30, 28, 10, 12, 10)
    header_fmt = f"{{:<{col_w[0]}}} {{:<{col_w[1]}}} {{:>{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}}"
    print(
        header_fmt.format(
            row['instance'],
            row['method'],
            f"{row['time_s']:.4f}",
            f"{row['cut']:.4f}",
            f"{row['imbalance']:.6f}",
        ),
        flush=True,
    )


# ── Coarsening & partition helpers ────────────────────────────────────────


def run_kahypar_like_multilevel(
    clique_graph_local,
    hyperedges_local,
    num_nodes_local,
    q_local,
    coarsen_to=500,
    verbose=False,
    use_lsh=False,
):
    """Run KaHyPar-like multilevel coarsening with optional LSH preprocessing."""
    from src.partition import coarsen_kahypar_like as shared_kahypar_like_coarsen

    stage_t0 = time.time()
    res = shared_kahypar_like_coarsen(
        hyperedges_local,
        num_nodes_local,
        q=q_local,
        coarsen_to=max(10, int(coarsen_to)),
        verbose=verbose,
        use_lsh=use_lsh,
    )
    _log(
        (
            f"[kahyper_like] shared_kahypar_like_coarsen: "
            f"n={num_nodes_local} -> {len(res['coarse_groups'])}, "
            f"nnz={int(res['coarse_graph']._nnz()) if res['coarse_graph'].is_sparse else 0}, "
            f"time={time.time() - stage_t0:.4f}s"
        ),
        verbose,
    )

    return (
        res['coarse_graph'],
        res['coarse_node_weights'],
        res['coarse_groups'],
        res['original_to_coarse'],
        res['initial_assignment'],
    )


def _compute_summary(final_assignment, hyperedges, q_ways):
    fem_cut_value, _ = evaluate_kahypar_cut_value(final_assignment, hyperedges, [1.0] * len(hyperedges))
    counts = np.bincount(final_assignment, minlength=q_ways)
    ideal = len(final_assignment) / q_ways
    max_imbalance = float(np.max(np.abs(counts - ideal) / ideal)) if ideal > 0 else 0.0
    return float(fem_cut_value), max_imbalance


def _parse_run_label(run_label):
    if '[' not in run_label or not run_label.endswith(']'):
        return run_label, None
    method_name, submode = run_label[:-1].split('[', 1)
    return method_name, submode


