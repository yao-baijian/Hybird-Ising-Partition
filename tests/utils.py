"""
Test utility module.

Functions that are also used by partition/ hypergraph code live in
src/partition/hyper_utils.py; normal-graph coarsening/refinement functions
live in src/partition/coarsen.py and src/partition/refine.py.
This module re-exports them for backward compatibility during the
transition, plus provides test-specific utilities.
"""

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


