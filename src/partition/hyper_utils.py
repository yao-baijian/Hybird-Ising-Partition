"""Utility functions for hypergraph partitioning (coarsening and refinement).

These functions were extracted from tests/utils.py and moved here to avoid
a reverse dependency (src/partition importing from tests/).
"""

import numpy as np
import torch
from itertools import combinations


def _sparse_coo_tensor_no_check(indices, values, size):
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, values, size)


def evaluate_kahypar_cut_value(assignment: np.ndarray, hyperedges: list, hyperedge_weights: list = None) -> float:
    """
    sum_{e in cut} (λ(e) - 1) * w(e)
    """
    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)
    
    total_cut_value = 0
    
    for hyperedge, weight in zip(hyperedges, hyperedge_weights):
        groups_in_hyperedge = set()
        if len(hyperedge) > 1:
            for vertex in hyperedge:
                groups_in_hyperedge.add(assignment[vertex])
        lambda_e = len(groups_in_hyperedge)
        if lambda_e > 1:
            total_cut_value += (lambda_e - 1) * weight
    
    arr = np.asarray(assignment, dtype=int)
    q = int(arr.max()) + 1
    counts = np.bincount(arr, minlength=q)
    ideal = arr.size / float(q)
    imbalance_per_group = np.abs(counts - ideal) / ideal
    max_imbalance = float(np.max(imbalance_per_group))
    return total_cut_value, max_imbalance


def build_clique_expanded_graph(hyperedges: list, num_nodes: int = None, normalize_weight: bool = True):
    if num_nodes is None:
        num_nodes = max((max(hyperedge) for hyperedge in hyperedges if hyperedge), default=-1) + 1

    rows = []
    cols = []
    values = []

    for hyperedge in hyperedges:
        if len(hyperedge) < 2:
            continue
        edge_weight = 1.0 / (len(hyperedge) - 1) if normalize_weight else 1.0
        for u, v in combinations(hyperedge, 2):
            rows.extend([u, v])
            cols.extend([v, u])
            values.extend([edge_weight, edge_weight])

    if not rows:
        return _sparse_coo_tensor_no_check(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            (num_nodes, num_nodes),
        ).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.long)
    weights = torch.tensor(values, dtype=torch.float32)
    return _sparse_coo_tensor_no_check(indices, weights, (num_nodes, num_nodes)).coalesce()


def greedy_initial_hypergraph_partition(
    hyperedges: list,
    vertex_weights,
    k: int,
    hyperedge_weights: list = None,
    epsilon: float = 0.03,
    seed: int = None,
):
    """
    Build a balanced initial k-way partition for a hypergraph using a greedy
    vertex placement heuristic that respects vertex weights.
    """
    rng = np.random.default_rng(seed)
    vertex_weights = np.asarray(vertex_weights, dtype=float)
    num_nodes = int(vertex_weights.shape[0])
    k = int(k)
    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)

    node_to_he = [[] for _ in range(num_nodes)]
    node_degree = np.zeros(num_nodes, dtype=float)
    for e_idx, he in enumerate(hyperedges):
        w = float(hyperedge_weights[e_idx])
        for v in he:
            if 0 <= v < num_nodes:
                node_to_he[v].append(e_idx)
                node_degree[v] += w

    order = np.arange(num_nodes)
    tie_breaker = rng.random(num_nodes)
    order = np.lexsort((tie_breaker, -vertex_weights, -node_degree))

    assignment = np.full(num_nodes, -1, dtype=np.int64)
    group_weights = np.zeros(k, dtype=float)
    total_weight = float(vertex_weights.sum())
    ideal_weight = total_weight / float(k) if k > 0 else 0.0
    max_weight = ideal_weight * (1.0 + float(epsilon))
    if max_weight <= 0.0:
        max_weight = float("inf")

    def boundary_cost(v, g):
        cost = 0.0
        for e_idx in node_to_he[v]:
            he = hyperedges[e_idx]
            w = float(hyperedge_weights[e_idx])
            pins = 0
            same = 0
            for u in he:
                au = assignment[u]
                if au != -1:
                    pins += 1
                    if au == g:
                        same += 1
            if same == 0:
                cost += w
            elif same == pins:
                cost -= w
        return cost

    for v in order:
        best_group = None
        best_cost = None
        candidates = np.arange(k)
        rng.shuffle(candidates)
        for g in candidates:
            if group_weights[g] + float(vertex_weights[v]) > max_weight:
                continue
            cost = boundary_cost(v, g)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_group = g

        if best_group is None:
            best_group = int(np.argmin(group_weights))

        assignment[v] = best_group
        group_weights[best_group] += float(vertex_weights[v])

    return assignment


def greedy_refine_hypergraph_incremental(
    assignment: np.ndarray,
    hyperedges: list,
    hyperedge_weights: list,
    q: int,
    max_passes: int = 5,
    max_imbalance: float = 0.05,
):
    """
    Incremental local refinement that only re-evaluates affected vertices
    (the moved vertex and its L1 hypergraph neighbors).
    """
    assignment = assignment.copy()
    num_nodes = len(assignment)

    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)

    he_pins = [np.zeros(q, dtype=np.int32) for _ in range(len(hyperedges))]
    node_to_he = [[] for _ in range(num_nodes)]
    vertex_neighbors = [set() for _ in range(num_nodes)]

    for e_idx, he in enumerate(hyperedges):
        for v in he:
            if v < num_nodes:
                he_pins[e_idx][assignment[v]] += 1
                node_to_he[v].append(e_idx)
        for u in he:
            if u < num_nodes:
                for v in he:
                    if u != v and v < num_nodes:
                        vertex_neighbors[u].add(v)

    group_sizes = np.bincount(assignment, minlength=q)
    ideal_size = num_nodes / float(q)
    max_size = ideal_size * (1.0 + max_imbalance)

    active = np.zeros(num_nodes, dtype=bool)
    queue = list(np.where(np.ones(num_nodes, dtype=bool))[0])

    def move_gain(v, new_group):
        old_group = assignment[v]
        if new_group == old_group:
            return 0.0
        gain = 0.0
        for e_idx in node_to_he[v]:
            pins = he_pins[e_idx]
            weight = hyperedge_weights[e_idx]
            if pins[old_group] == 1:
                gain += weight
            if pins[new_group] == 0:
                gain -= weight
        return gain

    for _pass in range(max_passes):
        moved_any = False
        while queue:
            v = queue.pop()
            active[v] = False

            old_group = assignment[v]
            best_gain = 0.0
            best_group = old_group

            for new_group in range(q):
                if new_group == old_group:
                    continue
                if group_sizes[new_group] + 1 > max_size:
                    continue
                gain = move_gain(v, new_group)
                if gain > best_gain:
                    best_gain = gain
                    best_group = new_group

            if best_group != old_group:
                assignment[v] = best_group
                group_sizes[old_group] -= 1
                group_sizes[best_group] += 1

                for e_idx in node_to_he[v]:
                    he_pins[e_idx][old_group] -= 1
                    he_pins[e_idx][best_group] += 1

                moved_any = True

                affected = set(vertex_neighbors[v])
                affected.add(v)
                for u in affected:
                    if not active[u]:
                        queue.append(u)
                        active[u] = True

        if not moved_any:
            break

        frontier = set()
        for v in range(num_nodes):
            for e_idx in node_to_he[v]:
                pins = he_pins[e_idx]
                if pins[assignment[v]] == 1:
                    frontier.add(v)
                    frontier.update(vertex_neighbors[v])
                    break
        queue = list(frontier)
        for v in queue:
            active[v] = True

    return assignment
