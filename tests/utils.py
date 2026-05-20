
import numpy as np
import torch
from itertools import combinations

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
        # print(f"Parsed {len(hyperedges)} hyperedges from {instance_path}")
        return hyperedges
    except Exception as e:
        print(f"Error parsing hypergraph: {e}")
        return []

def evaluate_cut_value(assignment: np.ndarray, hyperedges: list) -> int:
    cut_count = 0
    for hyperedge in hyperedges:
        groups_in_hyperedge = set()
        for vertex in hyperedge:
            if vertex < len(assignment):
                groups_in_hyperedge.add(assignment[vertex])
        

        if len(groups_in_hyperedge) > 1:
            cut_count += 1
    
    return cut_count

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
    avg_imbalance = float(np.mean(imbalance_per_group))
    return total_cut_value, avg_imbalance


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
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (num_nodes, num_nodes)).coalesce()

    indices = torch.tensor([rows, cols], dtype=torch.long)
    weights = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, weights, (num_nodes, num_nodes)).coalesce()


def _sparse_to_adjacency_dict(J: torch.Tensor):
    J = J.coalesce()
    n = J.shape[0]
    adjacency = [dict() for _ in range(n)]
    indices = J.indices()
    values = J.values()
    for idx in range(values.numel()):
        row = int(indices[0, idx])
        col = int(indices[1, idx])
        if row == col:
            continue
        adjacency[row][col] = adjacency[row].get(col, 0.0) + float(values[idx].item())
    return adjacency


def coarsen_graph_by_leaf_folding(J: torch.Tensor, node_weights=None, max_degree: int = 500, min_nodes: int = 5000, max_rounds: int = 1000):
    if not J.is_sparse:
        J = J.to_sparse()

    adjacency = _sparse_to_adjacency_dict(J)
    n = J.shape[0]
    active_nodes = set(range(n))
    groups = {node: [node] for node in range(n)}
    if node_weights is None:
        weights = {node: 1.0 for node in range(n)}
    else:
        weights = {node: float(node_weights[node]) for node in range(n)}

    def merge_leaf(leaf, neighbor):
        if leaf not in active_nodes or neighbor not in active_nodes:
            return False
        leaf_neighbors = list(adjacency[leaf].items())
        if len(leaf_neighbors) != 1:
            return False

        for other, edge_weight in leaf_neighbors:
            if other == neighbor:
                continue
            adjacency[neighbor][other] = adjacency[neighbor].get(other, 0.0) + edge_weight
            adjacency[other][neighbor] = adjacency[other].get(neighbor, 0.0) + edge_weight
            adjacency[other].pop(leaf, None)

        adjacency[neighbor].pop(leaf, None)
        adjacency[leaf].clear()
        active_nodes.remove(leaf)
        weights[neighbor] += weights[leaf]
        groups[neighbor].extend(groups[leaf])
        groups.pop(leaf, None)
        weights.pop(leaf, None)
        return True

    for _ in range(max_rounds):
        current_max_degree = max((len(adjacency[node]) for node in active_nodes), default=0)
        if current_max_degree < max_degree or len(active_nodes) <= min_nodes:
            break

        leaves = [node for node in list(active_nodes) if len(adjacency[node]) == 1]
        if not leaves:
            break

        changed = False
        for leaf in leaves:
            if leaf not in active_nodes or len(adjacency[leaf]) != 1:
                continue
            neighbor = next(iter(adjacency[leaf]))
            changed |= merge_leaf(leaf, neighbor)
        if not changed:
            break

    survivors = sorted(active_nodes)
    remap = {old: new for new, old in enumerate(survivors)}

    rows = []
    cols = []
    values = []
    coarse_groups = []
    coarse_weights = []
    original_to_coarse = np.empty(n, dtype=np.int64)

    for old_node in survivors:
        new_node = remap[old_node]
        coarse_groups.append(groups[old_node])
        coarse_weights.append(weights[old_node])
        for original_node in groups[old_node]:
            original_to_coarse[original_node] = new_node

        for neighbor, edge_weight in adjacency[old_node].items():
            if neighbor in remap:
                rows.append(new_node)
                cols.append(remap[neighbor])
                values.append(edge_weight)

    if rows:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        weights_tensor = torch.tensor(values, dtype=torch.float32)
        coarse_J = torch.sparse_coo_tensor(indices, weights_tensor, (len(survivors), len(survivors))).coalesce()
    else:
        coarse_J = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (len(survivors), len(survivors))).coalesce()

    coarse_node_weights = torch.tensor(coarse_weights, dtype=torch.float32)
    return coarse_J, coarse_node_weights, coarse_groups, original_to_coarse


def expand_coarse_labels(coarse_groups: list, coarse_labels: np.ndarray, num_nodes: int):
    labels = np.empty(num_nodes, dtype=np.int64)
    for coarse_node, members in enumerate(coarse_groups):
        for member in members:
            labels[member] = coarse_labels[coarse_node]
    return labels