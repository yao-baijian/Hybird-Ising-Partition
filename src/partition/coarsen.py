"""Normal-graph coarsening functions.

Provides matching-based graph coarsening for standard (non-hyper) graphs.
"""

import numpy as np
import torch


def _sparse_coo_tensor_no_check(indices, values, size):
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, values, size)


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


def coarsen_graph_by_matching(J: torch.Tensor, node_weights=None, max_node_weight=None,
                               coarsen_to: int = 500, max_rounds: int = 20,
                               verbose: bool = False):
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    
    groups = [[node] for node in range(n)]
    if node_weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.array(node_weights, dtype=np.float32)
        
    if max_node_weight is None:
        max_node_weight = max(weights.sum() / max(coarsen_to, 1), np.max(weights) * 2)
        
    current_J = J
    current_n = n
    current_weights = weights
    
    coarsen_rounds = 0
    for _ in range(max_rounds):
        if current_n <= coarsen_to:
            break
        coarsen_rounds += 1
        
        adjacency = _sparse_to_adjacency_dict(current_J)
        
        matched = np.zeros(current_n, dtype=bool)
        remap = np.full(current_n, -1, dtype=np.int64)
        new_n = 0
        
        visit_order = np.random.permutation(current_n)
        
        new_groups = []
        new_weights = []
        
        for u in visit_order:
            if matched[u]:
                continue
            matched[u] = True
            
            best_v = -1
            best_w = -1.0
            for v, edge_w in adjacency[u].items():
                if not matched[v] and current_weights[u] + current_weights[v] <= max_node_weight:
                    if edge_w > best_w:
                        best_w = edge_w
                        best_v = v
                        
            if best_v != -1:
                matched[best_v] = True
                remap[u] = new_n
                remap[best_v] = new_n
                new_groups.append(groups[u] + groups[best_v])
                new_weights.append(current_weights[u] + current_weights[best_v])
            else:
                remap[u] = new_n
                new_groups.append(groups[u])
                new_weights.append(current_weights[u])
                
            new_n += 1
            
        if new_n == current_n:
            break
            
        indices = current_J.indices()
        values = current_J.values()
        
        coarse_rows = remap[indices[0].cpu().numpy()]
        coarse_cols = remap[indices[1].cpu().numpy()]
        
        valid = coarse_rows != coarse_cols
        
        if np.any(valid):
            coarse_indices = torch.tensor(np.stack([coarse_rows[valid], coarse_cols[valid]]),
                                          dtype=torch.long, device=values.device)
            coarse_values = values[torch.from_numpy(valid)]
            current_J = _sparse_coo_tensor_no_check(coarse_indices, coarse_values, (new_n, new_n)).coalesce()
        else:
            current_J = _sparse_coo_tensor_no_check(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
                (new_n, new_n),
            ).coalesce()
            
        current_n = new_n
        current_weights = np.array(new_weights, dtype=np.float32)
        groups = new_groups
        
    coarse_node_weights = torch.tensor(current_weights, dtype=torch.float32)
    
    if verbose:
        print(f"[Coarsen] {n} -> {current_n} nodes in {coarsen_rounds} rounds")
    
    original_to_coarse = np.empty(n, dtype=np.int64)
    for c_node, members in enumerate(groups):
        for member in members:
            original_to_coarse[member] = c_node
            
    return current_J, coarse_node_weights, groups, original_to_coarse, coarsen_rounds


def expand_coarse_labels(coarse_groups: list, coarse_labels: np.ndarray, num_nodes: int):
    labels = np.empty(num_nodes, dtype=np.int64)
    for coarse_node, members in enumerate(coarse_groups):
        for member in members:
            labels[member] = coarse_labels[coarse_node]
    return labels
