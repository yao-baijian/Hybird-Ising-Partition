
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

class PUBOObjective:
    def __init__(self, hyperedges, hyperedge_weights, q, num_nodes, node_weights, imbalance_weight=5.0, obj_type='cut_net', max_degree=5):
        import torch
        from FEM.problem import weighted_imbalance_penalty
        self.groups = {}
        for size in range(2, max_degree + 1):
            self.groups[size] = {'indices': [], 'weights': []}
            
        large_he = []
        large_weights = []
        # Calculate degrees
        node_degrees = np.ones(num_nodes, dtype=np.float32) # Add 1 to avoid div by zero
        for he, w in zip(hyperedges, hyperedge_weights):
            for v in he:
                if v < num_nodes:
                    node_degrees[v] += w

            if len(he) <= max_degree:
                self.groups[len(he)]['indices'].append(he)
                self.groups[len(he)]['weights'].append(w)
            else:
                large_he.append(he)
                large_weights.append(w)
                
        self.tensors_by_size = {}
        for size, data in self.groups.items():
            if data['indices']:
                self.tensors_by_size[size] = {
                    'idx': torch.tensor(data['indices'], dtype=torch.long),
                    'weight': torch.tensor(data['weights'], dtype=torch.float32)
                }
                
        if large_he:
            clique_J = build_clique_expanded_graph(large_he, num_nodes=num_nodes, normalize_weight=True)
            self.clique_J = clique_J.to_dense()
        else:
            self.clique_J = None
            
        self.node_weights = torch.tensor(node_weights, dtype=torch.float32) if node_weights is not None else torch.ones(num_nodes)
        self.node_degrees = torch.tensor(node_degrees, dtype=torch.float32)
        self.imbalance_weight = imbalance_weight
        self.obj_type = obj_type
        self.weighted_imbalance_penalty = weighted_imbalance_penalty
        self.q = q
        
    def to(self, dev):
        for size in self.tensors_by_size:
            self.tensors_by_size[size]['idx'] = self.tensors_by_size[size]['idx'].to(dev)
            self.tensors_by_size[size]['weight'] = self.tensors_by_size[size]['weight'].to(dev)
        if self.clique_J is not None:
            self.clique_J = self.clique_J.to(dev)
        self.node_weights = self.node_weights.to(dev)
        self.node_degrees = self.node_degrees.to(dev)

    def expectation(self, _, p):
        # Optional: gradient scaling & clipping hook on p
        if p.requires_grad and not hasattr(p, 'pubo_hook_registered'):
            def scale_and_clip(grad):
                # Node degree normalization
                g = grad / self.node_degrees.view(1, -1, 1)
                # Gradient clipping
                g = torch.clamp(g, -5.0, 5.0)
                return g
            p.register_hook(scale_and_clip)
            p.pubo_hook_registered = True

        dev = p.device
        self.to(dev)
        
        loss = 0.0
        
        for size, t in self.tensors_by_size.items():
            idx = t['idx'] 
            weight = t['weight'] 
            
            p_e = p[:, idx, :] 
            
            if self.obj_type == 'cut_net':
                prod = p_e.prod(dim=2) 
                sum_prod = prod.sum(dim=2) 
                term = weight * (1.0 - sum_prod)
                loss = loss + term.sum(dim=1)
            elif self.obj_type == 'km1':
                prod = (1.0 - p_e).prod(dim=2) 
                sum_term = (1.0 - prod).sum(dim=2) 
                term = weight * (sum_term - 1.0)
                loss = loss + term.sum(dim=1)
                
        if self.clique_J is not None:
            clique_loss = ((self.clique_J @ p) * (1 - p)).sum(dim=(1, 2))
            loss = loss + clique_loss
            
        imb_penalty = self.weighted_imbalance_penalty(p, self.node_weights.cpu().numpy())
        loss = loss + self.imbalance_weight * imb_penalty
        
        return loss

    def inference(self, _, p):
        import torch
        # Dummy result since we recalculate cut with `evaluate_kahypar_cut_value` anyway. 
        # But FEM solver needs `config` and `results`.
        config = torch.zeros_like(p)
        config.scatter_(2, p.argmax(dim=2, keepdim=True), 1)
        # return dummy low objective values to allow FEM to just pick the best config based on argmax.
        dummy_results = torch.zeros(p.shape[0], device=p.device)
        return config, dummy_results

def greedy_refine_hypergraph(
    assignment: np.ndarray, 
    hyperedges: list, 
    hyperedge_weights: list, 
    q: int, 
    max_passes: int = 5,
    max_imbalance: float = 0.05
) -> np.ndarray:
    assignment = assignment.copy()
    num_nodes = len(assignment)
    
    if hyperedge_weights is None:
        hyperedge_weights = [1.0] * len(hyperedges)
        
    he_pins = [np.zeros(q, dtype=np.int32) for _ in range(len(hyperedges))]
    node_to_he = [[] for _ in range(num_nodes)]
    
    for e_idx, he in enumerate(hyperedges):
        for v in he:
            if v < num_nodes:
                he_pins[e_idx][assignment[v]] += 1
                node_to_he[v].append(e_idx)
                
    group_sizes = np.bincount(assignment, minlength=q)
    ideal_size = num_nodes / float(q)
    max_size = ideal_size * (1.0 + max_imbalance)
    
    for pass_idx in range(max_passes):
        moved_any = False
        nodes = np.arange(num_nodes)
        np.random.shuffle(nodes)
        
        for v in nodes:
            old_group = assignment[v]
            
            best_gain = 0.0
            best_group = old_group
            
            for new_group in range(q):
                if new_group == old_group:
                    continue
                    
                if group_sizes[new_group] + 1 > max_size:
                    continue
                    
                gain = 0.0
                for e_idx in node_to_he[v]:
                    pins = he_pins[e_idx]
                    weight = hyperedge_weights[e_idx]
                    
                    if pins[old_group] == 1:
                        gain += weight
                        
                    if pins[new_group] == 0:
                        gain -= weight
                        
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
                
        if not moved_any:
            break
            
    return assignment