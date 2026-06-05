"""Cyclic Expansion QUBO Refinement for Graph Partitioning.

This module implements the Cyclic Expansion refinement algorithm adapted
for k-way graph partitioning, using FEM to solve the QUBO subproblems.

The algorithm iteratively:
1. Selects candidate swap pairs (u, v) where u and v are in different partitions
2. Constructs a QUBO where each variable represents whether to swap a pair
3. Solves the QUBO using FEM to find the optimal set of swaps
4. Updates the partition and repeats until convergence

Reference: Algorithm adapted from "2312.15467v1.pdf" (Cyclic Expansion for QUBO)
"""

import numpy as np
import torch
import sys
from typing import List, Tuple, Dict, Optional, Set
from scipy.sparse import csr_matrix
from src.fem import FEM

# Virtual node sentinel encoding:
#   VIRTUAL_SENTINEL = -1  marks a virtual node (the second element).
#   When enable_virtual=True, the target partition for a virtual swap is
#   encoded by storing -(target_part + 2) as the "vertex" value.
#   decode: target_part = -value - 2.
def _encode_virtual_target(target_part: int) -> int:
    """Encode target partition as a negative sentinel value."""
    return -(target_part + 2)

def _decode_virtual_target(value: int) -> int:
    """Decode target partition from sentinel value."""
    return -value - 2

def find_boundary_vertices(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    q: int
) -> List[int]:
    """Find vertices that have at least one neighbor in a different partition.
    
    Args:
        adjacency: adjacency list where adjacency[i] = [(neighbor, weight), ...]
        partition: current partition assignment (length n)
        q: number of partitions
        
    Returns:
        List of boundary vertex indices
    """
    n = len(partition)
    boundary = []
    for i in range(n):
        my_part = partition[i]
        for j, w in adjacency[i]:
            if partition[j] != my_part:
                boundary.append(i)
                break
    return boundary


def compute_external_degree(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    vertex: int
) -> Dict[int, float]:
    """Compute the total edge weight from a vertex to each partition.
    
    Args:
        adjacency: adjacency list
        partition: current partition assignment
        vertex: vertex index
        
    Returns:
        Dict mapping partition_id -> total weight to that partition
    """
    ext_deg = {}
    my_part = partition[vertex]
    for j, w in adjacency[vertex]:
        part_j = partition[j]
        if part_j != my_part:
            ext_deg[part_j] = ext_deg.get(part_j, 0.0) + w
    return ext_deg


def select_candidate_pairs(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    q: int,
    max_candidates: int = 50,
    rng: Optional[np.random.Generator] = None,
    allow_nonadjacent: bool = False,
) -> List[Tuple[int, int]]:
    """Select candidate swap pairs for Cyclic Expansion.
    
    A candidate pair (u, v) satisfies:
    - u and v are in different partitions
    - u and v are connected by an edge (or have high external degree)
    - Each vertex appears in at most one pair
    
    Args:
        adjacency: adjacency list
        partition: current partition assignment
        q: number of partitions
        max_candidates: maximum number of candidate pairs to select
        
    Returns:
        List of (u, v) tuples representing candidate swap pairs
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(partition)
    boundary = find_boundary_vertices(adjacency, partition, q)

    # Randomize the boundary order so repeated runs explore different pairs.
    boundary = list(boundary)
    rng.shuffle(boundary)
    
    # Score each boundary vertex by its external degree
    vertex_scores = []
    for v in boundary:
        ext_deg = compute_external_degree(adjacency, partition, v)
        total_ext = sum(ext_deg.values())
        vertex_scores.append((v, total_ext))
    
    # Sort by external degree (descending)
    vertex_scores.sort(key=lambda x: -x[1])

    # Break ties randomly so equal-score vertices do not always produce
    # the same candidate pairs.
    rng.shuffle(vertex_scores)
    vertex_scores.sort(key=lambda x: -x[1])
    
    # Greedily select disjoint pairs
    used = set()
    pairs = []
    # Precompute boundary vertices by partition for non-adjacent selection
    boundary_by_part: Dict[int, List[int]] = {pid: [] for pid in range(q)}
    for v in boundary:
        boundary_by_part[int(partition[v])].append(v)
    
    for u, score_u in vertex_scores:
        if u in used:
            continue
        if len(pairs) >= max_candidates:
            break
            
        # Find best partner v for u
        best_v = -1
        best_score = -1.0

        neighbors = list(adjacency[u])
        rng.shuffle(neighbors)

        # First consider direct neighbors (prefer adjacent swaps)
        for j, w in neighbors:
            if j in used:
                continue
            if partition[j] == partition[u]:
                continue
            # Score by edge weight and external degree of j
            ext_deg_j = compute_external_degree(adjacency, partition, j)
            score_j = sum(ext_deg_j.values())
            combined_score = w + 0.5 * score_j
            
            if combined_score > best_score:
                best_score = combined_score
                best_v = j
        # If allowed, also consider non-adjacent candidates from other partitions
        if allow_nonadjacent:
            for pid in range(q):
                if pid == int(partition[u]):
                    continue
                # iterate boundary vertices in target partition
                candidates = list(boundary_by_part.get(pid, []))
                rng.shuffle(candidates)
                for j in candidates:
                    if j in used:
                        continue
                    if partition[j] == partition[u]:
                        continue
                    # If j is a neighbor it was already considered above; here w=0
                    w = 0.0
                    ext_deg_j = compute_external_degree(adjacency, partition, j)
                    score_j = sum(ext_deg_j.values())
                    combined_score = w + 0.5 * score_j
                    if combined_score > best_score:
                        best_score = combined_score
                        best_v = j

        if best_v != -1:
            pairs.append((u, best_v))
            used.add(u)
            used.add(best_v)
    
    return pairs


def compute_swap_gain(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    u: int,
    v: int
) -> float:
    """Compute the change in edge cut if we swap partitions of u and v.
    
    Args:
        adjacency: adjacency list
        partition: current partition assignment
        u, v: vertices to swap
        
    Returns:
        Gain (positive = cut decreases, negative = cut increases)
    """
    part_u = partition[u]
    part_v = partition[v]
    
    gain = 0.0
    
    # For u: moving from part_u to part_v
    # Gain = weight to old_part - weight to new_part
    for j, w in adjacency[u]:
        if j == v:
            continue  # Handle u-v edge separately
        part_j = partition[j]
        if part_j == part_u:
            gain -= w  # Lose this internal edge
        elif part_j == part_v:
            gain += w  # Gain this internal edge
    
    # For v: moving from part_v to part_u
    for j, w in adjacency[v]:
        if j == u:
            continue
        part_j = partition[j]
        if part_j == part_v:
            gain -= w
        elif part_j == part_u:
            gain += w
    
    # Handle the edge between u and v if it exists
    for j, w in adjacency[u]:
        if j == v:
            # After swap, u is in part_v and v is in part_u
            # If they were in different parts, they still are (just swapped)
            # So no change for this edge
            break
    
    return gain


def compute_single_vertex_move_gain(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    u: int,
    target_part: int
) -> float:
    """Compute the change in edge cut if we move single vertex u to target_part.
    
    This is used for virtual-node swaps where only one real vertex moves.
    
    Args:
        adjacency: adjacency list
        partition: current partition assignment
        u: vertex to move
        target_part: target partition
        
    Returns:
        Gain (positive = cut decreases, negative = cut increases)
    """
    my_part = partition[u]
    if my_part == target_part:
        return 0.0

    gain = 0.0
    for j, w in adjacency[u]:
        part_j = partition[j]
        if part_j == my_part:
            gain -= w  # Lose this internal edge
        elif part_j == target_part:
            gain += w  # Gain this internal edge
    return gain


def compute_interaction_gain(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    pair1: Tuple[int, int],
    pair2: Tuple[int, int]
) -> float:
    """Compute the interaction gain between two swap pairs.
    
    The interaction gain is the additional cut change when both pairs
    are swapped together, beyond the sum of individual gains.
    
    This occurs when there are edges between vertices of different pairs.
    
    Args:
        adjacency: adjacency list
        partition: current partition assignment
        pair1: (u1, v1) first swap pair
        pair2: (u2, v2) second swap pair
        
    Returns:
        Interaction gain (synergy term for QUBO off-diagonal)
    """
    u1, v1 = pair1
    u2, v2 = pair2
    part_u1, part_v1 = partition[u1], partition[v1]
    part_u2, part_v2 = partition[u2], partition[v2]
    
    interaction = 0.0
    
    # Check all edges between {u1, v1} and {u2, v2}
    pairs_to_check = [
        (u1, u2), (u1, v2),
        (v1, u2), (v1, v2)
    ]
    
    for a, b in pairs_to_check:
        # Find edge weight between a and b
        w_ab = 0.0
        for j, w in adjacency[a]:
            if j == b:
                w_ab = w
                break
        
        if w_ab == 0:
            continue
        
        # Determine if this edge crosses before and after swap
        part_a = partition[a]
        part_b = partition[b]
        
        # Before swap: edge crosses if part_a != part_b
        crosses_before = (part_a != part_b)
        
        # After swap: determine new partitions
        if a == u1:
            new_part_a = part_v1
        elif a == v1:
            new_part_a = part_u1
        elif a == u2:
            new_part_a = part_v2
        elif a == v2:
            new_part_a = part_u2
        else:
            new_part_a = part_a
            
        if b == u1:
            new_part_b = part_v1
        elif b == v1:
            new_part_b = part_u1
        elif b == u2:
            new_part_b = part_v2
        elif b == v2:
            new_part_b = part_u2
        else:
            new_part_b = part_b
            
        crosses_after = (new_part_a != new_part_b)
        
        # Interaction: change in contribution due to both swaps
        # If crosses_before and not crosses_after: gain w_ab
        # If not crosses_before and crosses_after: lose w_ab
        if crosses_before and not crosses_after:
            interaction += w_ab
        elif not crosses_before and crosses_after:
            interaction -= w_ab
    
    return interaction


def build_qubo_matrix(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    candidate_pairs: List[Tuple[int, int]],
    enable_virtual: bool = False,
) -> np.ndarray:
    """Build the QUBO matrix for the Cyclic Expansion swap problem.
    
    The QUBO is: min α^T Q α, where α_t ∈ {0, 1} indicates whether to swap pair t.
    
    Q_{t,t} = gain of swapping pair t alone
    Q_{t1,t2} = interaction gain when both t1 and t2 are swapped
    
    Args:
        adjacency: adjacency list
        partition: current partition assignment
        candidate_pairs: list of (u, v) candidate swap pairs.
            Virtual pairs are encoded as (u, -1) where -1 marks a virtual node,
            and the target partition is encoded via _encode_virtual_target().
        enable_virtual: If True, virtual pairs are present in candidate_pairs
            and their diagonal entries are computed using single-vertex move gain.
        
    Returns:
        QUBO matrix Q of shape (s, s) where s = len(candidate_pairs)
    """
    s = len(candidate_pairs)
    Q = np.zeros((s, s), dtype=float)
    
    def _is_virtual(val: int) -> bool:
        """Check if a vertex value is a virtual sentinel (negative)."""
        return val < 0
    
    # Compute diagonal (individual gains)
    for t, (u, v) in enumerate(candidate_pairs):
        if enable_virtual and (_is_virtual(u) or _is_virtual(v)):
            if _is_virtual(u):
                # (virtual, real_node): real node v moves to virtual's partition
                real_node = v
                target_part = _decode_virtual_target(u)
            else:
                # (real_node, virtual): real node u moves to virtual's partition
                real_node = u
                target_part = _decode_virtual_target(v)
            part_u = int(partition[real_node])
            if target_part == part_u:
                gain = 0.0
            else:
                gain = compute_single_vertex_move_gain(adjacency, partition, real_node, target_part)
        else:
            gain = compute_swap_gain(adjacency, partition, u, v)
        Q[t, t] = gain  # Note: we want to maximize gain, so QUBO minimizes -gain
    
    # Compute off-diagonal (interaction gains) - only between real pairs
    # Virtual pairs have no interaction with others (virtual node has no edges)
    for t1 in range(s):
        u1, v1 = candidate_pairs[t1]
        if enable_virtual and (_is_virtual(u1) or _is_virtual(v1)):
            continue
        for t2 in range(t1 + 1, s):
            u2, v2 = candidate_pairs[t2]
            if enable_virtual and (_is_virtual(u2) or _is_virtual(v2)):
                continue
            interaction = compute_interaction_gain(
                adjacency, partition,
                candidate_pairs[t1], candidate_pairs[t2]
            )
            Q[t1, t2] = interaction
            Q[t2, t1] = interaction
    
    # Negate because we want to maximize gain but QUBO minimizes
    Q = -Q
    
    return Q


def solve_qubo_with_fem(
    Q: np.ndarray,
    num_trials: int = 8,
    num_steps: int = 200,
    dev: str = 'cpu'
) -> np.ndarray:
    """Solve the QUBO problem using FEM.
    
    Args:
        Q: QUBO matrix (s x s)
        num_trials: number of FEM trials
        num_steps: number of FEM steps
        dev: device ('cpu' or 'cuda')
        
    Returns:
        Binary assignment array of length s
    """

    
    s = Q.shape[0]
    if s == 0:
        return np.array([], dtype=int)
    
    # Convert Q to torch tensor
    Q_tensor = torch.tensor(Q, dtype=torch.float32)
    if Q_tensor.is_sparse:
        Q_tensor = Q_tensor.to_dense()
    
    # Create custom expected function for QUBO
    def expected_qubo(_, p: torch.Tensor) -> torch.Tensor:
        """Compute E[α^T Q α] under product distribution p."""
        if p.dim() == 3:
            p1 = p[..., 1]  # (batch, s)
        elif p.dim() == 2:
            if p.shape[1] == 2:
                # (batch*s, 2) -> reshape
                batch = p.shape[0] // s
                p1 = p.reshape(batch, s, 2)[..., 1]
            else:
                p1 = p  # (batch, s)
        elif p.dim() == 1:
            p1 = p.unsqueeze(0)  # (s,) -> (1, s)
        else:
            raise ValueError(f"Unexpected p shape: {p.shape}")
        
        # Compute p1^T Q p1
        Q_t = Q_tensor.to(p1.device)
        left = p1 @ Q_t
        vals = (left * p1).sum(dim=1)
        return vals
    
    def inference_qubo(_, p: torch.Tensor):
        """Convert marginals to discrete configuration."""
        if p.dim() == 3:
            idx = p.argmax(dim=2, keepdim=True)
            config = torch.zeros_like(p)
            config.scatter_(2, idx, 1.0)
        elif p.dim() == 2:
            if p.shape[1] == 2:
                batch = p.shape[0] // s
                p_resh = p.reshape(batch, s, 2)
                idx = p_resh.argmax(dim=2, keepdim=True)
                config = torch.zeros_like(p_resh)
                config.scatter_(2, idx, 1.0)
            else:
                # (batch, s) probability -> threshold
                batch, nloc = p.shape
                config = torch.zeros((batch, nloc, 2), dtype=p.dtype, device=p.device)
                x = (p >= 0.5).long()
                config[..., 1] = x.float()
                config[..., 0] = 1.0 - config[..., 1]
        elif p.dim() == 1:
            p1 = p.unsqueeze(0)
            config = torch.zeros((1, s, 2), dtype=p.dtype, device=p.device)
            x = (p1 >= 0.5).long()
            config[..., 1] = x.float()
            config[..., 0] = 1.0 - config[..., 1]
        else:
            raise ValueError(f"Unexpected p shape: {p.shape}")
        
        return config, torch.zeros(config.shape[0], device=config.device)
    
    # Set up FEM problem
    fem = FEM()
    fem.set_up_problem(
        s, 0, 'customize', Q_tensor,
        customize_expected_func=expected_qubo,
        customize_infer_func=inference_qubo
    )
    fem.set_up_solver(num_trials, num_steps, anneal='lin', dev=dev, q=2, manual_grad=False)
    
    config, result = fem.solve()
    
    # Pick best configuration
    best_idx = int(torch.argmin(result).item())
    best = config[best_idx]
    assignment = best.argmax(dim=1).cpu().numpy().astype(int)
    
    return assignment


def apply_swaps(
    partition: np.ndarray,
    candidate_pairs: List[Tuple[int, int]],
    swap_decision: np.ndarray
) -> np.ndarray:
    """Apply the selected swaps to the partition.
    
    Only real pairs (both u and v are valid vertex indices) are applied.
    Virtual pairs (encoded with sentinel -1) are ignored here and should be
    handled by the caller.
    
    Args:
        partition: current partition assignment
        candidate_pairs: list of (u, v) candidate pairs
        swap_decision: binary array indicating which pairs to swap
        
    Returns:
        Updated partition assignment
    """
    new_partition = partition.copy()
    
    for t, (u, v) in enumerate(candidate_pairs):
        if t >= len(swap_decision):
            break
        if swap_decision[t] != 1:
            continue
        if u < 0 or v < 0:
            # Skip virtual pairs; they are applied separately by the caller
            continue
        # Real swap
        new_partition[u], new_partition[v] = new_partition[v], new_partition[u]
    
    return new_partition


def cyclic_expansion_refine(
    adjacency: List[List[Tuple[int, float]]],
    partition: np.ndarray,
    q: int,
    max_iterations: int = 20,
    max_candidates: int = 50,
    num_trials: int = 8,
    num_steps: int = 200,
    dev: str = 'cpu',
    patience: int = 5,
    verbose: bool = False,
    seed: Optional[int] = None,
    allow_nonadjacent: bool = False,
    enable_virtual: bool = False,
) -> np.ndarray:
    """Run Cyclic Expansion QUBO refinement on a graph partition.
    
    Args:
        adjacency: adjacency list where adjacency[i] = [(neighbor, weight), ...]
        partition: initial partition assignment (length n, values in 0..q-1)
        q: number of partitions
        max_iterations: maximum number of refinement iterations
        max_candidates: maximum candidate pairs per iteration
        num_trials: FEM trials
        num_steps: FEM steps
        dev: device for FEM
        patience: iterations without improvement before stopping
        seed: optional RNG seed for randomized candidate selection
        enable_virtual: If True, add virtual nodes to balance swap groups
            so nodes can flow to partitions with fewer elements.
            Experimental feature.
        
    Returns:
        Refined partition assignment
    """
    rng = np.random.default_rng(seed)
    n = len(partition)
    best_partition = partition.copy()
    
    def compute_cut(adj, parts):
        """Compute edge cut value."""
        cut = 0.0
        for i in range(n):
            for j, w in adj[i]:
                if i < j and parts[i] != parts[j]:
                    cut += w
        return cut
    
    best_cut = compute_cut(adjacency, best_partition)
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        # Select candidate pairs
        candidate_pairs = select_candidate_pairs(
            adjacency, best_partition, q, max_candidates, rng=rng, allow_nonadjacent=allow_nonadjacent
        )
        
        if len(candidate_pairs) == 0:
            break
        
        # Extend with virtual pairs if enabled
        extended_pairs = list(candidate_pairs)
        if enable_virtual:
            # Compute partition sizes
            part_sizes = {p: int(np.sum(best_partition == p)) for p in range(q)}
            
            # Group pairs by the partition pair they span
            pair_groups: Dict[Tuple[int, int], List[int]] = {}
            for t, (u, v) in enumerate(candidate_pairs):
                pu, pv = int(best_partition[u]), int(best_partition[v])
                key = tuple(sorted((pu, pv)))
                if key not in pair_groups:
                    pair_groups[key] = []
                pair_groups[key].append(t)
            
            used_nodes: Set[int] = set()
            for u, v in candidate_pairs:
                used_nodes.add(u)
                used_nodes.add(v)
            
            for (pa, pb), indices in pair_groups.items():
                size_a = part_sizes.get(pa, 0)
                size_b = part_sizes.get(pb, 0)
                if size_a == size_b:
                    continue
                
                # The smaller partition should receive flow from the larger one
                smaller = pa if size_a < size_b else pb
                larger = pb if size_a < size_b else pa
                size_diff = abs(size_a - size_b)
                
                # Find boundary nodes from the larger partition not already paired
                boundary_larger = [
                    v for v in find_boundary_vertices(adjacency, best_partition, q)
                    if int(best_partition[v]) == larger and v not in used_nodes
                ]
                
                # Add virtual pairs encoded with target partition
                virtual_count = min(size_diff, len(boundary_larger), len(indices))
                rng.shuffle(boundary_larger)
                for i in range(virtual_count):
                    # Encode target (smaller) partition in the virtual sentinel
                    encoded_target = _encode_virtual_target(smaller)
                    extended_pairs.append((boundary_larger[i], encoded_target))
                    used_nodes.add(boundary_larger[i])
        
        # Build QUBO matrix (may include virtual pairs)
        Q = build_qubo_matrix(adjacency, best_partition, extended_pairs, enable_virtual=enable_virtual)
        
        # Solve QUBO with FEM
        swap_decision = solve_qubo_with_fem(Q, num_trials, num_steps, dev)
        
        # Apply real swaps
        new_partition = apply_swaps(best_partition, extended_pairs, swap_decision)
        
        # Apply virtual swaps: move the real node to the encoded target partition
        if enable_virtual:
            def _is_virtual(val: int) -> bool:
                return val < 0
            for virt_idx, (u, v) in enumerate(extended_pairs[len(candidate_pairs):], start=len(candidate_pairs)):
                if virt_idx < len(swap_decision) and swap_decision[virt_idx] == 1:
                    if _is_virtual(u):
                        # (virtual, real_node): move real node v
                        target_part = _decode_virtual_target(u)
                        new_partition[v] = target_part
                    else:
                        # (real_node, virtual): move real node u
                        target_part = _decode_virtual_target(v)
                        new_partition[u] = target_part
        
        new_cut = compute_cut(adjacency, new_partition)

        if verbose:
            print(f"CyclicExp iter={iteration} cand={len(candidate_pairs)} virt={len(extended_pairs)-len(candidate_pairs) if enable_virtual else 0} new_cut={new_cut} best_cut={best_cut}")

        # Only accept strictly improving partitions. This ensures monotonicity
        # w.r.t. best_cut — longer `max_iterations` cannot worsen the returned
        # partition. If we ever observe best_cut increasing, log details.
        if new_cut < best_cut:
            best_cut = new_cut
            best_partition = new_partition
            no_improve_count = 0
        else:
            # Log non-improving result for diagnostics
            if verbose:
                print(
                    f"CyclicExp: iteration {iteration} produced non-improving partition (new_cut={new_cut} >= best_cut={best_cut})",
                    file=sys.stderr,
                )
            no_improve_count += 1
            if no_improve_count >= patience:
                break
    
    # Sanity: ensure we return a partition no worse than the initial one
    final_cut = compute_cut(adjacency, best_partition)
    if final_cut > best_cut:
        # This should not happen; fallback to the original input partition
        if verbose:
            print(f"CyclicExp: sanity failed final_cut={final_cut} > best_cut={best_cut}; reverting.", file=sys.stderr)
        return partition.copy()

    return best_partition


def adjacency_from_sparse(
    J: torch.Tensor
) -> List[List[Tuple[int, float]]]:
    """Convert a sparse torch tensor to adjacency list format.
    
    Args:
        J: sparse or dense torch tensor (n x n)
        
    Returns:
        adjacency list where adjacency[i] = [(neighbor, weight), ...]
    """
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    
    n = J.shape[0]
    adjacency = [[] for _ in range(n)]
    
    indices = J.indices()
    values = J.values()
    
    for idx in range(values.numel()):
        i = int(indices[0, idx])
        j = int(indices[1, idx])
        w = float(values[idx].item())
        
        if i != j:  # Skip self-loops
            adjacency[i].append((j, w))
    
    return adjacency
