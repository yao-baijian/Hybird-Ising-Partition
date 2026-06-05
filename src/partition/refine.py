"""Normal-graph refinement functions.

Provides FM-style local refinement and PyMetis wrappers for graph partitioning.
"""

import heapq
import numpy as np
import importlib
import inspect
import sys


def _fm_refinement(adjacency_list, q, part, epsilon=0.05, max_passes=10):
    """FM-style local refinement for graph partitioning.
    
    This is a self-contained refinement routine that can be used when
    pymetis doesn't support passing an initial partition.
    """
    n = len(adjacency_list)
    part = [int(x) for x in part]
    
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in adjacency_list[i]:
            if j != i:
                neighbors[i].append((j, 1.0))
    
    def edgecut_of(parts):
        cut = 0.0
        for i in range(n):
            pi = parts[i]
            for j, w in neighbors[i]:
                if i < j and pi != parts[j]:
                    cut += w
        return int(round(cut))
    
    def best_destination(vertex, parts):
        old = parts[vertex]
        weight_to = np.zeros(q, dtype=float)
        for nbr, w in neighbors[vertex]:
            weight_to[parts[nbr]] += w
        best_group = old
        best_delta = 0.0
        for g in range(q):
            if g == old:
                continue
            delta = weight_to[old] - weight_to[g]
            if delta < best_delta:
                best_delta = delta
                best_group = g
        return best_group, float(best_delta)
    
    counts = np.bincount(np.asarray(part, dtype=int), minlength=q).astype(int)
    ideal = n / float(q)
    max_size = ideal * (1.0 + float(epsilon))
    
    def feasible_move(group_sizes, old_group, new_group):
        return group_sizes[new_group] + 1 <= max_size
    
    for _pass in range(max_passes):
        locked = np.zeros(n, dtype=bool)
        current_parts = part[:]
        current_counts = counts.copy()
        pass_start_cut = edgecut_of(current_parts)
        current_cut = pass_start_cut
        best_cut = current_cut
        best_state = current_parts[:]
        
        scale = 1000.0
        buckets = {}
        vertex_target = {}
        
        def gain_key(delta):
            return int(round(-delta * scale))
        
        def insert_vertex(v):
            if locked[v]:
                return
            g, delta = best_destination(v, current_parts)
            if g == current_parts[v] or not feasible_move(current_counts, current_parts[v], g):
                return
            k = gain_key(delta)
            buckets.setdefault(k, []).append(v)
            vertex_target[v] = g
        
        for v in range(n):
            insert_vertex(v)
        
        moved = False
        while buckets:
            best_k = max(buckets.keys())
            v = buckets[best_k].pop()
            if not buckets[best_k]:
                del buckets[best_k]
            
            if locked[v]:
                vertex_target.pop(v, None)
                continue
            
            best_g, best_delta = best_destination(v, current_parts)
            k_new = gain_key(best_delta)
            if best_g != vertex_target.get(v, None) or k_new != best_k:
                vertex_target[v] = best_g
                if best_g != current_parts[v] and feasible_move(current_counts, current_parts[v], best_g):
                    buckets.setdefault(k_new, []).append(v)
                else:
                    vertex_target.pop(v, None)
                continue
            
            g = best_g
            delta = best_delta
            if g == current_parts[v] or not feasible_move(current_counts, current_parts[v], g):
                locked[v] = True
                vertex_target.pop(v, None)
                continue
            
            old = current_parts[v]
            current_parts[v] = g
            current_counts[old] -= 1
            current_counts[g] += 1
            locked[v] = True
            moved = True
            vertex_target.pop(v, None)
            
            current_cut += int(round(delta))
            if current_cut < best_cut:
                best_cut = current_cut
                best_state = current_parts[:]
            
            for nbr, _w in neighbors[v]:
                if not locked[nbr]:
                    insert_vertex(nbr)
        
        if not moved:
            break
        
        part = best_state
        counts = np.bincount(np.asarray(part, dtype=int), minlength=q).astype(int)
        
        if best_cut >= pass_start_cut:
            break
    
    return edgecut_of(part), part


def simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, someflag=False,
                  arg7=0, arg8=0, part=None, max_passes=10, num_restarts=5):
    """Simple replacement for kaffpa: FM-style local refinement with perturbation restarts.
    
    Returns (edgecut, part_list).
    """
    n = len(vwgt)
    vwgt = np.asarray(vwgt, dtype=float)
    if part is None:
        base = np.arange(n) % q
        np.random.shuffle(base)
        part = base.tolist()
    else:
        part = [int(x) for x in part]

    total_weight = float(vwgt.sum())
    ideal = total_weight / float(q) if q > 0 else 0.0
    max_block_weight = ideal * (1.0 + float(epsilon))
    min_block_weight = ideal * (1.0 - float(epsilon))

    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for idx in range(xadj[i], xadj[i + 1]):
            j = int(adjncy[idx])
            w = float(adjcwgt[idx])
            if j != i:
                neighbors[i].append((j, w))

    block_weights = np.zeros(q, dtype=float)
    for vertex, block in enumerate(part):
        block_weights[block] += float(vwgt[vertex])

    def edgecut_of(parts):
        cut = 0.0
        for i in range(n):
            pi = parts[i]
            for j, w in neighbors[i]:
                if i < j and pi != parts[j]:
                    cut += w
        return int(round(cut))

    def partition_imbalance(weights):
        if ideal <= 0.0:
            return 0.0
        return float(np.max(np.abs(weights - ideal) / ideal))

    def recompute_block_weights(parts):
        weights = np.zeros(q, dtype=float)
        for vertex, block in enumerate(parts):
            weights[block] += float(vwgt[vertex])
        return weights

    def cut_delta_for_move(vertex, old_group, new_group, parts):
        weight_to = np.zeros(q, dtype=float)
        for nbr, w in neighbors[vertex]:
            weight_to[parts[nbr]] += w
        return float(weight_to[old_group] - weight_to[new_group])

    def best_feasible_destination(vertex, parts, weights):
        old_group = parts[vertex]
        vertex_weight = float(vwgt[vertex])
        weight_to = np.zeros(q, dtype=float)
        for nbr, w in neighbors[vertex]:
            weight_to[parts[nbr]] += w

        best_group = old_group
        best_delta = 0.0
        best_balance = abs(weights[old_group] - ideal)
        for new_group in range(q):
            if new_group == old_group:
                continue
            if weights[new_group] + vertex_weight > max_block_weight:
                continue
            delta = float(weight_to[old_group] - weight_to[new_group])
            new_balance = abs(weights[new_group] + vertex_weight - ideal)
            if delta > best_delta or (delta == best_delta and new_balance < best_balance):
                best_delta = delta
                best_group = new_group
                best_balance = new_balance
        return best_group, best_delta

    def apply_move(parts, weights, vertex, old_group, new_group):
        parts[vertex] = new_group
        vertex_weight = float(vwgt[vertex])
        weights[old_group] -= vertex_weight
        weights[new_group] += vertex_weight

    def fm_refine(parts, weights):
        current_parts = parts[:]
        current_weights = weights.copy()

        for _pass in range(max_passes):
            locked = np.zeros(n, dtype=bool)
            pass_start_cut = edgecut_of(current_parts)
            best_cut = pass_start_cut
            best_state = current_parts[:]
            best_weights = current_weights.copy()
            moved_any = False

            heap = []
            for v in range(n):
                old_group = current_parts[v]
                best_group, best_delta = best_feasible_destination(v, current_parts, current_weights)
                if best_group != old_group:
                    heapq.heappush(heap, (-best_delta, v, best_group))

            while heap:
                neg_delta, vertex, queued_group = heapq.heappop(heap)
                if locked[vertex]:
                    continue

                old_group = current_parts[vertex]
                best_group, best_delta = best_feasible_destination(vertex, current_parts, current_weights)
                if best_group == old_group:
                    locked[vertex] = True
                    continue
                if best_group != queued_group or abs(best_delta + neg_delta) > 1e-9:
                    heapq.heappush(heap, (-best_delta, vertex, best_group))
                    continue

                apply_move(current_parts, current_weights, vertex, old_group, best_group)
                locked[vertex] = True
                moved_any = True

                current_cut = edgecut_of(current_parts)
                if current_cut < best_cut:
                    best_cut = current_cut
                    best_state = current_parts[:]
                    best_weights = current_weights.copy()

            if not moved_any:
                break

            current_parts = best_state
            current_weights = best_weights

            if best_cut >= pass_start_cut:
                break

        return current_parts, current_weights

    def repair_imbalance(parts, weights):
        parts = parts[:]
        weights = weights.copy()

        for _ in range(max_passes * max(1, q)):
            if partition_imbalance(weights) <= float(epsilon):
                break

            overloaded = [g for g in range(q) if weights[g] > max_block_weight]
            if not overloaded:
                break

            underloaded = [g for g in range(q) if weights[g] < min_block_weight]

            best_move = None
            best_key = None

            for from_group in overloaded:
                for vertex in range(n):
                    if parts[vertex] != from_group:
                        continue

                    vertex_weight = float(vwgt[vertex])
                    weight_to = np.zeros(q, dtype=float)
                    for nbr, w in neighbors[vertex]:
                        weight_to[parts[nbr]] += w

                    candidate_groups = underloaded if underloaded else list(range(q))
                    for to_group in candidate_groups:
                        if to_group == from_group:
                            continue
                        if weights[to_group] + vertex_weight > max_block_weight:
                            continue

                        delta = float(weight_to[from_group] - weight_to[to_group])
                        target_is_underloaded = weights[to_group] < min_block_weight
                        key = (1 if target_is_underloaded else 0, delta, -weights[to_group])
                        if best_key is None or key > best_key:
                            best_key = key
                            best_move = (vertex, from_group, to_group)

            if best_move is None:
                break

            vertex, from_group, to_group = best_move
            apply_move(parts, weights, vertex, from_group, to_group)

        return parts, weights

    def _run_one_fm(parts_in, weights_in):
        p, w = fm_refine(parts_in, weights_in)
        p, w = repair_imbalance(p, w)
        return p, w, edgecut_of(p)

    best_part, best_weights, best_cut = _run_one_fm(part, block_weights)

    for _restart in range(num_restarts):
        boundary = []
        for v in range(n):
            pv = best_part[v]
            for nbr, _ in neighbors[v]:
                if best_part[nbr] != pv:
                    boundary.append(v)
                    break

        if len(boundary) < 2:
            break

        np.random.shuffle(boundary)
        perturb_count = max(1, len(boundary) // 6)
        perturbed = best_part[:]
        p_weights = best_weights.copy()
        for v in boundary[:perturb_count]:
            old = perturbed[v]
            new = int(np.random.randint(0, q - 1))
            if new >= old:
                new += 1
            perturbed[v] = new
            vw = float(vwgt[v])
            p_weights[old] -= vw
            p_weights[new] += vw

        p2, w2, c2 = _run_one_fm(perturbed, p_weights)
        if c2 < best_cut:
            best_cut = c2
            best_part = p2
            best_weights = w2

    return best_cut, best_part


def call_pymetis_with_part(q, adjacency_list, part=None, epsilon=0.05, max_passes=10, verbose=False):
    """Call pymetis.part_graph with optional initial partition refinement.
    
    When `part` is provided and pymetis supports it, passes the initial
    partition. Otherwise uses our own FM-style refinement.
    
    Returns:
        (edgecuts, parts) tuple
    """
    try:
        pymetis = importlib.import_module('pymetis')
    except Exception as e:
        raise ImportError(f"pymetis is not available: {e}")

    if part is None:
        result = pymetis.part_graph(q, adjacency=adjacency_list)
        if hasattr(result, 'edgecut'):
            return result.edgecut, list(result.partition)
        return result
    
    try:
        sig = inspect.signature(pymetis.part_graph)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    if 'part' in params:
        return pymetis.part_graph(q, adjacency=adjacency_list, part=part)
    else:
        if verbose:
            print(f"[INFO] pymetis.part_graph does not accept 'part'; using FM refinement (q={q})",
                  file=sys.stderr)
        return _fm_refinement(adjacency_list, q, part, epsilon=epsilon, max_passes=max_passes)
