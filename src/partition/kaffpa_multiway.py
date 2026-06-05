"""Multi-level graph partitioner (KaFFPa-style) with FEM initial partition.

Two entry points:
  - kaffpa_multiway_kway(...)   — greedy+FM init (for kaffpa_kway)
  - fem_multilevel_refine(...)  — FEM init on coarsest (for coarse_fem_refine_kaffpa)

Features:
  - Multi-level coarsening with ~50% reduction per round
  - Look-ahead FM refinement (boundary tracking, negative-gain moves)
  - Multiple perturbation restarts
  - Global balance optimisation
  - Adaptive coarsening target for small graphs
"""

import numpy as np
import torch
import heapq
import time
from typing import List, Tuple, Dict, Optional, Set, Sequence


# =============================================================================
# Level storage
# =============================================================================

class CoarseningLevel:
    __slots__ = ('fine_adj', 'fine_vwgt', 'coarse_to_fine')
    def __init__(self, fine_adj, fine_vwgt, coarse_to_fine):
        self.fine_adj = fine_adj
        self.fine_vwgt = fine_vwgt
        self.coarse_to_fine = coarse_to_fine


# =============================================================================
# Look-ahead FM refinement
# =============================================================================

def _edgecut_of(part, adj):
    cut = 0.0
    for i in range(len(part)):
        pi = part[i]
        for j, w in adj[i]:
            if i < j and pi != part[j]:
                cut += w
    return cut


def fm_refine_lookahead(
    part, adj, vwgt, q,
    epsilon=0.05, max_passes=15, seed=42,
):
    """Look-ahead FM with boundary tracking and perturbation restarts.

    Allows negative-gain moves (hill climbing) to escape local optima.
    Uses relaxed balance during search, repairs at end.
    """
    rng = np.random.default_rng(seed)
    n = len(part)
    part = list(part)
    vw_a = np.asarray(vwgt, dtype=float)
    tw = float(vw_a.sum())
    ideal = tw / float(q) if q > 0 else 0.0
    # Heavily relaxed balance during passes — 5x epsilon for exploration
    pass_max = ideal * (1.0 + 2.0 * epsilon)
    hard_max = ideal * (1.0 + epsilon)

    best_global = list(part)
    best_global_cut = _edgecut_of(part, adj)

    def _gain(v, old_g, new_g, parts):
        g = 0.0
        for nb, w in adj[v]:
            pn = parts[nb]
            if pn == old_g:
                g -= w
            elif pn == new_g:
                g += w
        return g

    def _best_dest(v, parts, weights, max_w):
        """Return (best_group, best_delta). Allows negative delta (look-ahead)."""
        old = parts[v]
        wt = np.zeros(q, dtype=float)
        for nb, w in adj[v]:
            pn = parts[nb]
            if pn >= 0:
                wt[pn] += w
        bg, bd = old, -1e9
        for g in range(q):
            if g == old:
                continue
            if weights[g] + vw_a[v] > max_w:
                continue
            d = wt[g] - wt[old]
            if d > bd:
                bd, bg = d, g
        if bg == old:
            return old, 0.0
        return bg, bd

    def _one_pass(parts, weights, max_w, passes):
        cur = list(parts)
        cw = weights.copy()
        current_cut = _edgecut_of(cur, adj)
        n = len(cur)
        for _ in range(passes):
            on_b = np.zeros(n, dtype=bool)
            for v in range(n):
                pv = cur[v]
                for nb, _ in adj[v]:
                    if cur[nb] != pv:
                        on_b[v] = True
                        break
            locked = np.zeros(n, dtype=bool)
            start_cut = current_cut
            heap, in_h = [], np.zeros(n, dtype=bool)

            def _push(vv):
                if locked[vv] or not on_b[vv] or in_h[vv]:
                    return
                bg, bd = _best_dest(vv, cur, cw, max_w)
                if bg != cur[vv]:
                    heapq.heappush(heap, (-bd, vv, bg))
                    in_h[vv] = True

            for vv in range(n):
                if on_b[vv]:
                    _push(vv)
            if not heap:
                break

            # ---- Move history with locked vertices (standard FM) ----
            history = []          # (v, old_g, new_g, bd)
            cuts = [current_cut]  # cuts[0] = start; cuts[i] = cut after history[i-1]

            while heap:
                nd, vv, qg = heapq.heappop(heap)
                in_h[vv] = False
                if locked[vv] or not on_b[vv]:
                    continue
                old = cur[vv]
                bg, bd = _best_dest(vv, cur, cw, max_w)
                if bg == old:
                    locked[vv] = True
                    continue
                if bg != qg or abs(bd + nd) > 1e-9:
                    _push(vv)
                    continue
                # Execute move
                cur[vv] = bg; cw[old] -= vw_a[vv]; cw[bg] += vw_a[vv]
                current_cut -= bd
                locked[vv] = True
                history.append((vv, old, bg, bd))
                cuts.append(current_cut)
                # Update boundary
                aff = {vv}; aff.update(nb for nb, _ in adj[vv])
                for u in aff:
                    pu = cur[u]
                    on_b[u] = any(cur[nb] != pu for nb, _ in adj[u])
                for u in aff:
                    if not locked[u] and on_b[u]:
                        _push(u)

            # Best-prefix rollback: find the prefix with minimum cut
            if not history:
                break
            best_idx = int(np.argmin(cuts))
            if best_idx == 0:
                # No improvement — restore start state
                for i in range(len(history)):
                    vv, old, bg, bd = history[i]
                    if cur[vv] == bg:
                        cur[vv] = old
                        cw[bg] -= vw_a[vv]; cw[old] += vw_a[vv]
                current_cut = cuts[0]
                break

            # Rollback: undo moves after best_idx
            for i in range(len(history) - 1, best_idx - 1, -1):
                vv, old, bg, bd = history[i]
                if cur[vv] == bg:
                    cur[vv] = old
                    cw[bg] -= vw_a[vv]; cw[old] += vw_a[vv]
            current_cut = cuts[best_idx]

            if current_cut >= start_cut:
                break
        return cur, cw, current_cut

    # ---- Main: outer restarts ----
    bw = np.zeros(q, dtype=float)
    for vv, blk in enumerate(part):
        bw[blk] += vw_a[vv]

    for _outer in range(2):
        cp, cw, cc = _one_pass(part, bw, pass_max, max_passes)
        if cc < best_global_cut:
            best_global, best_global_cut = list(cp), cc

        # Perturb boundary
        bdry = []
        for vv in range(n):
            pv = best_global[vv]
            if any(best_global[nb] != pv for nb, _ in adj[vv]):
                bdry.append(vv)
        if len(bdry) < 3:
            break
        rng.shuffle(bdry)
        npert = max(1, len(bdry) // 8)
        pert = list(best_global)
        pw = np.zeros(q, dtype=float)
        for vv in range(n):
            pw[pert[vv]] += vw_a[vv]
        for vv in bdry[:npert]:
            old = pert[vv]
            new = (old + 1 + rng.integers(0, q - 1)) % q
            pert[vv] = new
            pw[old] -= vw_a[vv]; pw[new] += vw_a[vv]
        cp2, _, cc2 = _one_pass(pert, pw, pass_max, max_passes // 2)
        if cc2 < best_global_cut:
            best_global, best_global_cut = list(cp2), cc2

    # ---- Balance repair (minimum cut increase) ----
    fw = np.zeros(q, dtype=float)
    for vv, blk in enumerate(best_global):
        fw[blk] += vw_a[vv]
    imb = float(np.max(np.abs(fw - ideal) / ideal)) if ideal > 0 else 0.0
    if imb > epsilon:
        # Build a priority queue of best moves: (cut_increase, v, from_g, to_g)
        repair_heap = []
        for vv in range(n):
            old = best_global[vv]
            if fw[old] <= hard_max:
                continue  # only consider moving vertices OUT of overloaded blocks
            for tg in range(q):
                if tg == old or fw[tg] + vw_a[vv] > hard_max:
                    continue
                gg = _gain(vv, old, tg, best_global)
                repair_heap.append((-gg, vv, old, tg))  # -gg = cut increase
        heapq.heapify(repair_heap)

        best_during = list(best_global)
        best_dc = _edgecut_of(best_during, adj)
        max_cut_increase = 0.15 * abs(best_dc) + 1.0  # allow up to 15% cut increase

        current_repair_cut = best_dc
        total_increase = 0.0
        while repair_heap and total_increase < max_cut_increase:
            imb = float(np.max(np.abs(fw - ideal) / ideal)) if ideal > 0 else 0.0
            if imb <= epsilon:
                break
            over = [g for g in range(q) if fw[g] > hard_max]
            if not over:
                break
            neg_gg, vv, fg, tg = heapq.heappop(repair_heap)
            if best_global[vv] != fg:
                continue
            if fw[tg] + vw_a[vv] > hard_max:
                continue
            # Apply move
            best_global[vv] = tg
            fw[fg] -= vw_a[vv]; fw[tg] += vw_a[vv]
            # CORRECTED: Use incremental cut update instead of O(E) _edgecut_of
            delta = -neg_gg  # neg_gg is the cut increase, so -neg_gg is the actual delta
            total_increase += delta
            current_repair_cut += delta
            if current_repair_cut < best_dc:
                best_dc, best_during = current_repair_cut, list(best_global)

        # After the while loop, if we successfully reached the balance target,
        # keep the result even if the cut increased within our 15% budget.
        # Only revert if we failed to reach the target AND the cut exploded.
        final_imb = float(np.max(np.abs(fw - ideal) / ideal))
        if final_imb > epsilon and _edgecut_of(best_global, adj) > (best_dc * 1.15):
            best_global = best_during  # Only revert as a last resort

    return int(round(_edgecut_of(best_global, adj))), best_global


# =============================================================================
# Coarsening
# =============================================================================

def _he_match_one_round(adj, node_weights, max_node_weight, rng):
    """Heavy-edge matching with degree-ordered processing.

    Vertices with highest total incident weight are matched first, leading to
    more balanced coarse nodes and better structure preservation.
    """
    n = len(adj)
    # Compute vertex degree (total incident edge weight)
    deg = np.array([sum(abs(w) for _, w in d.items()) for d in adj], dtype=float)
    # Process in descending degree order (ties broken randomly)
    order = np.argsort(-deg, kind='stable')
    # Add random tie-breaking
    tie = rng.random(n)
    order = np.lexsort((tie, -deg))

    matched = np.zeros(n, dtype=bool)
    coarse_of = np.full(n, -1, dtype=int)
    next_c = 0
    for u in order:
        if matched[u]:
            continue
        matched[u] = True
        best_v, best_w = -1, -1.0
        for v, ew in adj[u].items():
            if not matched[v] and node_weights[u] + node_weights[v] <= max_node_weight:
                if ew > best_w:
                    best_w, best_v = ew, v
        if best_v != -1:
            matched[best_v] = True
            coarse_of[u] = coarse_of[best_v] = next_c
        else:
            coarse_of[u] = next_c
        next_c += 1
    return dict(enumerate(coarse_of))


# =============================================================================
# Initial partition
# =============================================================================

def _greedy_grow(adj, vwgt, q, seed=0):
    rng = np.random.default_rng(seed)
    n = len(adj)
    vw_a = np.asarray(vwgt, dtype=float)
    total = float(vw_a.sum())
    target = total / float(q)
    part = np.full(n, -1, dtype=int)
    for block in range(q):
        bt = target if block < q - 1 else vw_a[part == -1].sum()
        bw = 0.0
        unass = np.where(part == -1)[0]
        if not len(unass):
            break
        sv = int(rng.choice(unass))
        part[sv] = block
        bw += vw_a[sv]
        heap, in_h = [], set()
        internal = np.zeros(n, dtype=float)
        def _push(v):
            if part[v] != -1 or v in in_h:
                return
            ext = 0.0
            for nb, w in adj[v]:
                if part[nb] == block:
                    internal[v] += w
                elif part[nb] == -1:
                    ext += w
            heapq.heappush(heap, (-(internal[v] - ext), v))
            in_h.add(v)
        for nb, w in adj[sv]:
            _push(nb)
        while heap and bw < bt:
            _, v = heapq.heappop(heap)
            in_h.discard(v)
            if part[v] != -1 or (bw + vw_a[v] > bt and block < q - 1):
                continue
            part[v] = block
            bw += vw_a[v]
            for nb, w in adj[v]:
                if part[nb] == -1:
                    internal[nb] = 0.0
                    for nb2, w2 in adj[nb]:
                        if part[nb2] == block or nb2 == v:
                            internal[nb] += w2
                    _push(nb)
    for block in range(q):
        mask = part == -1
        if not mask.any():
            break
        part[mask] = block
    return int(round(_edgecut_of(part.tolist(), adj))), part.tolist()


def initial_partition_greedy_fm(adj, vwgt, q, num_trials=5, seed=42):
    best_cut, best_part = float('inf'), None
    for t in range(num_trials):
        s = seed + t * 7 + 1
        cut, p = _greedy_grow(adj, vwgt, q, seed=s)
        c2, p2 = fm_refine_lookahead(p, adj, vwgt, q, epsilon=0.1, max_passes=3, seed=s + 100)
        if c2 < best_cut:
            best_cut, best_part = c2, p2
    if best_part is None:
        best_part = list(np.arange(len(adj)) % q)
        best_cut = int(1e9)
    return best_cut, best_part


def initial_partition_kahip(adj, vwgt, q):
    """Initial partition via kahip on the coarse graph."""
    import kahip
    n = len(adj)
    g = kahip.kahip_graph()
    g.set_num_nodes(n)
    seen = set()
    for i in range(n):
        for j, w in adj[i]:
            if (i, j) not in seen:
                g.add_undirected_edge(i, j, int(round(w)))
                seen.add((i, j))
                seen.add((j, i))
    vwgt_arr, xadj, adjcwgt, adjncy = g.get_csr_arrays()
    _, part = kahip.kaffpa(vwgt_arr, xadj, adjcwgt, adjncy,
                            q, 0.03, 0, 0, int(kahip.ECO))
    return _edgecut_of(part, adj), part


def initial_partition_fem(coarse_adj, coarse_vwgt, q,
                           num_trials=8, num_steps=200, dev='cpu', anneal='lin'):
    """Initial partition via FEM QUBO solver on the coarse graph."""
    from src.fem.initial_partition import fem_initial_partition_kway
    nc = len(coarse_adj)
    mat = np.zeros((nc, nc), dtype=float)
    for i in range(nc):
        for j, w in coarse_adj[i]:
            mat[i, j] = w
    c_np = np.asarray(coarse_vwgt, dtype=float).reshape(-1)
    assign = fem_initial_partition_kway(mat, None, None, c_np, k=q,
        lambda_penalty=1.0, num_trials=num_trials, num_steps=num_steps,
        dev=dev, anneal=anneal)
    return int(round(_edgecut_of(assign.tolist(), coarse_adj))), assign.tolist()


def initial_partition_sbm(coarse_adj, coarse_vwgt, q,
                           num_trials=8, num_steps=200, dev='cpu'):
    """Initial partition via SBM (Simulated Bifurcation) on the coarse graph.

    Uses bsb_bmincut_batch from sbm.py.  For q > 2, falls back to recursive
    bisection (same logic as direct_sbm in test_bmincut_base).
    """
    from src.sbm.sbm import bsb_bmincut_batch

    nc = len(coarse_adj)
    # Build dense coupling matrix for the coarse graph
    mat = np.zeros((nc, nc), dtype=float)
    for i in range(nc):
        for j, w in coarse_adj[i]:
            mat[i, j] = w
    J = torch.tensor(mat, dtype=torch.float32, device=dev)

    dt = 0.1

    if q == 2:
        batch_size = num_trials
        init_x = 2 * torch.rand(batch_size, nc, device=dev) - 1
        init_y = 2 * torch.rand(batch_size, nc, device=dev) - 1
        _, sol, cut_values, _ = bsb_bmincut_batch(
            J, init_x, init_y, num_steps, dt, lambda_balance=1.0,
        )
        best_idx = int(torch.argmin(cut_values).item())
        spins = sol[best_idx]
        part = [0 if s == 1 else 1 for s in spins.tolist()]
    else:
        # Recursive bisection
        current_part = torch.zeros(nc, dtype=torch.long, device=dev)
        n_parts = 1
        next_label = 1
        while n_parts < q:
            sizes = torch.bincount(current_part, minlength=n_parts)
            largest = int(torch.argmax(sizes).item())
            mask = current_part == largest
            sub_idx = torch.where(mask)[0]
            if len(sub_idx) <= 1:
                break
            J_sub = J[sub_idx][:, sub_idx]
            sub_n = len(sub_idx)
            bs = max(num_trials, 5)
            i_x = 2 * torch.rand(bs, sub_n, device=dev) - 1
            i_y = 2 * torch.rand(bs, sub_n, device=dev) - 1
            _, sol_sub, cut_sub, _ = bsb_bmincut_batch(
                J_sub, i_x, i_y, max(num_steps // n_parts, 50),
                dt, lambda_balance=1.0,
            )
            best_sub = int(torch.argmin(cut_sub).item())
            spins_sub = sol_sub[best_sub]
            current_part[sub_idx[spins_sub == -1]] = next_label
            next_label += 1
            n_parts += 1
        part = [min(int(c.item()), q - 1) for c in current_part]

    return int(round(_edgecut_of(part, coarse_adj))), part


# =============================================================================
# Shared uncoarsening
# =============================================================================

def _uncoarsen_and_refine(levels, part, adj, vwgt, q, epsilon, refine_passes, seed, verbose):
    part = list(part)
    for li in range(len(levels) - 1, -1, -1):
        lev = levels[li]
        nf = len(lev.fine_adj)
        proj = [0] * nf
        for cn, members in lev.coarse_to_fine.items():
            for m in members:
                proj[m] = part[cn]
        bc = _edgecut_of(proj, lev.fine_adj)
        # Use only 2-3 passes for intermediate levels; full passes for the final one
        passes = 2 if li > 0 else refine_passes
        _, part = fm_refine_lookahead(proj, lev.fine_adj, lev.fine_vwgt, q,
                                       epsilon=epsilon, max_passes=passes,
                                       seed=seed + li)
        if verbose:
            ac = _edgecut_of(part, lev.fine_adj)
            print(f"  Level {li}: {nf} nodes {bc:.0f} -> {ac:.0f}")
    # Global polish
    best_p, best_c = list(part), _edgecut_of(part, adj)
    rng = np.random.default_rng(seed + 999)
    for gr in range(5):
        bdry = [v for v in range(len(adj))
                if any(best_p[nb] != best_p[v] for nb, _ in adj[v])]
        if len(bdry) < 3:
            break
        rng.shuffle(bdry)
        npert = max(1, len(bdry) // 10)
        pert = list(best_p)
        for v in bdry[:npert]:
            old = pert[v]
            pert[v] = (old + 1 + rng.integers(0, q - 1)) % q
        _, ref = fm_refine_lookahead(pert, adj, vwgt, q,
                                      epsilon=epsilon, max_passes=refine_passes // 2,
                                      seed=seed + 2000 + gr)
        cc = _edgecut_of(ref, adj)
        if cc < best_c:
            best_c, best_p = cc, ref
    return best_p


# =============================================================================
# Simplified: coarsen + save levels, then multilevel partition
# =============================================================================

def _coarsen_and_save_levels(adj, vwgt, coarsen_to, max_rounds, seed):
    """Returns (levels, coarse_adj, coarse_vwgt)."""
    rng = np.random.default_rng(seed)
    levels = []
    cur_adj = [dict(nbrs) for nbrs in adj]
    cur_vw = np.asarray(vwgt, dtype=float)
    for _ in range(max_rounds):
        if cur_vw.shape[0] <= coarsen_to:
            break
        max_nw = max(cur_vw.sum() / max(coarsen_to, 1), np.max(cur_vw) * 2)
        co = _he_match_one_round(cur_adj, cur_vw, max_nw, rng)
        grps = {}
        for ov, nv in co.items():
            grps.setdefault(nv, []).append(ov)
        nc = len(grps)
        ca = [{} for _ in range(nc)]
        cv = np.zeros(nc, dtype=float)
        for ci, members in grps.items():
            for ov in members:
                cv[ci] += cur_vw[ov]
                for nb, ew in cur_adj[ov].items():
                    cn = co[nb]
                    if cn != ci:
                        ca[ci][cn] = ca[ci].get(cn, 0.0) + ew
        fine = [[(nb, w) for nb, w in d.items()] for d in cur_adj]
        levels.append(CoarseningLevel(fine, list(cur_vw),
                                       {c: m for c, m in grps.items()}))
        cur_adj, cur_vw = ca, cv
    coarse_adj = [[(nb, w) for nb, w in d.items()] for d in cur_adj]
    coarse_vwgt = list(cur_vw)
    return levels, coarse_adj, coarse_vwgt


def multilevel_partition_v2(
    adj, vwgt, q,
    coarsen_to=20, epsilon=0.05, max_coarse_rounds=20,
    num_init_trials=10, refine_passes=10,
    use_fem_init=False, fem_trials=8, fem_steps=200, fem_dev='cpu', fem_anneal='lin',
    use_kahip_init=False,
    use_sbm_init=False, sbm_trials=8, sbm_steps=200, sbm_dev='cpu',
    skip_coarsen_small=False,
    seed=42, verbose=False,
):
    """Multi-level graph partition (correct version that returns coarsest adj)."""
    t0 = time.perf_counter()
    n = len(adj)

    # For small graphs (n < 2000), skip coarsening — direct FM on full graph
    if skip_coarsen_small and n < 2000:
        if verbose:
            print(f"[ML] n={n} < 2000, skipping coarsening, direct FM")
        ti_s = time.perf_counter()
        # More trials for the full graph since it's larger
        local_trials = max(num_init_trials, 8)
        _, part = initial_partition_greedy_fm(adj, vwgt, q,
                                               num_trials=local_trials, seed=seed)
        ti = time.perf_counter() - ti_s
        tr_s = time.perf_counter()
        # Multiple FM rounds with perturbations
        best_part, best_cut = list(part), _edgecut_of(part, adj)
        rng = np.random.default_rng(seed + 999)
        for fm_round in range(3):
            _, part = fm_refine_lookahead(part, adj, vwgt, q,
                                           epsilon=epsilon, max_passes=refine_passes,
                                           seed=seed + fm_round)
            cc = _edgecut_of(part, adj)
            if cc < best_cut:
                best_cut, best_part = cc, list(part)
            # Perturb and retry
            bdry = [v for v in range(n)
                    if any(best_part[nb] != best_part[v] for nb, _ in adj[v])]
            if len(bdry) < 3:
                break
            rng.shuffle(bdry)
            npert = max(1, len(bdry) // 10)
            pert = list(best_part)
            for v in bdry[:npert]:
                old = pert[v]
                pert[v] = (old + 1 + rng.integers(0, q - 1)) % q
            part = pert
        tr = time.perf_counter() - tr_s
        return best_part, 0.0, ti, tr, time.perf_counter() - t0, 0

    tc_s = time.perf_counter()
    levels, coarse_adj, coarse_vwgt = _coarsen_and_save_levels(
        adj, vwgt, coarsen_to, max_coarse_rounds, seed)
    coarsen_rounds = len(levels)
    tc = time.perf_counter() - tc_s
    if verbose:
        print(f"[ML] Coarsened {n} -> {len(coarse_adj)} nodes ({coarsen_rounds} rounds)")

    ti_s = time.perf_counter()
    if use_sbm_init:
        _, part = initial_partition_sbm(coarse_adj, coarse_vwgt, q,
                                         num_trials=sbm_trials, num_steps=sbm_steps,
                                         dev=sbm_dev)
    elif use_fem_init:
        _, part = initial_partition_fem(coarse_adj, coarse_vwgt, q,
                                         num_trials=fem_trials, num_steps=fem_steps,
                                         dev=fem_dev, anneal=fem_anneal)
    elif use_kahip_init:
        _, part = initial_partition_kahip(coarse_adj, coarse_vwgt, q)
    else:
        _, part = initial_partition_greedy_fm(coarse_adj, coarse_vwgt, q,
                                               num_trials=num_init_trials, seed=seed)
    ti = time.perf_counter() - ti_s
    if verbose:
        print(f"[ML] Initial cut: {_edgecut_of(part, coarse_adj):.0f}")

    tr_s = time.perf_counter()
    part = _uncoarsen_and_refine(levels, part, adj, vwgt, q, epsilon, refine_passes, seed, verbose)
    tr = time.perf_counter() - tr_s
    tt = time.perf_counter() - t0

    # Final balance check
    fw = np.zeros(q, dtype=float)
    for v, blk in enumerate(part):
        fw[blk] += vwgt[v]
    imb = float(np.max(np.abs(fw - (sum(vwgt) / q)) / (sum(vwgt) / q))) if q > 0 and sum(vwgt) > 0 else 0.0
    if imb > epsilon:
        if verbose:
            print(f"[ML] Balance exceeded ({imb:.4f} > {epsilon}); repairing...")
        _, part = fm_refine_lookahead(part, adj, vwgt, q, epsilon=epsilon,
                                       max_passes=refine_passes, seed=seed + 5000)

    return part, tc, ti, tr, tt, coarsen_rounds


# =============================================================================
# Public entry points
# =============================================================================

def kaffpa_multiway_kway(
    J, q, coarsen_to=20,
    epsilon=0.05, max_coarse_rounds=20,
    num_init_trials=10, refine_passes=10,
    seed=42, verbose=False,
):
    """Multi-level partitioning (greedy+FM init). Replaces kaffpa_kway."""
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    adj = [[] for _ in range(n)]
    idxs, vals = J.indices(), J.values()
    for i in range(idxs.shape[1]):
        r, c = int(idxs[0, i]), int(idxs[1, i])
        w = float(vals[i].item())
        if r != c:
            adj[r].append((c, w))
    vwgt = [1.0] * n

    part, tc, ti, tr, _, coarsen_rounds = multilevel_partition_v2(
        adj, vwgt, q,
        coarsen_to=coarsen_to, epsilon=epsilon,
        max_coarse_rounds=max_coarse_rounds,
        num_init_trials=num_init_trials, refine_passes=refine_passes,
        use_fem_init=False, seed=seed, verbose=verbose,
    )

    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i, lab in enumerate(part):
        p[i, lab] = 1.0
    from src.fem.problem import infer_bmincut
    _, cut = infer_bmincut(J, p.unsqueeze(0))
    return p, float(cut.item()), tc, ti, tr, coarsen_rounds


def fem_multilevel_refine(
    J, q, coarsen_to=20,
    epsilon=0.05, refine_passes=10,
    fem_trials=8, fem_steps=200, fem_dev='cpu', fem_anneal='lin',
    seed=42, verbose=False,
):
    """FEM + multi-level refinement. Replaces coarse_fem_refine_kaffpa."""
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    adj = [[] for _ in range(n)]
    idxs, vals = J.indices(), J.values()
    for i in range(idxs.shape[1]):
        r, c = int(idxs[0, i]), int(idxs[1, i])
        w = float(vals[i].item())
        if r != c:
            adj[r].append((c, w))
    vwgt = [1.0] * n

    part, tc, ti, tr, _, coarsen_rounds = multilevel_partition_v2(
        adj, vwgt, q,
        coarsen_to=coarsen_to, epsilon=epsilon,
        refine_passes=refine_passes,
        use_fem_init=True,
        fem_trials=fem_trials, fem_steps=fem_steps,
        fem_dev=fem_dev, fem_anneal=fem_anneal,
        seed=seed, verbose=verbose,
    )

    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i, lab in enumerate(part):
        p[i, lab] = 1.0
    from src.fem.problem import infer_bmincut
    _, cut = infer_bmincut(J, p.unsqueeze(0))
    return p, float(cut.item()), tc, ti, tr, coarsen_rounds


def sbm_multilevel_refine(
    J, q, coarsen_to=20,
    epsilon=0.05, refine_passes=10,
    sbm_trials=8, sbm_steps=200, sbm_dev='cpu',
    seed=42, verbose=False,
):
    """SBM + multi-level refinement. Uses SB for coarse initial partition."""
    if not J.is_sparse:
        J = J.to_sparse()
    J = J.coalesce()
    n = J.shape[0]
    adj = [[] for _ in range(n)]
    idxs, vals = J.indices(), J.values()
    for i in range(idxs.shape[1]):
        r, c = int(idxs[0, i]), int(idxs[1, i])
        w = float(vals[i].item())
        if r != c:
            adj[r].append((c, w))
    vwgt = [1.0] * n

    part, tc, ti, tr, _, coarsen_rounds = multilevel_partition_v2(
        adj, vwgt, q,
        coarsen_to=coarsen_to, epsilon=epsilon,
        refine_passes=refine_passes,
        use_sbm_init=True,
        sbm_trials=sbm_trials, sbm_steps=sbm_steps, sbm_dev=sbm_dev,
        seed=seed, verbose=verbose,
    )

    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
    for i, lab in enumerate(part):
        p[i, lab] = 1.0
    from src.fem.problem import infer_bmincut
    _, cut = infer_bmincut(J, p.unsqueeze(0))
    return p, float(cut.item()), tc, ti, tr, coarsen_rounds
