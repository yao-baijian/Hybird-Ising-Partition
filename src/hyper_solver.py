"""
Hypergraph solver classes.

Each solver type encapsulates a single concern:
- KahyparLikeSolver  → HEM / LSH coarsening (produces coarse structure)
- FemCoarsenSolver   → FEM-based initial partition on a coarsened hypergraph
- HyperRefineSolver  → local refinement on the full hypergraph

Usage::
    # 1. Coarsen once (returns hierarchy_stack for V-cycle)
    res = kahypar_solver.coarsen(hyperedges, num_nodes, q)

    # 2. Apply different initial partition strategies to the SAME coarse result
    greedy_assignment = kahypar_solver.initial_partition_greedy(
        res['coarse_hyperedges'], res['coarse_node_weights'], q)
    fem_assignment = fem_solver.initial_partition(
        res['coarse_hyperedges'], res['coarse_node_weights'], q)

    # 3. V-Cycle: project up through the hierarchy, refining at each level
    final = vcycle_uncoarsen(
        fem_assignment, res['hierarchy_stack'], hyperedges, q,
        refine_solver, verbose=True,
    )
"""

from __future__ import annotations
import heapq
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.partition.hyper_utils import (
    build_clique_expanded_graph,
    evaluate_kahypar_cut_value,
    greedy_initial_hypergraph_partition,
)


# ── Helper: build coarse hyperedges from groups ──────────────────────────


def _build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes):
    coarse_hyperedges = []
    for he in hyperedges:
        coarse_he = []
        seen = set()
        for v in he:
            if v < num_nodes:
                c = int(original_to_coarse[v])
                if c not in seen:
                    coarse_he.append(c)
                    seen.add(c)
        if len(coarse_he) > 1:
            coarse_hyperedges.append(coarse_he)
    return coarse_hyperedges


# ── Hypergraph solver base ───────────────────────────────────────────────


class HyperSolverBase:
    """Base class for hypergraph solvers."""

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self._config: Dict[str, Any] = {}

    def get_param(self, key: str, default=None):
        return self._config.get(key, default)

    def set_param(self, key: str, value):
        self._config[key] = value

    def update_params(self, **kwargs):
        self._config.update(kwargs)

    def get_all_params(self) -> Dict[str, Any]:
        return dict(self._config)


# ── KaHyPar-like coarsening solver ───────────────────────────────────────


class KahyparLikeSolver(HyperSolverBase):
    """HEM (heavy-edge matching) coarsening directly on hyperedges.

    This solver ONLY does coarsening – it returns the coarse hypergraph
    structure (groups, hyperedges, node weights) but does NOT produce an
    initial partition.  Use ``.initial_partition_greedy()`` for that, or
    pass the coarse structure to ``FemCoarsenSolver.initial_partition()``.
    """

    def coarsen(self, hyperedges, num_nodes, q, **overrides):
        """Run HEM coarsening rounds until ``coarsen_to`` is reached.

        Returns a dict with keys:
            coarse_groups, coarse_hyperedges, coarse_node_weights,
            original_to_coarse, coarse_graph.
        Does NOT include ``initial_assignment``.
        """
        p = {**self._config, **overrides}
        rng = np.random.default_rng(p.get('seed', None))

        target_coarse = max(1, int(p.get('coarsen_to', 50)))
        verbose = p.get('verbose', False)

        # ── Optionally pre-coarsen with LSH ───────────────────────────────
        use_lsh = p.get('use_lsh', False)
        if use_lsh:
            lsh_map, lsh_groups = _lsh_bucketize_vertices(
                hyperedges, num_nodes,
                target_buckets=max(1, target_coarse * 4),
                seed=p.get('seed', None),
                verbose=verbose,
            )
            if verbose:
                print(f"[kahypar_like] LSH pre-coarsen: {num_nodes} -> {len(lsh_groups)} buckets")
            current_hyperedges = _rebuild_hyperedges_from_groups(hyperedges, lsh_map, len(lsh_groups))
            current_groups = [list(g) for g in lsh_groups]
        else:
            current_hyperedges = [list(dict.fromkeys(he)) for he in hyperedges if len(set(he)) > 1]
            current_groups = [[i] for i in range(num_nodes)]

        current_n = len(current_groups)
        if current_n == 0:
            empty_graph = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32), (0, 0),
            ).coalesce()
            return {
                'coarse_graph': empty_graph,
                'coarse_node_weights': torch.empty((0,), dtype=torch.float32),
                'coarse_groups': [],
                'original_to_coarse': np.empty((0,), dtype=np.int64),
                'coarse_hyperedges': [],
            }

        # ── HEM matching rounds ──────────────────────────────────────────
        hierarchy_stack: list[dict] = []

        # Build incidence ONCE — updated statefully through the loop
        vertex_to_edges, edge_vertices, edge_weights = _build_incidence(current_hyperedges, current_n)

        round_id = 0
        while current_n > target_coarse:
            round_id += 1
            alive = np.ones(current_n, dtype=bool)
            matched = np.zeros(current_n, dtype=bool)
            partner = np.full(current_n, -1, dtype=np.int64)

            # vertex_to_edges / edge_vertices are already up-to-date from
            # the previous round's stateful merge — no rebuild needed.
            order = rng.permutation(current_n)
            pair_count = 0

            for u in order:
                if not alive[u] or matched[u]:
                    continue
                ratings = {}
                for eid in vertex_to_edges[u]:
                    verts = edge_vertices[eid]
                    if len(verts) < 2:
                        continue
                    contrib = float(edge_weights[eid]) / float(len(verts) - 1)
                    for v in verts:
                        if v != u and alive[v] and not matched[v]:
                            ratings[v] = ratings.get(v, 0.0) + contrib
                if not ratings:
                    continue
                v = max(ratings.items(), key=lambda item: (item[1], -item[0]))[0]
                if ratings[v] <= 0.0 or matched[v] or not alive[v]:
                    continue
                matched[u] = True
                matched[v] = True
                partner[u] = v
                partner[v] = u
                pair_count += 1

            if pair_count == 0:
                break

            remap = np.full(current_n + pair_count, -1, dtype=np.int64)
            new_groups = []
            new_id = 0
            used = np.zeros(current_n, dtype=bool)
            for u in range(current_n):
                if not alive[u] or used[u]:
                    continue
                v = partner[u]
                if v != -1 and used[v]:
                    continue
                if v != -1:
                    used[u] = used[v] = True
                    remap[u] = remap[v] = new_id
                    new_groups.append(current_groups[u] + current_groups[v])
                else:
                    used[u] = True
                    remap[u] = new_id
                    new_groups.append(current_groups[u])
                new_id += 1

            # ── Combined pass: build new_hyperedges (hierarchy stack)   ──
            #     AND simultaneously build updated vertex_to_edges so that
            #     we never need _build_incidence again inside the loop.
            new_hyperedges = []
            new_vertex_to_edges = [set() for _ in range(new_id)]

            for he in current_hyperedges:
                mapped = []
                seen = set()
                for v in he:
                    nv = remap[v]
                    if nv < 0 or nv in seen:
                        continue
                    mapped.append(int(nv))
                    seen.add(int(nv))
                if len(mapped) > 1:
                    eid = len(new_hyperedges)
                    new_hyperedges.append(mapped)
                    for mv in mapped:
                        new_vertex_to_edges[mv].add(eid)

            if new_id == current_n:
                break

            # ── Save hierarchy entry before transitioning ──
            hierarchy_stack.append({
                'hyperedges': [list(he) for he in current_hyperedges],
                'groups': [list(g) for g in current_groups],
                'remap': remap.copy(),
                'num_nodes': current_n,
            })

            # ── Update incidence statefully for the next round ──
            # edge_vertices and new_hyperedges share the same structure
            # (deduped vertex lists per hyperedge), so we can reuse the
            # new_hyperedges list directly.
            edge_vertices = new_hyperedges
            vertex_to_edges = new_vertex_to_edges
            # edge_weights stays at all-1.0 — unchanged by contraction

            current_hyperedges = new_hyperedges
            current_groups = new_groups
            current_n = new_id

        # ── Build output ─────────────────────────────────────────────────
        original_to_coarse = np.empty(num_nodes, dtype=np.int64)
        for idx, members in enumerate(current_groups):
            for member in members:
                if member < num_nodes:
                    original_to_coarse[member] = idx

        coarse_hyperedges_out = _build_coarse_hyperedges(hyperedges, original_to_coarse, num_nodes)
        coarse_graph = build_clique_expanded_graph(
            coarse_hyperedges_out, num_nodes=len(current_groups), normalize_weight=True,
        )
        coarse_node_weights = torch.tensor([len(g) for g in current_groups], dtype=torch.float32)

        return {
            'coarse_groups': current_groups,
            'coarse_hyperedges': coarse_hyperedges_out,
            'coarse_node_weights': coarse_node_weights,
            'original_to_coarse': original_to_coarse,
            'coarse_graph': coarse_graph,
            'hierarchy_stack': hierarchy_stack,
        }

    def initial_partition_greedy(self, coarse_hyperedges, coarse_node_weights, q, **overrides):
        """Greedy initial partition on the coarse hypergraph (respects node weights)."""
        p = {**self._config, **overrides}
        return greedy_initial_hypergraph_partition(
            coarse_hyperedges,
            coarse_node_weights.cpu().numpy() if torch.is_tensor(coarse_node_weights) else coarse_node_weights,
            q,
            hyperedge_weights=[1.0] * len(coarse_hyperedges),
            epsilon=p.get('epsilon', 0.03),
            seed=p.get('seed', None),
        )


# ── FEM initial-partition solver ─────────────────────────────────────────


class FemCoarsenSolver(HyperSolverBase):
    """FEM-based (or PUBO-based) initial partition on a coarsened hypergraph.

    This solver does NOT do coarsening.  It takes the coarse hypergraph
    (hyperedges + node weights) produced by ``KahyparLikeSolver`` and
    runs an optimization to obtain a weighted-balanced partition.

    Parameters (via ``update_params`` or ``**overrides``):
        method       — ``'fem'`` (default) or ``'pubo'``.
        map_type     — ``'clique'`` (default) or ``'star'`` expansion.
        num_trials, num_steps, dev, anneal — FEM solver settings.
    """

    def initial_partition(self, coarse_hyperedges, coarse_node_weights, q, **overrides):
        from src.fem import FEM as _FEM
        from src.fem.utils import hyperedge_list_to_coupling
        from src.partition.utils import make_q4_pubo_object

        p = {**self._config, **overrides}
        num_coarse = len(coarse_node_weights)
        num_trials = int(p.get('num_trials', 1))
        num_steps = int(p.get('num_steps', 10))
        dev = p.get('dev', 'cpu')
        anneal = p.get('anneal', 'lin')
        method = p.get('method', 'fem')
        map_type = p.get('map_type', 'clique')

        # --- Build coupling matrix from coarse hyperedges ---
        coarse_coupling = hyperedge_list_to_coupling(
            coarse_hyperedges, num_coarse, map_type=map_type,
        )
        if map_type == 'star':
            # Star expansion adds extra auxiliary nodes — we need to handle
            # the extended coupling matrix and extend node_weights.
            num_coarse = coarse_coupling.shape[0]
            extra = num_coarse - len(coarse_node_weights)
            cw = coarse_node_weights
            if torch.is_tensor(cw):
                cw = cw.cpu().numpy()
            cw = np.concatenate([cw, np.ones(extra, dtype=np.float32)])
            coarse_node_weights = torch.tensor(cw, dtype=torch.float32)

        if coarse_coupling.is_sparse:
            num_coupling = coarse_coupling._nnz() // 2
        else:
            num_coupling = int(torch.count_nonzero(coarse_coupling).item() // 2)

        # --- FEM path ---
        if method == 'fem':
            fem = _FEM.from_couplings(
                'bmincut_weighted', num_coarse, num_coupling,
                coarse_coupling, node_weights=coarse_node_weights,
            )
            fem.set_up_solver(
                num_trials, num_steps, dev=dev,
                q=max(2, int(q)), anneal=anneal,
            )
            configs, results = fem.solve()
            best_idx = int(torch.argmin(results).item())
            assignment = configs[best_idx].argmax(dim=1).cpu().numpy().astype(np.int64)
            # If star expansion, strip auxiliary node assignments
            if map_type == 'star':
                assignment = assignment[:len(coarse_node_weights) - extra]
            return assignment

        # --- PUBO path ---
        elif method == 'pubo':
            pubo_obj = _Q4PUBOWrapper(
                coarse_hyperedges, coarse_node_weights, q,
                num_coarse, imbalance_weight=5.0,
            )
            dummy = torch.zeros((num_coarse, num_coarse))
            case = _FEM()
            case.set_up_problem(
                num_coarse, 0, 'customize', dummy,
                q=q, customize_expected_func=pubo_obj.expectation,
                customize_infer_func=pubo_obj.inference,
            )
            case.set_up_solver(
                num_trials, num_steps, dev=dev,
                q=q, manual_grad=False, anneal=anneal,
            )
            configs, results = case.solve()
            best = configs[0].argmax(dim=1).cpu().numpy().astype(np.int64)
            return best

        else:
            raise ValueError(f"Unknown method '{method}' — use 'fem' or 'pubo'")


class _Q4PUBOWrapper:
    """Minimal PUBO wrapper for coarse hypergraph (q=4 cut-net)."""
    def __init__(self, hyperedges, node_weights, q, num_nodes, imbalance_weight=5.0):
        self.hyperedges = hyperedges
        self.node_weights = node_weights
        self.q = q
        self.num_nodes = num_nodes
        self.imbalance_weight = imbalance_weight

    def expectation(self, _, p):
        from src.fem.problem import weighted_imbalance_penalty
        from src.partition.hyper_utils import evaluate_kahypar_cut_value
        batch = p.shape[0]
        total = 0.0
        for b in range(batch):
            assign = p[b].argmax(dim=1).cpu().numpy()
            cut, _ = evaluate_kahypar_cut_value(
                assign, self.hyperedges,
                hyperedge_weights=[1.0] * len(self.hyperedges),
            )
            total = total + cut
        total = total / batch
        imb = self.imbalance_weight * weighted_imbalance_penalty(
            p, self.node_weights)
        return total + imb

    def inference(self, _, p):
        config = torch.zeros_like(p)
        config.scatter_(2, p.argmax(dim=2, keepdim=True), 1)
        return config, torch.zeros(config.shape[0], device=p.device)


# ── Refinement solver ────────────────────────────────────────────────────


class HyperRefineSolver(HyperSolverBase):
    """Local refinement on the original hypergraph.

    When ``mode_cycle=('flow',)`` (default), runs simple FM (greedy
    incremental refinement).  For other cycles (e.g. ``('mcts', 'flow')``)
    runs the full hybrid pipeline (MCTS / evolution / flow).

    Parameters (via ``update_params`` or ``**overrides``):
        mode_cycle      — tuple of modes: ('flow',) for FM, or hybrid
        rounds          — number of hybrid rounds (ignored for pure flow)
        flow_passes     — FM passes per flow stage
        max_imbalance   — balance constraint (default 0.05)
        repair_balance  — if True, actively repair balance after refinement
                          using ``_repair_balance_fast`` (default True).
                          Hybrid mode always repairs; this flag controls
                          the simple FM path.
        node_weights    — per-vertex weights for weighted balance (default None)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(config_dir)
        self._config['mode_cycle'] = ('flow',)

    def refine(self, assignment, hyperedges, q, node_weights=None, **overrides):
        p = {**self._config, **overrides}
        mode_cycle = p.get('mode_cycle', ('flow',))
        repair = p.get('repair_balance', True)

        # Simple FM mode
        if mode_cycle == ('flow',):
            return _refine_flow(
                assignment, hyperedges, q,
                max_passes=p.get('flow_passes', 5),
                max_imbalance=p.get('max_imbalance', 0.05),
                repair_balance=repair,
                verbose=p.get('verbose', False),
                node_weights=node_weights,
            )

        # Hybrid mode (MCTS / evolution / flow)
        return _refine_hybrid(
            assignment, hyperedges, q,
            mode_cycle=mode_cycle,
            rounds=p.get('rounds', 3),
            max_imbalance=p.get('max_imbalance', 0.05),
            flow_passes=p.get('flow_passes', 3),
            mcts_rollouts=p.get('mcts_rollouts', 16),
            mcts_depth=p.get('mcts_depth', 3),
            evolution_population=p.get('evolution_population', 8),
            evolution_generations=p.get('evolution_generations', 5),
            evolution_mutation=p.get('evolution_mutation', 0.1),
            skip_exploration_if_good=p.get('skip_exploration_if_good', True),
            verbose=p.get('verbose', False),
            node_weights=node_weights,
        )


# ═════════════════════════════════════════════════════════════════════════
#  Refine helpers (moved from src/partition/hyper_refine.py)
# ═════════════════════════════════════════════════════════════════════════


def _refine_flow(assignment, hyperedges, q, max_passes=5, max_imbalance=0.05,
                 repair_balance=True, verbose=False, node_weights=None):
    """Simple FM (greedy incremental) refinement.

    Parameters
    ----------
    repair_balance : bool
        If True, actively repair balance via ``_repair_balance_fast``
        after FM refinement finishes.  This mirrors the behaviour of the
        hybrid pipeline and ensures the result satisfies ``max_imbalance``.
    node_weights : np.ndarray or None
        Per-vertex weights for weighted balance computation.

    Note
    ----
    The underlying ``greedy_refine_hypergraph_incremental`` (C-extension)
    may not support ``node_weights`` yet.  If so, the Python-side balance
    repair logic (below) strictly uses the weights, but the greedy moves
    inside the extension still treat every vertex as weight-1.
    """
    from src.partition.hyper_utils import greedy_refine_hypergraph_incremental
    if verbose:
        print(f"[refine:flow] start max_passes={max_passes} max_imbalance={max_imbalance}")
    result = greedy_refine_hypergraph_incremental(
        assignment, hyperedges,
        hyperedge_weights=[1.0] * len(hyperedges),
        q=q, max_passes=max_passes, max_imbalance=max_imbalance, node_weights=node_weights,
    )
    _, _, current_imb = _partition_summary(result, q=q, node_weights=node_weights)
    if repair_balance and current_imb > max_imbalance:
        result = _repair_balance_fast(
            result, hyperedges, max_imbalance=max_imbalance, q=q,
            node_weights=node_weights,
        )
        if verbose:
            _, _, imb = _partition_summary(result, q=q, node_weights=node_weights)
            print(f"[refine:flow] balance repair applied, imb={imb:.4f}")
    return result


def _target_counts(n, q):
    if q <= 0:
        return np.zeros(0, dtype=int)
    base = n // q
    remainder = n % q
    return np.array([base + (1 if i < remainder else 0) for i in range(q)], dtype=int)


def _balance_limits(assignment, max_imbalance, q=None, node_weights=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if q is None:
        q = int(assignment.max()) + 1 if assignment.size else 2
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
        counts = np.zeros(q, dtype=np.float64)
        np.add.at(counts, assignment, node_weights)
        total = float(node_weights.sum())
    else:
        counts = np.bincount(assignment, minlength=q).astype(np.float64)
        total = float(assignment.size)
    ideal = total / float(q) if q > 0 else 0.0
    max_size = ideal * (1.0 + float(max_imbalance)) if total > 0 else 0.0
    min_size = ideal * (1.0 - float(max_imbalance)) if total > 0 else 0.0
    return q, counts, min_size, max_size


def _repair_balance_fast(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None,
                         node_weights=None, max_iterations_mult=1):
    """Fast balance repair without cut evaluation.

    If ``node_weights`` is provided, balance is computed and repaired
    using weighted block sums instead of raw vertex counts.

    Parameters
    ----------
    max_iterations_mult : int
        Multiplier on the default iteration count (``assignment.size * 2``).
        Use >1 when coarse vertices have heavy weights that make single
        moves larger, requiring more passes to converge.
    """
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    if assignment.size == 0:
        return assignment
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
    q, counts, min_size, max_size = _balance_limits(
        assignment, max_imbalance, q=q, node_weights=node_weights,
    )
    node_degree = np.zeros(assignment.size, dtype=float)
    for he in hyperedges:
        for v in he:
            if 0 <= v < assignment.size:
                node_degree[v] += 1.0
    w = (lambda v: node_weights[v]) if node_weights is not None else (lambda v: 1.0)
    max_iters = max(1, assignment.size * 2 * max_iterations_mult)
    for _ in range(max_iters):
        over = np.where(counts > max_size)[0]
        if len(over) == 0:
            break
        under = np.where(counts < min_size)[0]
        if len(under) == 0:
            under = np.array([int(np.argmin(counts))], dtype=int)
        donor = int(over[np.argmax(counts[over] - max_size)])
        donor_vertices = np.where(assignment == donor)[0]
        if donor_vertices.size == 0:
            break
        rng.shuffle(donor_vertices)
        donor_vertices = donor_vertices[np.argsort(node_degree[donor_vertices], kind='mergesort')]
        moved = False
        for v in donor_vertices:
            vw = w(v)
            for g in under:
                g = int(g)
                if g == donor:
                    continue
                if counts[g] + vw > max_size:
                    continue
                assignment[v] = g
                counts[donor] -= vw
                counts[g] += vw
                moved = True
                break
            if moved:
                break
        if not moved:
            # Fallback: find the lightest vertex that doesn't overshoot,
            # or if none exists, the lightest vertex overall.
            g = int(np.argmin(counts))
            best_v = int(donor_vertices[0])
            best_vw = w(best_v)
            for v in donor_vertices:
                vw = w(v)
                if counts[g] + vw <= max_size:
                    best_v = int(v)
                    best_vw = vw
                    break
                if vw < best_vw:
                    best_v = int(v)
                    best_vw = vw
            assignment[best_v] = g
            counts[donor] -= best_vw
            counts[g] += best_vw
    return assignment


def _repair_balance(assignment, hyperedges, max_imbalance=0.05, seed=None, q=None,
                    node_weights=None):
    """Cut-aware balance repair using FM-style O(deg(v)) delta evaluation.

    Pre-computes ``he_pins`` (shape ``[num_hyperedges, q]``) and ``node_to_he``,
    then evaluates candidate moves via ``move_gain`` instead of calling
    ``evaluate_kahypar_cut_value`` on a full copy of the assignment.
    """
    rng = np.random.default_rng(seed)
    assignment = np.asarray(assignment, dtype=np.int64).copy()
    num_nodes = len(assignment)
    if num_nodes == 0:
        return assignment
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
    q, counts, min_size, max_size = _balance_limits(
        assignment, max_imbalance, q=q, node_weights=node_weights,
    )
    if node_weights is not None:
        total_weight = float(node_weights.sum())
        targets = np.full(q, total_weight / float(q) if q > 0 else 0.0, dtype=np.float64)
    else:
        targets = _target_counts(len(assignment), q).astype(np.float64)
    w = (lambda v: node_weights[v]) if node_weights is not None else (lambda v: 1.0)

    # ── Build tracking structures (FM-style) ─────────────────────────────
    he_pins = np.zeros((len(hyperedges), q), dtype=np.int32)
    node_to_he = [[] for _ in range(num_nodes)]
    for e_idx, he in enumerate(hyperedges):
        for v in he:
            if v < num_nodes:
                he_pins[e_idx][assignment[v]] += 1
                node_to_he[v].append(e_idx)

    # ── Compute base cut ─────────────────────────────────────────────────
    hyperedge_weights = [1.0] * len(hyperedges)
    current_cut = 0.0
    for e_idx in range(len(hyperedges)):
        num_groups = int(np.count_nonzero(he_pins[e_idx]))
        if num_groups > 1:
            current_cut += (num_groups - 1) * hyperedge_weights[e_idx]

    for _ in range(max(1, assignment.size)):
        over = np.where(counts > targets)[0]
        under = np.where(counts < targets)[0]
        if len(over) == 0 or len(under) == 0:
            break
        moved = False
        candidates = np.where(np.isin(assignment, over))[0]
        rng.shuffle(candidates)
        for v in candidates:
            vw = w(v)
            old = int(assignment[v])
            best_g = None
            best_gain = -float('inf')
            for g in under:
                g = int(g)
                if g == old:
                    continue
                if counts[g] + vw > targets[g]:
                    continue
                # FM-style delta evaluation — O(deg(v)), no copy needed
                gain = 0.0
                for e_idx in node_to_he[v]:
                    pins = he_pins[e_idx]
                    wgt = hyperedge_weights[e_idx]
                    if pins[old] == 1:
                        gain += wgt
                    if pins[g] == 0:
                        gain -= wgt
                if gain > best_gain:
                    best_gain = gain
                    best_g = g
            if best_g is not None:
                # ── Apply move ──
                assignment[v] = best_g
                counts[old] -= vw
                counts[best_g] += vw
                current_cut -= best_gain
                # ── Update tracking structures ──
                for e_idx in node_to_he[v]:
                    he_pins[e_idx][old] -= 1
                    he_pins[e_idx][best_g] += 1
                moved = True
                break
        if not moved:
            break
    return assignment


def _partition_summary(assignment, q=None, node_weights=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.size == 0:
        return 0, np.zeros(0, dtype=int), 0.0
    if q is None:
        q = int(assignment.max()) + 1
    if node_weights is not None:
        node_weights = np.asarray(node_weights, dtype=np.float64)
        counts = np.zeros(q, dtype=np.float64)
        np.add.at(counts, assignment, node_weights)
        total = float(node_weights.sum())
    else:
        counts = np.bincount(assignment, minlength=q).astype(np.float64)
        total = float(assignment.size)
    ideal = total / float(q) if q > 0 else 0.0
    imb = float(np.max(np.abs(counts - ideal) / ideal)) if ideal > 0 else 0.0
    return q, counts, imb


def _assignment_cache_key(assignment):
    assignment = np.asarray(assignment, dtype=np.int64)
    return assignment.shape, assignment.tobytes()


def _cached_cut_and_imbalance(assignment, hyperedges, cache=None, node_weights=None):
    from src.partition.hyper_utils import evaluate_kahypar_cut_value
    if cache is None:
        cache = {}
    key = _assignment_cache_key(assignment)
    if key not in cache:
        assignment_arr = np.asarray(assignment, dtype=np.int64)
        cut = evaluate_kahypar_cut_value(assignment_arr, hyperedges, [1.0] * len(hyperedges))[0]
        _, _, imb = _partition_summary(assignment_arr, node_weights=node_weights)
        cache[key] = (float(cut), float(imb))
    return cache[key]


def _refine_mcts(assignment, hyperedges, q, num_rollouts=16, depth=3, seed=None,
                 max_imbalance=0.05, verbose=False, metrics_cache=None,
                 node_weights=None):
    """Monte-Carlo style refinement via randomized move simulations."""
    rng = np.random.default_rng(seed)
    best = np.asarray(assignment, dtype=np.int64).copy()
    base = best.copy()
    if metrics_cache is None:
        metrics_cache = {}
    best_score, _ = _cached_cut_and_imbalance(best, hyperedges, metrics_cache, node_weights=node_weights)
    q = int(q) if q is not None else (int(best.max()) + 1 if best.size else 2)
    if best.size:
        node_to_he = [[] for _ in range(best.size)]
        for e_idx, he in enumerate(hyperedges):
            for v in he:
                if 0 <= v < best.size:
                    node_to_he[v].append(e_idx)
    if verbose:
        _, _, imb = _partition_summary(best, q=q, node_weights=node_weights)
        print(f"[refine:mcts] start rollouts={num_rollouts} depth={depth} cut={best_score} imb={imb:.4f}")

    # Calculate boundary vertices ONCE before rollouts begin
    boundary_vertices = []
    for v in range(best.size):
        for e_idx in node_to_he[v]:
            if len(set(best[u] for u in hyperedges[e_idx] if 0 <= u < best.size)) > 1:
                boundary_vertices.append(v)
                break

    if not boundary_vertices:
        return best

    boundary_vertices = np.asarray(boundary_vertices, dtype=np.int64)

    for _ in range(max(1, int(num_rollouts))):
        cand = best.copy()
        for _step in range(max(1, int(depth))):
            v = int(boundary_vertices[int(rng.integers(0, boundary_vertices.size))])
            old = int(cand[v])
            new_g = int(rng.integers(0, q - 1))
            if new_g >= old:
                new_g += 1
            if new_g != old:
                cand[v] = new_g
        score, _ = _cached_cut_and_imbalance(cand, hyperedges, metrics_cache, node_weights=node_weights)
        if score < best_score:
            best_score = score
            best = cand
    if _partition_summary(best, q=q, node_weights=node_weights)[2] > max_imbalance:
        best = _repair_balance_fast(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                     node_weights=node_weights)
    if verbose:
        _, _, imb = _partition_summary(best, q=q, node_weights=node_weights)
        print(f"[refine:mcts] done cut={best_score} imb={imb:.4f}")
    return best


def _refine_evolution(assignment, hyperedges, q, population_size=8, generations=5,
                      mutation_rate=0.1, seed=None, max_imbalance=0.05,
                      verbose=False, metrics_cache=None, node_weights=None):
    """Small evolutionary search over discrete assignments."""
    rng = np.random.default_rng(seed)
    base = np.asarray(assignment, dtype=np.int64)
    if metrics_cache is None:
        metrics_cache = {}
    q = int(q) if q is not None else (int(base.max()) + 1 if base.size else 2)
    base_score, _ = _cached_cut_and_imbalance(base, hyperedges, metrics_cache, node_weights=node_weights)
    _, _, base_imb = _partition_summary(base, q=q, node_weights=node_weights)
    low_cut_mode = base_score < 200
    if low_cut_mode:
        mutation_rate = min(float(mutation_rate), 0.01)
        generations = min(int(generations), 3)
    if verbose:
        print(f"[refine:evolution] start pop={population_size} gens={generations} cut={base_score} imb={base_imb:.4f}")
    population = [base.copy() for _ in range(max(1, int(population_size)))]
    mutant_count = max(1, len(population) // 4)
    for idx in range(1, min(len(population), mutant_count + 1)):
        cand = base.copy()
        mask = rng.random(cand.shape[0]) < float(mutation_rate)
        if mask.any():
            cand[mask] = rng.integers(0, q, size=int(mask.sum()))
            if _partition_summary(cand, q=q, node_weights=node_weights)[2] > max_imbalance:
                cand = _repair_balance_fast(cand, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                             node_weights=node_weights)
        population[idx] = cand
    for _gen in range(max(1, int(generations))):
        scored = []
        for cand in population:
            score, _ = _cached_cut_and_imbalance(cand, hyperedges, metrics_cache, node_weights=node_weights)
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] > base_score:
            scored = [(base_score, base.copy())] + scored
        elites = [base.copy()]
        elites.extend(cand.copy() for _, cand in scored[: max(1, len(scored) // 3)])
        next_population = elites[:]
        while len(next_population) < len(population):
            p1 = elites[int(rng.integers(0, len(elites)))]
            p2 = elites[int(rng.integers(0, len(elites)))]
            child = np.where(rng.random(base.shape[0]) < 0.5, p1, p2).copy()
            mut_mask = rng.random(child.shape[0]) < float(mutation_rate)
            if mut_mask.any():
                child[mut_mask] = rng.integers(0, q, size=int(mut_mask.sum()))
                if _partition_summary(child, q=q, node_weights=node_weights)[2] > max_imbalance:
                    child = _repair_balance_fast(child, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                                 node_weights=node_weights)
            next_population.append(child)
        population = next_population
    scored = [(_cached_cut_and_imbalance(cand, hyperedges, metrics_cache, node_weights=node_weights)[0], cand) for cand in population]
    scored.sort(key=lambda x: x[0])
    best_score, best = scored[0]
    if best_score > base_score:
        best = base.copy()
        best_score = base_score
    if _partition_summary(best, q=q, node_weights=node_weights)[2] > max_imbalance:
        best = _repair_balance_fast(best, hyperedges, max_imbalance=max_imbalance, seed=seed, q=q,
                                     node_weights=node_weights)
    if verbose:
        _, _, imb = _partition_summary(best, q=q, node_weights=node_weights)
        print(f"[refine:evolution] done cut={best_score} imb={imb:.4f}")
    return best


def _refine_hybrid(
    assignment, hyperedges, q,
    mode_cycle=('mcts', 'evolution', 'flow'),
    rounds=3, seed=None, max_imbalance=0.05,
    flow_passes=3, mcts_rollouts=16, mcts_depth=3,
    evolution_population=8, evolution_generations=5, evolution_mutation=0.1,
    skip_exploration_if_good=True, verbose=False,
    node_weights=None,
):
    """Hybrid refinement pipeline (MCTS / evolution / flow).

    Parameters
    ----------
    node_weights : np.ndarray or None
        Per-vertex weights for weighted balance computation.
    """
    refined = np.asarray(assignment, dtype=np.int64).copy()
    if q is None:
        q = int(refined.max()) + 1 if refined.size else 2

    metrics_cache = {}

    def evaluate(candidate):
        return _cached_cut_and_imbalance(candidate, hyperedges, metrics_cache,
                                          node_weights=node_weights)

    def ensure_balanced(candidate):
        candidate = np.asarray(candidate, dtype=np.int64).copy()
        if _partition_summary(candidate, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return candidate

        # ── Attempt 1: cut-aware repair ──
        repaired = _repair_balance(candidate, hyperedges, max_imbalance=max_imbalance,
                                    seed=seed, q=q, node_weights=node_weights)
        if _partition_summary(repaired, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return repaired

        # ── Attempt 2: fast repair (more passes) + cut-aware repair ──
        repaired = _repair_balance_fast(
            repaired, hyperedges, max_imbalance=max_imbalance,
            seed=seed, q=q, node_weights=node_weights,
            max_iterations_mult=10,  # more aggressive with heavy coarse vertices
        )
        repaired = _repair_balance(repaired, hyperedges, max_imbalance=max_imbalance,
                                    seed=seed, q=q, node_weights=node_weights)
        if _partition_summary(repaired, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return repaired

        # ── Attempt 3: fast repair only (most aggressive) ──
        repaired = _repair_balance_fast(
            repaired, hyperedges, max_imbalance=max_imbalance,
            seed=seed, q=q, node_weights=node_weights,
            max_iterations_mult=50,
        )
        if _partition_summary(repaired, q=q, node_weights=node_weights)[2] <= max_imbalance:
            return repaired

        # ── Fallback: best effort — warn but do not crash.
        # Intermediate V-cycle levels will be re-refined at the next finer
        # level anyway, so a minor balance violation is tolerable.
        if verbose:
            _, _, imb = _partition_summary(repaired, q=q, node_weights=node_weights)
            print(f"  [warn] ensure_balanced: best imb={imb:.4f} > {max_imbalance}, continuing")
        return repaired

    if verbose:
        cut, _ = evaluate(refined)
        _, counts, imb = _partition_summary(refined, q=q, node_weights=node_weights)
        print(f"[refine:hybrid] start q={q} cut={cut} counts={counts.tolist()} imb={imb:.4f}")

    refined = ensure_balanced(refined)
    best = refined.copy()
    best_cut, best_imb = evaluate(best)

    def maybe_repair_and_accept(candidate):
        nonlocal best, best_cut, best_imb
        cand = np.asarray(candidate, dtype=np.int64).copy()
        cand = ensure_balanced(cand)
        cand_cut, cand_imb = evaluate(cand)
        if cand_cut < best_cut or (cand_cut == best_cut and cand_imb <= best_imb):
            best = cand.copy()
            best_cut = float(cand_cut)
            best_imb = float(cand_imb)
            return cand
        return best.copy()

    dynamic_good_cut_threshold = max(1.0, 0.1 * float(best_cut))
    good_initial = skip_exploration_if_good and best_cut <= dynamic_good_cut_threshold and best_imb <= float(max_imbalance)
    effective_mode_cycle = ('flow',) if good_initial else tuple(mode_cycle)
    effective_rounds = 1 if good_initial else max(1, int(rounds))
    effective_flow_passes = 1 if good_initial else int(flow_passes)

    for round_idx in range(effective_rounds):
        if verbose:
            print(f"[refine:hybrid] round {round_idx + 1}/{int(effective_rounds)}")
        if 'mcts' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=MCTS")
            candidate = _refine_mcts(
                refined, hyperedges, q,
                num_rollouts=mcts_rollouts, depth=mcts_depth, seed=seed,
                max_imbalance=max_imbalance, verbose=verbose,
                metrics_cache=metrics_cache, node_weights=node_weights,
            )
            refined = maybe_repair_and_accept(candidate)
        if 'evolution' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=Evolution")
            candidate = _refine_evolution(
                refined, hyperedges, q,
                population_size=evolution_population,
                generations=evolution_generations,
                mutation_rate=evolution_mutation, seed=seed,
                max_imbalance=max_imbalance, verbose=verbose,
                metrics_cache=metrics_cache, node_weights=node_weights,
            )
            refined = maybe_repair_and_accept(candidate)
        if 'flow' in effective_mode_cycle:
            if verbose:
                print("[refine:hybrid] stage=Flow")
            candidate = _refine_flow(
                refined, hyperedges, q,
                max_passes=effective_flow_passes,
                max_imbalance=max_imbalance, verbose=verbose,
                node_weights=node_weights,
            )
            refined = maybe_repair_and_accept(candidate)
        if _partition_summary(refined, q=q, node_weights=node_weights)[2] > max_imbalance:
            refined = _repair_balance(refined, hyperedges, max_imbalance=max_imbalance,
                                       seed=seed, q=q, node_weights=node_weights)
        refined = maybe_repair_and_accept(refined)
        if verbose:
            print(f"[refine:hybrid] round_done cut={best_cut} counts={_partition_summary(best, q=q, node_weights=node_weights)[1].tolist()} imb={best_imb:.4f}")

    refined = best.copy()
    if _partition_summary(refined, q=q, node_weights=node_weights)[2] > max_imbalance:
        refined = _repair_balance(refined, hyperedges, max_imbalance=max_imbalance,
                                   seed=seed, q=q, node_weights=node_weights)
    refined = maybe_repair_and_accept(refined)
    if verbose:
        cut, _ = evaluate(refined)
        _, counts, imb = _partition_summary(refined, q=q, node_weights=node_weights)
        print(f"[refine:hybrid] done cut={cut} counts={counts.tolist()} imb={imb:.4f}")
    return refined


# ═════════════════════════════════════════════════════════════════════════
#  Internal helpers (moved from hyper_coarsen.py)
# ═════════════════════════════════════════════════════════════════════════


def _build_incidence(hyperedge_list, vertex_count):
    vertex_to_edges = [set() for _ in range(vertex_count)]
    edge_vertices = []
    edge_weights = []
    for eid, he in enumerate(hyperedge_list):
        verts = []
        seen = set()
        for v in he:
            if 0 <= v < vertex_count and v not in seen:
                verts.append(int(v))
                seen.add(int(v))
        if len(verts) > 1:
            edge_vertices.append(verts)
            edge_weights.append(1.0)
            for v in verts:
                vertex_to_edges[v].add(len(edge_vertices) - 1)
    return vertex_to_edges, edge_vertices, edge_weights


def _rebuild_hyperedges_from_groups(hyperedges, original_to_bucket, bucket_count):
    coarse_hyperedges = []
    for he in hyperedges:
        mapped = []
        seen = set()
        for v in he:
            if 0 <= v < len(original_to_bucket):
                c = int(original_to_bucket[v])
                if c not in seen:
                    mapped.append(c)
                    seen.add(c)
        if len(mapped) > 1:
            coarse_hyperedges.append(mapped)
    return coarse_hyperedges


def _lsh_bucketize_vertices(
    hyperedges, num_nodes, target_buckets=None,
    num_planes=4, num_tables=32, seed=None,
    jaccard_threshold=0.1, num_hashes=128, verbose=False,
):
    """Pre-coarsen vertices using MinHash/LSH over incident hyperedge sets."""
    if num_nodes == 0:
        return np.arange(0, dtype=np.int64), []

    incident_edge_sets = _vertex_incident_edge_sets(hyperedges, num_nodes)

    from src.partition.hyper_coarsen import _lsh_groups_from_incident_sets
    groups = _lsh_groups_from_incident_sets(
        incident_edge_sets,
        num_hashes=num_hashes,
        num_bands=max(1, int(num_tables)),
        rows_per_band=max(1, int(num_planes)),
        target_buckets=max(1, int(target_buckets)) if target_buckets is not None else max(1, int(num_nodes // 4)),
        threshold=float(jaccard_threshold),
        min_threshold=0.02,
        seed=seed,
        verbose=verbose,
    )

    original_to_bucket = np.empty(num_nodes, dtype=np.int64)
    for idx, verts in enumerate(groups):
        for v in verts:
            original_to_bucket[v] = idx
    return original_to_bucket, groups


def _vertex_incident_edge_sets(hyperedges, num_nodes):
    incident = [set() for _ in range(num_nodes)]
    for eid, he in enumerate(hyperedges):
        for v in he:
            if 0 <= v < num_nodes:
                incident[v].add(eid)
    return incident


# ═════════════════════════════════════════════════════════════════════════
#  V-Cycle uncoarsening helper
# ═════════════════════════════════════════════════════════════════════════


def vcycle_uncoarsen(
    coarse_assignment,
    hierarchy_stack,
    original_hyperedges,
    q,
    refine_solver: HyperRefineSolver,
    verbose: bool = True,
) -> np.ndarray:
    """Multilevel V-Cycle: iteratively project and refine through the hierarchy.

    Takes an initial partition on the coarsest level, then walks back up
    through the saved hierarchy levels (finest-first stack).  At each step:

        1. Project the current assignment to the next finer level via ``remap``.
        2. Run ``refine_solver.refine()`` on that finer-level hypergraph.

    After all hierarchy levels are processed, a final refinement pass is
    run on the **original** hypergraph.

    Parameters
    ----------
    coarse_assignment : np.ndarray
        Partition assignment at the coarsest level (output of
        ``FemCoarsenSolver.initial_partition`` or ``initial_partition_greedy``).
        Length must equal the number of coarse nodes.
    hierarchy_stack : list of dict
        The ``hierarchy_stack`` returned by ``KahyparLikeSolver.coarsen()``.
        Each entry has keys ``hyperedges``, ``remap``, ``num_nodes``, ``groups``.
        Ordered from finest (first contraction) to coarsest (last contraction).
    original_hyperedges : list of list of int
        The original (finest-level) hyperedges — used for the final
        refinement step after all hierarchy levels are processed.
    q : int
        Number of blocks (partitions).
    refine_solver : HyperRefineSolver
        Refinement solver (FM / hybrid) to apply at each level.
    verbose : bool
        If True, print cut / imbalance after each level.

    Returns
    -------
    np.ndarray
        Final assignment on the original hypergraph.
    """
    assignment = np.asarray(coarse_assignment, dtype=np.int64).copy()
    n_levels = len(hierarchy_stack)

    # Number of vertices in the original hypergraph (for final unit-weight pass)
    num_original_nodes = max(
        (max(he) for he in original_hyperedges if he), default=-1,
    ) + 1

    # Walk back up: coarsest → finest
    for level_idx, level in enumerate(reversed(hierarchy_stack)):
        fine_hyperedges = level['hyperedges']
        remap = level['remap']
        fine_n = level['num_nodes']

        # ── Extract node weights for this level (cluster sizes) ──
        fine_weights = np.array([len(g) for g in level['groups']], dtype=np.float32)

        # ── Project: fine_assignment[v] = coarse_assignment[remap[v]] ──
        projected = np.array([assignment[remap[v]] for v in range(fine_n)], dtype=np.int64)

        if verbose:
            cut, imb = evaluate_kahypar_cut_value(
                projected, fine_hyperedges,
                hyperedge_weights=[1.0] * len(fine_hyperedges),
            )
            lvl_label = n_levels - level_idx
            print(f'  [V-cycle] level {lvl_label}/{n_levels}: projected  cut={cut}, imb={imb:.4f}')

        # ── Refine at this level (with weighted balance) ──
        assignment = refine_solver.refine(projected, fine_hyperedges, q,
                                          node_weights=fine_weights)

        if verbose:
            cut, imb = evaluate_kahypar_cut_value(
                assignment, fine_hyperedges,
                hyperedge_weights=[1.0] * len(fine_hyperedges),
            )
            lvl_label = n_levels - level_idx
            print(f'  [V-cycle] level {lvl_label}/{n_levels}: refined   cut={cut}, imb={imb:.4f}')

    # ── Final refinement on the original hypergraph (unit weights) ──
    orig_weights = np.ones(num_original_nodes, dtype=np.float32)

    if verbose:
        cut, imb = evaluate_kahypar_cut_value(
            assignment, original_hyperedges,
            hyperedge_weights=[1.0] * len(original_hyperedges),
        )
        print(f'  [V-cycle] original (pre-refine): cut={cut}, imb={imb:.4f}')

    assignment = refine_solver.refine(assignment, original_hyperedges, q,
                                      node_weights=orig_weights)

    if verbose:
        cut, imb = evaluate_kahypar_cut_value(
            assignment, original_hyperedges,
            hyperedge_weights=[1.0] * len(original_hyperedges),
        )
        print(f'  [V-cycle] original (post-refine): cut={cut}, imb={imb:.4f}')

    return assignment
