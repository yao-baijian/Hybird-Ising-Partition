"""
Solver base classes — each solver type becomes a class that loads its own
config once on initialisation and exposes phase methods (coarsen,
initial_partition, refine).

Usage::
    from src.solver_base import FemSolver, SbmSolver, KaffpaSolver

    fem = FemSolver()              # loads config/fem.json
    fem.get_param("num_trials")    # read any config key
    fem.set_param("num_trials", 20)  # override at runtime
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from src.method_registry import load_config


def _default_config_dir() -> Path:
    return Path.cwd() / "config"


# ── Tensor conversion helpers ─────────────────────────────────────────────

def _tensor_to_adj_list(T):
    """Convert a sparse or dense tensor to a weighted adjacency list."""
    n = T.shape[0]
    adj = [[] for _ in range(n)]
    if T.is_sparse:
        T = T.coalesce()
        idxs, vals = T.indices(), T.values()
        for i in range(idxs.shape[1]):
            r, c = int(idxs[0, i]), int(idxs[1, i])
            if r != c:
                adj[r].append((c, float(vals[i].item())))
    else:
        for r in range(n):
            for c in range(n):
                w = T[r, c].item()
                if r != c and w != 0:
                    adj[r].append((c, float(w)))
    return adj


def _tensor_to_adj_list_no_weights(T):
    """Convert a sparse or dense tensor to an unweighted adjacency list."""
    n = T.shape[0]
    adj = [[] for _ in range(n)]
    if T.is_sparse:
        T = T.coalesce()
        idxs = T.indices()
        for i in range(idxs.shape[1]):
            r, c = int(idxs[0, i]), int(idxs[1, i])
            if r != c:
                adj[r].append(c)
    else:
        for r in range(n):
            for c in range(n):
                if r != c and T[r, c].item() != 0:
                    adj[r].append(c)
    return adj


class SolverBase:
    type: str = "base"

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = Path(config_dir) if config_dir else _default_config_dir()
        self._config: Dict[str, Any] = load_config(self.type, self._config_dir)

    def get_param(self, key: str, default=None):
        return self._config.get(key, default)

    def set_param(self, key: str, value):
        self._config[key] = value

    def update_params(self, **kwargs):
        self._config.update(kwargs)

    def get_all_params(self) -> Dict[str, Any]:
        return dict(self._config)

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        raise NotImplementedError

    def refine(self, J, initial_partition, q, **overrides):
        raise NotImplementedError

    def solve_direct(self, J, q, **overrides):
        raise NotImplementedError


class FemSolver(SolverBase):
    type = "fem"

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        import torch
        from src.fem import FEM
        p = {**self._config, **overrides}
        nc = coarse_J.shape[0]
        nnz = coarse_J._nnz() if coarse_J.is_sparse else torch.count_nonzero(coarse_J).item()
        case = FEM.from_couplings("bmincut", nc, int(nnz // 2),
                                   coarse_J, node_weights=coarse_weights)
        case.set_up_solver(p.get("num_trials", 10), p.get("num_steps", 1000),
                            anneal=p.get("anneal", "lin"), dev=p.get("dev", "cpu"),
                            q=q, manual_grad=p.get("manual_grad", False),
                            use_compile=p.get("use_compile", False))
        config, result = case.solve()
        best = config[torch.argwhere(result == result.min()).reshape(-1)[0]]
        return best.argmax(dim=1).cpu().numpy()

    def solve_direct(self, case_type, instance, index_start, q, **overrides):
        import time
        from src.fem import FEM
        p = {**self._config, **overrides}
        case = FEM.from_file(case_type, instance, index_start)
        case.set_up_solver(p.get("num_trials", 10), p.get("num_steps", 1000),
                            anneal=p.get("anneal", "lin"), dev=p.get("dev", "cpu"),
                            q=q, manual_grad=p.get("manual_grad", False),
                            use_compile=p.get("use_compile", False))
        t0 = time.perf_counter()
        config, result = case.solve()
        dt = time.perf_counter() - t0
        out = config[torch.argwhere(result == result.min()).reshape(-1)[0]]
        return out, result.min().item(), dt


class SbmSolver(SolverBase):
    type = "sbm"

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        from src.sbm.sbm import bsb_bmincut_batch
        p = {**self._config, **overrides}
        nc = coarse_J.shape[0]
        nt = p.get("num_trials", 10)
        ns = p.get("num_steps", 1000)
        dt = p.get("dt", 0.1)
        lb = p.get("lambda_balance", 1.0)
        uc = p.get("use_compile", False)
        dev = p.get("dev", "cpu")

        if q == 2:
            ix = 2 * torch.rand(nt, nc, device=dev) - 1
            iy = 2 * torch.rand(nt, nc, device=dev) - 1
            _, sol, cuts, _ = bsb_bmincut_batch(coarse_J, ix, iy, ns, dt,
                                                  lambda_balance=lb, use_compile=uc)
            return np.where(sol[torch.argmin(cuts)].cpu().numpy() == 1, 0, 1)
        else:
            part = torch.zeros(nc, dtype=torch.long, device=dev)
            n_p, nxt = 1, 1
            while n_p < q:
                sizes = torch.bincount(part, minlength=n_p)
                largest = int(torch.argmax(sizes))
                mask = part == largest
                sub = torch.where(mask)[0]
                if len(sub) <= 1:
                    break
                Jsub = coarse_J[sub][:, sub]
                bs = max(nt, 5)
                ix = 2 * torch.rand(bs, len(sub), device=dev) - 1
                iy = 2 * torch.rand(bs, len(sub), device=dev) - 1
                _, sol, cuts, _ = bsb_bmincut_batch(Jsub, ix, iy, max(ns // n_p, 50),
                                                      dt, lambda_balance=lb, use_compile=uc)
                part[sub[sol[torch.argmin(cuts)] == -1]] = nxt
                nxt += 1
                n_p += 1
            return part.cpu().numpy()

    def solve_direct(self, case_type, instance, index_start, q, **overrides):
        import time
        from src.sbm.sbm import bsb_bmincut_batch
        from src.fem import FEM
        from src.fem.problem import infer_bmincut
        p = {**self._config, **overrides}
        case = FEM.from_file(case_type, instance, index_start)
        J = case.problem.coupling_matrix.to(p.get("dev", "cpu"))
        n = J.shape[0]
        t0 = time.perf_counter()
        dt_sbm = p.get("dt", 0.1)
        nt = p.get("num_trials", 10)
        ns = p.get("num_steps", 1000)
        uc = p.get("use_compile", False)
        dev = p.get("dev", "cpu")

        if q == 2:
            ix = 2 * torch.rand(nt, n, device=dev) - 1
            iy = 2 * torch.rand(nt, n, device=dev) - 1
            _, sol, cuts, _ = bsb_bmincut_batch(J, ix, iy, ns, dt_sbm,
                                                  lambda_balance=1.0, use_compile=uc)
            spins = sol[torch.argmin(cuts)]
            pmat = torch.zeros(n, q, device=dev)
            pmat[spins == 1, 0] = 1.0
            pmat[spins == -1, 1] = 1.0
        else:
            part = torch.zeros(n, dtype=torch.long, device=dev)
            n_p, nxt = 1, 1
            while n_p < q:
                sizes = torch.bincount(part, minlength=n_p)
                largest = int(torch.argmax(sizes))
                mask = part == largest
                sub = torch.where(mask)[0]
                if len(sub) <= 1:
                    break
                Jsub = J[sub][:, sub]
                bs = max(nt, 5)
                ix = 2 * torch.rand(bs, len(sub), device=dev) - 1
                iy = 2 * torch.rand(bs, len(sub), device=dev) - 1
                _, sol, cuts, _ = bsb_bmincut_batch(Jsub, ix, iy, max(ns // n_p, 50),
                                                      dt_sbm, lambda_balance=1.0, use_compile=uc)
                part[sub[sol[torch.argmin(cuts)] == -1]] = nxt
                nxt += 1
                n_p += 1
            pmat = torch.zeros(n, q, device=dev)
            for i in range(n):
                pmat[i, min(int(part[i].item()), q - 1)] = 1.0

        dt = time.perf_counter() - t0
        _, cut = infer_bmincut(J, pmat.unsqueeze(0))
        return pmat, cut.item(), dt


class KaffpaSolver(SolverBase):
    type = "kaffpa"

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        """Greedy+FM initial partition on a coarse graph."""
        from src.partition.kaffpa_multiway import initial_partition_greedy_fm
        nc = coarse_J.shape[0]
        adj = _tensor_to_adj_list(coarse_J)
        p = {**self._config, **overrides}
        vwgt = [1] * nc
        _, part = initial_partition_greedy_fm(
            adj, vwgt, q,
            num_trials=p.get("num_init_trials", 5),
        )
        return np.array(part)

    def refine(self, J, initial_partition, q, **overrides):
        """FM-style local refinement on the full graph (CSR format)."""
        from src.partition.refine import simple_kaffpa
        n = J.shape[0]
        p = {**self._config, **overrides}

        vwgt = [1] * n
        # Build CSR from adjacency list built from tensor
        adj = _tensor_to_adj_list(J)
        xadj, adjncy, adjcwgt = [0], [], []
        for i in range(n):
            for c, w in adj[i]:
                adjncy.append(c)
                adjcwgt.append(int(w))
            xadj.append(len(adjncy))

        _, refined = simple_kaffpa(
            vwgt, xadj, adjcwgt, adjncy, q,
            epsilon=p.get("epsilon", 0.05),
            part=list(initial_partition),
            max_passes=p.get("refine_passes", 10),
        )
        return np.array(refined)

    def solve_direct(self, J, q, **overrides):
        import time
        from src.partition.kaffpa_multiway import kaffpa_multiway_kway
        p = {**self._config, **overrides}
        t0 = time.perf_counter()
        result = kaffpa_multiway_kway(J, q, coarsen_to=p.get("coarsen_to", 50),
                                       epsilon=p.get("epsilon", 0.05),
                                       refine_passes=p.get("refine_passes", 10))
        dt = time.perf_counter() - t0
        # result is (p, cut, tc, ti, tr) — pack time into the tuple
        pmat, cut, tc, ti, tr = result
        return pmat, cut, dt


class MetisSolver(SolverBase):
    type = "metis"

    def initial_partition(self, coarse_J, coarse_weights, coarse_groups, q, **overrides):
        """METIS initial partition on a coarse graph."""
        import pymetis
        adj = _tensor_to_adj_list_no_weights(coarse_J)
        _, parts = pymetis.part_graph(q, adjacency=adj)
        return np.array(parts)

    def refine(self, J, initial_partition, q, **overrides):
        from src.partition.refine import call_pymetis_with_part
        adj = _tensor_to_adj_list_no_weights(J)
        _, parts = call_pymetis_with_part(q, adj, part=initial_partition.tolist())
        return np.array(parts)
    def solve_direct(self, J, q, **overrides):
        import time, pymetis
        from src.fem.problem import infer_bmincut
        n = J.shape[0]
        adj = _tensor_to_adj_list_no_weights(J)
        t0 = time.perf_counter()
        _, parts = pymetis.part_graph(q, adjacency=adj)
        dt = time.perf_counter() - t0
        pmat = torch.zeros(n, q, dtype=J.dtype, device=J.device)
        for i, g in enumerate(parts):
            pmat[i, g] = 1.0
        _, cut = infer_bmincut(J, pmat.unsqueeze(0))
        return pmat, cut.item(), dt


class CyclicSolver(SolverBase):
    type = "cyclic"

    def refine(self, J, initial_partition, q, **overrides):
        from src.fem.cyclic_expansion import cyclic_expansion_refine, adjacency_from_sparse
        p = {**self._config, **overrides}
        adj = adjacency_from_sparse(J)
        return cyclic_expansion_refine(adj, initial_partition, q,
            max_iterations=p.get("max_iterations", 50),
            max_candidates=p.get("max_candidates", 60),
            num_trials=p.get("num_trials", 5),
            num_steps=p.get("num_steps_cyclic", 100),
            dev=p.get("dev", "cpu"),
            patience=p.get("patience", 10),
            verbose=p.get("verbose", False),
            allow_nonadjacent=p.get("allow_nonadjacent", True))


# ── Solver cache ──────────────────────────────────────────────────────────

_solver_cache: Dict[str, SolverBase] = {}


def get_solver(type_name: str, config_dir=None) -> SolverBase:
    """Return a cached solver instance (one per type, loads config once)."""
    if type_name not in _solver_cache:
        for cls in SolverBase.__subclasses__():
            if cls.type == type_name:
                _solver_cache[type_name] = cls(config_dir=config_dir)
                break
        else:
            raise ValueError(f"Unknown solver type: {type_name}")
    return _solver_cache[type_name]


# ── Composite pipeline runner ─────────────────────────────────────────────


def run_composite_method(J, q, init_solver: SolverBase,
                          refine_solver: SolverBase, **overrides):
    """Run a hybrid pipeline using pre-configured solver instances.

    Returns
    -------
    (p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds)
    """
    import time
    from src.partition.coarsen import coarsen_graph_by_matching, expand_coarse_labels
    from src.fem.problem import infer_bmincut

    coarsen_to = overrides.get("coarsen_to", init_solver.get_param("coarsen_to", 50))

    t0 = time.perf_counter()
    coarse_J, cw, groups, *_ = coarsen_graph_by_matching(J, coarsen_to=coarsen_to)
    # Solvers expect dense matrices for QUBO arithmetic (sparse + dense not supported)
    if coarse_J.is_sparse:
        coarse_J = coarse_J.to_dense()
    coarsen_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    coarse_part = init_solver.initial_partition(coarse_J, cw, groups, q, **overrides)
    init_partition_time_s = time.perf_counter() - t1

    fine_part = expand_coarse_labels(groups, coarse_part, J.shape[0])

    t2 = time.perf_counter()
    final_part = refine_solver.refine(J, fine_part, q, **overrides)
    refine_time_s = time.perf_counter() - t2

    n = J.shape[0]
    p = torch.zeros(n, q, dtype=J.dtype, device=J.device)
    for i in range(n):
        p[i, int(final_part[i])] = 1.0

    _, cut = infer_bmincut(J, p.unsqueeze(0))
    return p, cut.item(), coarsen_time_s, init_partition_time_s, refine_time_s, 0
