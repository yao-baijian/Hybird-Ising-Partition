"""
Hypergraph initial-partition comparison test using solver mode.

Pipeline:
  1. KahyparLikeSolver coarsens the hypergraph once (HEM matching).
  2. The SAME coarse result is fed to two different initial partition
     strategies: greedy (built into KahyparLikeSolver) and FEM-based
     (FemCoarsenSolver).
  3. Both assignments are projected back and optionally FM-refined.

Configurable parameters at the top of this file:
  coarsen_to      — target coarse nodes (controls KahyparLikeSolver)
  fem_method      — ``'fem'`` (default) or ``'pubo'``
  fem_map_type    — ``'clique'`` (default) or ``'star'`` expansion
  fem_anneal      — anneal schedule for FEM solver (default ``'lin'``)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyper_solver import KahyparLikeSolver, FemCoarsenSolver, HyperRefineSolver
from src.partition.coarsen import expand_coarse_labels
from src.partition.hyper_utils import evaluate_kahypar_cut_value
from utils import parse_hypergraph_edges

import numpy as np

# ── Configurable parameters ──────────────────────────────────────────────

coarsen_to = 50          # target coarse nodes (all methods)
q = 4                    # number of partitions
refine_passes = 5        # FM refinement passes
verbose = True

# FemCoarsenSolver options (see class docstring for details)
fem_method = 'fem'       # 'fem' or 'pubo'
fem_map_type = 'star'  # 'clique' or 'star'
fem_anneal = 'lin'       # anneal schedule for FEM solver

# ── Solver instances ─────────────────────────────────────────────────────
# KahyparLikeSolver  → coarsening + greedy initial partition
# FemCoarsenSolver   → FEM/PUBO initial partition on coarsened graph
# refine_solver      → FM refinement (mode_cycle=('flow',))

kahypar_solver = KahyparLikeSolver()
kahypar_solver.update_params(
    coarsen_to=coarsen_to,
    verbose=verbose,
)

fem_solver = FemCoarsenSolver()
fem_solver.update_params(
    num_trials=1,
    num_steps=1000,
    dev='cpu',
    method=fem_method,
    map_type=fem_map_type,
    anneal=fem_anneal,
)

refine_solver = HyperRefineSolver()
refine_solver.update_params(
    mode_cycle=('flow',),
    repair_balance=True,
    flow_passes=refine_passes,
    max_imbalance=0.05,
)


def _load_hypergraph():
    instance = '../partition/full_benchmark_set/powersim.mtx.hgr'
    hyperedges = parse_hypergraph_edges(str(instance))
    num_nodes = max((max(h) for h in hyperedges if h), default=-1) + 1
    return hyperedges, num_nodes


def _report(name, assignment, hyperedges, num_nodes):
    """Evaluate and print cut / imbalance for an assignment."""
    cut, imb = evaluate_kahypar_cut_value(
        assignment, hyperedges, hyperedge_weights=[1.0] * len(hyperedges),
    )
    print(
        f'{name}  cut = {cut}, imbalance = {imb:.15f}'
    )
    return cut, imb


def test_compare_initial_partition_modes():
    hyperedges, num_nodes = _load_hypergraph()

    # ── Phase 1: Coarsen ONCE ────────────────────────────────────────────
    res = kahypar_solver.coarsen(hyperedges, num_nodes, q)
    coarse_nodes = len(res['coarse_groups'])
    print(f'Coarsened: {num_nodes} -> {coarse_nodes} nodes (target={coarsen_to})')
    print()

    # ── Phase 2: Two initial partition strategies on THE SAME coarse result
    greedy_assignment = kahypar_solver.initial_partition_greedy(
        res['coarse_hyperedges'], res['coarse_node_weights'], q,
    )
    fem_assignment = fem_solver.initial_partition(
        res['coarse_hyperedges'], res['coarse_node_weights'], q,
    )

    # Coarse-level metrics
    print('── Coarse ──')
    gc, gi = _report('greedy', greedy_assignment, res['coarse_hyperedges'], coarse_nodes)
    fc, fi = _report('fem', fem_assignment, res['coarse_hyperedges'], coarse_nodes)

    # ── Phase 3: Project back to original hypergraph ─────────────────────
    greedy_projected = expand_coarse_labels(res['coarse_groups'], greedy_assignment, num_nodes)
    fem_projected = expand_coarse_labels(res['coarse_groups'], fem_assignment, num_nodes)

    print('── Projected ──')
    gpc, gpi = _report('greedy', greedy_projected, hyperedges, num_nodes)
    fpc, fpi = _report('fem', fem_projected, hyperedges, num_nodes)

    # ── Phase 4: FM refine via HyperRefineSolver ────────────────────────
    greedy_refined = refine_solver.refine(greedy_projected, hyperedges, q)
    fem_refined = refine_solver.refine(fem_projected, hyperedges, q)

    print('── Refined ──')
    grc, gri = _report('greedy', greedy_refined, hyperedges, num_nodes)
    frc, fri = _report('fem', fem_refined, hyperedges, num_nodes)

    # ── Assertions ───────────────────────────────────────────────────────
    for v in (gc, gi, fc, fi, gpc, gpi, fpc, fpi, grc, gri, frc, fri):
        assert isinstance(v, (int, float)) and v >= 0

    print()
    print(f'comparison (coarse): greedy={gc} vs fem={fc}')
    print(f'comparison (projected): greedy={gpc}/{gpi} vs fem={fpc}/{fpi}')
    print(f'comparison (refined):  greedy={grc}/{gri} vs fem={frc}/{fri}')


if __name__ == '__main__':
    test_compare_initial_partition_modes()
    print('smoke ok')