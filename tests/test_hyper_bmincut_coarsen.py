"""
Hypergraph initial-partition comparison test using solver mode.

Pipeline:
  1. KahyparLikeSolver coarsens the hypergraph once (HEM matching).
  2. The SAME coarse result is fed to two different initial partition
     strategies: greedy (built into KahyparLikeSolver) and FEM-based
     (FemCoarsenSolver).
  3. Both assignments go through a V-Cycle (hierarchical uncoarsening
     with refinement at each level) via ``vcycle_uncoarsen``.
  4. The best result across ``num_runs`` outer trials is reported.

When ``verbose=False``, only a single pipe-delimited table line is printed:
  |name|q|greedy_cut|greedy_imb|fem_cut|fem_imb|n_levels|

Configurable parameters at the top of this file:
  coarsen_to      — target coarse nodes (controls KahyparLikeSolver)
  num_runs        — outer-loop trials (best result reported)
  fem_method      — ``'fem'`` (default) or ``'pubo'``
  fem_map_type    — ``'clique'`` (default) or ``'star'`` expansion
  fem_anneal      — anneal schedule for FEM solver (default ``'lin'``)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyper_solver import (
    KahyparLikeSolver, FemCoarsenSolver, HyperRefineSolver,
    vcycle_uncoarsen,
)
from src.partition.hyper_utils import evaluate_kahypar_cut_value
from utils import parse_hypergraph_edges

import numpy as np

# ── Configurable parameters ──────────────────────────────────────────────

coarsen_to = 15          # target coarse nodes (all methods)
q = 4                    # number of partitions
refine_passes = 6        # FM refinement passes
verbose = False
num_runs = 15            # outer trials (best result across all runs)

# FemCoarsenSolver options (see class docstring for details)
fem_method = 'fem'       # 'fem' or 'pubo'
fem_map_type = 'star'  # 'clique' or 'star'
fem_anneal = 'inverse'       # anneal schedule for FEM solver
num_steps = 2000
num_trials = 10
dev = 'cpu'

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
    num_trials=num_trials,
    num_steps=num_steps,
    dev=dev,
    method=fem_method,
    map_type=fem_map_type,
    anneal=fem_anneal,
)

refine_solver = HyperRefineSolver()
refine_solver.update_params(
    mode_cycle=('mcts', 'evolution', 'flow'),
    repair_balance=True,
    flow_passes=refine_passes,
    max_imbalance=0.05,
)


def _load_hypergraph():
    instance = '../partition/full_benchmark_set/powersim.mtx.hgr'
    hyperedges = parse_hypergraph_edges(str(instance))
    num_nodes = max((max(h) for h in hyperedges if h), default=-1) + 1
    return hyperedges, num_nodes, Path(instance).stem


def _report(name, assignment, hyperedges, num_nodes):
    """Evaluate and print cut / imbalance for an assignment."""
    cut, imb = evaluate_kahypar_cut_value(
        assignment, hyperedges, hyperedge_weights=[1.0] * len(hyperedges),
    )
    if verbose:
        print(
            f'{name}  cut = {cut}, imbalance = {imb:.15f}'
        )
    return cut, imb


def test_compare_initial_partition_modes():
    hyperedges, num_nodes, instance_name = _load_hypergraph()

    # ── Phase 1: Coarsen ONCE ────────────────────────────────────────────
    res = kahypar_solver.coarsen(hyperedges, num_nodes, q)
    coarse_nodes = len(res['coarse_groups'])
    hierarchy_stack = res.get('hierarchy_stack', [])
    n_levels = len(hierarchy_stack)

    if verbose:
        print(f'Coarsened: {num_nodes} -> {coarse_nodes} nodes (target={coarsen_to})')
        print(f'Hierarchy levels: {n_levels}')
        print()

    # ── Phase 2+3: Best-of-N runs on the SAME coarse result ──────────────
    best_g = {'cut': float('inf'), 'imb': 0.0}
    best_f = {'cut': float('inf'), 'imb': 0.0}

    for run in range(num_runs):
        if verbose:
            print(f'── Run {run+1}/{num_runs} ──')

        greedy_assignment = kahypar_solver.initial_partition_greedy(
            res['coarse_hyperedges'], res['coarse_node_weights'], q,
            seed=run,
        )
        fem_assignment = fem_solver.initial_partition(
            res['coarse_hyperedges'], res['coarse_node_weights'], q,
        )

        if verbose:
            _report(f'greedy (coarse)', greedy_assignment, res['coarse_hyperedges'], coarse_nodes)
            _report(f'fem (coarse)', fem_assignment, res['coarse_hyperedges'], coarse_nodes)

        # ── V-Cycle uncoarsening ──
        if verbose and hierarchy_stack:
            print(f'── V-Cycle ({n_levels} levels) ──')

        greedy_vcycle = vcycle_uncoarsen(
            greedy_assignment, hierarchy_stack, hyperedges, q,
            refine_solver, verbose=verbose,
        )
        fem_vcycle = vcycle_uncoarsen(
            fem_assignment, hierarchy_stack, hyperedges, q,
            refine_solver, verbose=verbose,
        )

        grc, gri = evaluate_kahypar_cut_value(
            greedy_vcycle, hyperedges, hyperedge_weights=[1.0] * len(hyperedges),
        )
        frc, fri = evaluate_kahypar_cut_value(
            fem_vcycle, hyperedges, hyperedge_weights=[1.0] * len(hyperedges),
        )

        if verbose:
            print(f'  greedy cut={grc}, imb={gri:.4f}  |  fem cut={frc}, imb={fri:.4f}')
            print()

        if grc < best_g['cut']:
            best_g = {'cut': grc, 'imb': gri}
        if frc < best_f['cut']:
            best_f = {'cut': frc, 'imb': fri}

    # ── Single-line table output (always printed) ────────────────────────
    print(
        f"|{instance_name}|{q}|"
        f"{best_g['cut']}|{best_g['imb']:.15f}|"
        f"{best_f['cut']}|{best_f['imb']:.15f}|"
        f"{n_levels}|"
    )

    # ── Assertions ───────────────────────────────────────────────────────
    for v in (best_g['cut'], best_g['imb'], best_f['cut'], best_f['imb']):
        assert isinstance(v, (int, float)) and v >= 0


if __name__ == '__main__':
    test_compare_initial_partition_modes()
    if verbose:
        print('smoke ok')