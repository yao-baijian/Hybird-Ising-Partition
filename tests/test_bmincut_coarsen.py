from test_bmincut_base import *
from src.method_registry import registry, ensure_configs, merge_config
from src.solver_base import (
    get_solver, run_composite_method,
    FemSolver, SbmSolver, KaffpaSolver, MetisSolver, CyclicSolver,
)
from pathlib import Path

# ── Ensure default JSON configs exist in config/ ──────────────────────────
ensure_configs()
config_dir = Path.cwd() / "config"

# ── Initialise solvers once — each loads its own config ───────────────────
fem_solver = FemSolver(config_dir)
sbm_solver = SbmSolver(config_dir)
kaffpa_solver = KaffpaSolver(config_dir)
metis_solver = MetisSolver(config_dir)
cyclic_solver = CyclicSolver(config_dir)

num_steps = 1000
dev = 'cpu'
anneal = 'lin'          # default schedule (can be overridden per-method in JSON)
case_type = 'bmincut'

partition_methods = [
    'direct_fem',
    'direct_sbm',
    'kaffpa',
    'init_fem_refine_kaffpa',
    'init_sbm_refine_kaffpa',
    'init_kaffpa_refine_fem',
]

instance_dir = '../partition/gset/'
instances = [f'G{i}' for i in range(1, 5)]
q_values = [2, 4]  # Number of partitions

best_rows = []

print_header()

for instance in instances:    
    n, m, J = read_graph(instance_dir + instance, index_start = 1)
    for q in q_values:
        for partition_method in partition_methods:
            p = None
            best_config = None
            best_row = None
            coarsen_to = merge_config(partition_method, {}, config_dir).get('coarsen_to', 50) \
                if partition_method not in ('direct_fem', 'direct_sbm', 'metis', 'kahip') else 0
            no_coarsen = False
            coarsen_time_s = 0.0
            init_partition_time_s = 0.0
            refine_time_s = 0.0
            start_time = time.perf_counter()

            if partition_method == 'direct_fem':

                p, cut, partition_time_s = fem_solver.solve_direct(
                    case_type, instance_dir + instance, 1, q,
                )
                no_coarsen = True

            elif partition_method == 'direct_sbm':

                p, cut, partition_time_s = sbm_solver.solve_direct(
                    case_type, instance_dir + instance, 1, q,
                )
                no_coarsen = True

            elif partition_method == 'metis':

                p, cut, partition_time_s = metis_kway(J, q)
                no_coarsen = True

            elif partition_method == 'init_fem_refine_metis':

                p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = run_composite_method(J, q, fem_solver, metis_solver)

            elif partition_method == 'init_fem_refine_kaffpa':

                p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = run_composite_method(J, q, fem_solver, kaffpa_solver)

            elif partition_method == 'init_sbm_refine_kaffpa':

                p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = run_composite_method(J, q, sbm_solver, kaffpa_solver)

            elif partition_method == 'kaffpa':

                cfg = merge_config(partition_method, {}, config_dir)
                p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = kaffpa_kway(
                    J, q, cfg.get('coarsen_to', 50),
                    epsilon=cfg.get('epsilon', 0.05),
                    refine_passes=cfg.get('refine_passes', 10),
                )

            elif partition_method == 'kahip':

                p, cut, coarsen_time_s, init_partition_time_s,refine_time_s = kahip_kway(J, q, coarsen_to)
                no_coarsen = True

            elif partition_method == 'init_metis_refine_fem':

                p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = run_composite_method(J, q, metis_solver, cyclic_solver)

            elif partition_method == 'init_kaffpa_refine_fem':

                p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = run_composite_method(J, q, kaffpa_solver, cyclic_solver)

            else:
                raise ValueError(f"Unknown partition method: {partition_method}")

            n = J.shape[0]
            final_assignment = p.argmax(dim=1).cpu().numpy()
            counts = np.bincount(final_assignment, minlength=q)
            ideal = n / q
            imbalance = float(np.max(np.abs(counts - ideal) / ideal))

            try:
                cut_value = float(cut.item())
            except Exception:
                cut_value = float(cut)

            total_time_s = time.perf_counter() - start_time

            row = {
                'instance': instance,
                'q': q,
                'partition_method': partition_method,
                'coarsen_to': coarsen_to if not no_coarsen else 0,
                'cut_value': cut_value,
                'imbalance': imbalance,
                'total_time_s': total_time_s,
                'coarsen_time_s': coarsen_time_s,
                'init_partition_time_s': init_partition_time_s,
                'refine_time_s': refine_time_s,
            }

            best_rows.append(row)
            print_row(row)

save_to_csv(best_rows)

