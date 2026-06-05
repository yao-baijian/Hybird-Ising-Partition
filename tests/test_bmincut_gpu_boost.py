from test_bmincut_base import *

num_trials = 10
num_steps = 1000
devices = ['cpu', 'gpu']
anneal = 'lin'
manual_grad = False
case_type = 'bmincut'

partition_methods = [
    'direct_fem',
    'coarse_fem_refine_metis',
    'coarse_fem_refine_kaffpa',
    'coarse_metis_refine_fem',
    'coarse_kaffpa_refine_fem',
]

instance_dir = '../partition/gset/'
instances = [f'G{i}' for i in range(1, 12)]
q_values = [2, 4, 6]  # Number of partitions
coarsen_list = [50, 100, 200]

detailed_fieldnames = [
    'instance', 'q', 'partition_method', 'device', 'coarsen_to',
    'cut_value', 'imbalance',
    'total_time_s', 'coarsen_time_s',
    'init_partition_time_s', 'refine_time_s',
]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
detailed_csv_path = os.path.join(build_dir, f'detailed_timing_{timestamp}.csv')

detailed_rows = []

print_header()

for instance in instances:
    n, m, J = read_graph(instance_dir + instance, index_start=1)
    for q in q_values:
        for partition_method in partition_methods:
            for dev in devices:
                for coarsen_to in coarsen_list:
                    no_coarsen = False
                    coarsen_time_s = 0.0
                    init_partition_time_s = 0.0
                    refine_time_s = 0.0
                    start_time = time.perf_counter()

                    if partition_method == 'direct_fem':
                        p, cut, partition_time_s = direct_fem(
                            case_type, instance_dir + instance, 1,
                            num_trials, num_steps, anneal, dev, q, manual_grad,
                        )
                        no_coarsen = True

                    elif partition_method == 'coarse_fem_refine_metis':
                        p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = coarse_fem_refine_metis(
                            J, q, coarsen_to,
                            num_trials, num_steps, anneal, dev, manual_grad,
                        )

                    elif partition_method == 'coarse_fem_refine_kaffpa':
                        p, cut, coarsen_time_s, init_partition_time_s, refine_time_s, coarsen_rounds = coarse_fem_refine_kaffpa(
                            J, q, coarsen_to,
                            num_trials, num_steps, anneal, dev, manual_grad,
                        )

                    elif partition_method == 'coarse_metis_refine_fem':
                        max_iterations = 50
                        num_steps_cyclic = 100
                        max_candidates = 60
                        num_trials_cyclic = 5
                        patience = 10
                        allow_nonadjacent = True
                        p, cut, coarsen_time_s, init_partition_time_s, refine_time_s = coarse_metis_refine_fem(
                            J, q, coarsen_to, anneal, dev, manual_grad,
                            max_iterations, num_steps_cyclic, max_candidates,
                            num_trials_cyclic, patience, allow_nonadjacent,
                        )

                    elif partition_method == 'coarse_kaffpa_refine_fem':
                        max_iterations = 50
                        num_steps_cyclic = 100
                        max_candidates = 60
                        num_trials_cyclic = 5
                        patience = 10
                        allow_nonadjacent = True
                        p, cut, coarsen_time_s, init_partition_time_s, refine_time_s = coarse_kaffpa_refine_fem(
                            J, q, coarsen_to, anneal, dev, manual_grad,
                            max_iterations, num_steps_cyclic, max_candidates,
                            num_trials_cyclic, patience, allow_nonadjacent,
                        )

                    else:
                        raise ValueError(f"Unknown partition method: {partition_method}")

                    n_nodes = J.shape[0]
                    final_assignment = p.argmax(dim=1).cpu().numpy()
                    counts = np.bincount(final_assignment, minlength=q)
                    ideal = n_nodes / q
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
                        'device': dev,
                        'coarsen_to': coarsen_to if not no_coarsen else 0,
                        'cut_value': cut_value,
                        'imbalance': imbalance,
                        'total_time_s': total_time_s,
                        'coarsen_time_s': coarsen_time_s,
                        'init_partition_time_s': init_partition_time_s,
                        'refine_time_s': refine_time_s,
                    }

                    detailed_rows.append(row)
                    print_row(row)

                    if no_coarsen:
                        break

# Save detailed timing CSV with device label
with open(detailed_csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
    writer.writeheader()
    for row in detailed_rows:
        writer.writerow(row)
print(f"Saved detailed timing to: {detailed_csv_path}")

