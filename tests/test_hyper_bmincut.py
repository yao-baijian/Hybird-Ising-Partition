import sys
sys.path.append('.')
import subprocess
from FEM import FEM
import torch
import pandas as pd

from utils import *

# num_trials = 500
# num_steps = 1000

num_trials = 10
num_steps = 500
dev = 'cpu' # if you do not have gpu in your computing devices, then choose 'cpu' here

case_type = 'hyperbmincut'
instance_root_dir = '../../partition/data/hypergraph_set/'

# instance_root_dir = './tests/test_instances/'

# map_types = ['normal', 'star', 'clique', 'weighted_clique', 'bisecgraph']
# epsilons = torch.linspace(0.01, 0.05, 5)
# epsilons = [0.03, 0.02, 0.01]
# grad_options = [True, False]
# q_values = [2, 4, 8, 16, 32, 64]
# instance_list = ['bibd_49_3.mtx.hgr',
#                 #  'Pd_rhs.mtx.hgr',
#                  'dac2012_superblue19.hgr',
#                  'ISPD98_ibm07.hgr',
#                  'G2_circuit.mtx.hgr']

map_types = ['normal']
epsilons = [0.02]
grad_options = [False]
q_values = [4]
instance_list = ['ISPD98_ibm01.hgr']
# instance_list = ['bibd_49_3.mtx.hgr']

results = []
total_experiments = len(instance_list) * len(map_types) * len(epsilons) * len(grad_options) * len(q_values)
current_experiment = 0

for instance in instance_list:

    hyperedges = parse_hypergraph_edges(instance_root_dir + instance)

    for map_type in map_types:
        for epsilon in epsilons:
            for grad_type in grad_options:
                for q in q_values:
                    current_experiment += 1
                    fpga_wrapper = None
                    print(f"Progress: {current_experiment}/{total_experiments} ({current_experiment/total_experiments*100:.1f}%)")
                    case_bmincut = FEM.from_file(case_type, instance_root_dir + instance, fpga_wrapper, index_start=1, epsilon=epsilon, q=q, hyperedges = hyperedges, map_type=map_type)
                    case_bmincut.set_up_solver(num_trials, num_steps, optimizer='adam', learning_rate=0.2, dev=dev, q=q, manual_grad= grad_type)
                    config, result = case_bmincut.solve()
                    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
                    best_config = config[optimal_inds[0]]
                    print(f'{instance}, optimal value {result.min()}')

                    group_assignment = best_config.argmax(dim=1).cpu().numpy()
                    fem_cut_value = evaluate_kahypar_cut_value(group_assignment, hyperedges)
                    group_counts = np.bincount(group_assignment, minlength=q)

                    result = {            
                        'instance': instance,
                        'map_type': map_type,
                        'epsilon': epsilon,
                        'grad_type': grad_type,
                        'q': q,
                        'fem_cut_value': fem_cut_value,
                        'group_counts': group_counts
                    }

                    results.append(result)


print("="*100)
print(f"{'Instance':<20} {'Map Type':<15} {'Epsilon':<8} {'Manual Grad':<12} {'Q':<8} {'FEM Cut':<10} {'Group Counts':<20}")
print("-"*100)

for result in results:
    group_counts_str = str(result['group_counts'].tolist())
    print(f"{result['instance']:<20} {result['map_type']:<15} {result['epsilon']:<8} "
            f"{str(result['grad_type']):<12} {result['q']:<8} {result['fem_cut_value']:<10.4f} {group_counts_str:<20}")

df = pd.DataFrame(results)
df.to_csv('fem_hypergraph_results.csv', index=False)
print(f"\nResults saved to fem_hypergraph_results.csv")