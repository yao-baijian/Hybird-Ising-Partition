import sys
sys.path.append('.')
from FEM import FEM
import torch
import time
import numpy as np

from utils import *

# num_trials = 500
# num_steps = 1000

num_trials = 1
num_steps = 100
dev = 'cpu' # if you do not have gpu in your computing devices, then choose 'cpu' here

# normal graph
# case_type = 'bmincut'
# instance = 'tests/test_instances/karate.txt'
# case_bmincut = FEM.from_file(case_type, instance, index_start=1)
# case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=3)
# config, result = case_bmincut.solve()
# optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
# print(f'{instance}, optimal value {result.min()}')

# instance = '../partition/data/ash219/ash219.mtx'
# case_bmincut = FEM.from_file(case_type, instance, index_start=1)
# case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=3)
# config, result = case_bmincut.solve()
# optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
# print(f'{instance}, optimal value {result.min()}')

instance = '../partition/full_benchmark_set/as-caida.mtx.hgr'

hyperedges = parse_hypergraph_edges(instance)
num_nodes = max((max(hyperedge) for hyperedge in hyperedges if hyperedge), default=-1) + 1
clique_graph = build_clique_expanded_graph(hyperedges, num_nodes=num_nodes, normalize_weight=True)
coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_leaf_folding(
	clique_graph,
	node_weights=torch.ones(num_nodes, dtype=torch.float32),
	max_degree=500,
	min_nodes=5000,
)

fpga_wrapper = None
start_time = time.time()
case_bmincut = FEM.from_couplings(
	'bmincut',
	coarse_graph.shape[0],
	int(coarse_graph._nnz() // 2),
	coarse_graph,
	node_weights=coarse_node_weights,
)

case_bmincut.set_up_solver(num_trials, num_steps, anneal='lin',dev=dev, q=2, manual_grad= False)

config, result = case_bmincut.solve()
print(f"solve took: {time.time() - start_time:.4f} seconds")

optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
best_config = config[optimal_inds[0]]
coarse_assignment = best_config.argmax(dim=1).cpu().numpy()
group_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, num_nodes)
fem_cut_value, avg_imbalance = evaluate_kahypar_cut_value(group_assignment, hyperedges, [1.0] * len(hyperedges))

print(f'{instance}, fem result {fem_cut_value}, avg imbalance {avg_imbalance}')