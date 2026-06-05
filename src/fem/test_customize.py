import sys
sys.path.append('.')
from src.fem import FEM, read_graph
import torch

num_trials = 100
num_steps = 1000
dev = 'cpu'

def customize_expected_func(J, p):
    couplings = -torch.eye(J.shape[0], dtype=J.dtype, device=J.device) + 2 * J
    return torch.bmm(
        (p @ couplings).reshape(-1, 1, J.shape[1]),
        p.reshape(-1, p.shape[1], 1)
    ).reshape(-1)

def customize_infer_func(J, p):
    config = p.round()
    return config, customize_expected_func(J, config)


num_nodes, num_interactions, couplings = read_graph(
    'tests/test_instances/mis.txt'
)
case_customize = FEM.from_couplings(
    'customize', num_nodes, num_interactions, couplings,
    customize_expected_func=customize_expected_func,
    customize_infer_func=customize_infer_func
)
case_customize.set_up_solver(num_trials, num_steps, dev=dev)
case_customize.solver.binary = True
config, result = case_customize.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
print(f'customize (maximum independent set) test instance, optimal value {result.min()}')
optimal_configs = torch.unique(config[optimal_inds], dim=0)
print('optimal configs are')
for conf in optimal_configs:
    print(conf)
