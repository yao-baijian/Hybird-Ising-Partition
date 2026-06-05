import sys
sys.path.append('.')
from src.fem import FEM
import torch

num_trials = 1000
num_steps = 1000
dev = 'cuda'

case_maxcut = FEM.from_file(
    'maxcut', 'tests/test_instances/G1.txt', index_start=1, discretization=True
)
case_maxcut.set_up_solver(
    num_trials, num_steps, manual_grad=True, betamin=0.001, betamax=0.5, 
    learning_rate=0.1, optimizer='rmsprop', dev=dev
)
config, result = case_maxcut.solve()
optimal_inds = torch.argwhere(result==result.max()).reshape(-1)
print(f'maxcut test instance, optimal value {result.max()}')

case_bmincut = FEM.from_file(
    'bmincut', 'tests/test_instances/karate.txt', index_start=1
)
case_bmincut.set_up_solver(num_trials, num_steps, dev=dev)
config, result = case_bmincut.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
print(f'bmincut test instance, optimal value {result.min()}')

case_vertexcover = FEM.from_file(
    'vertexcover', 'tests/test_instances/vertexcover.txt', index_start=0
)
case_vertexcover.set_up_solver(
    num_trials, num_steps, betamin=10, betamax=30, h_factor=1, dev=dev,
    learning_rate=0.01, anneal='exp'
)
config, result =case_vertexcover.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
print(f'vertexcover test instance, optimal value {result.min()}')

case_maxksat = FEM.from_file(
    'maxksat', 'tests/test_instances/s3v70c1000-1.cnf'
)
case_maxksat.set_up_solver(
    num_trials, num_steps, manual_grad=True, h_factor=0.3, anneal='lin',
    betamin=0.01, betamax=30, learning_rate=1.1, sparse=True, dev=dev,
)
config, result = case_maxksat.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
print(f'maxksat test instance, optimal value {result.min()}')