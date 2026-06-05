# qis3_mincut.py
import torch
from src.qis3.qis3 import QIS3   # 之前定义的 QIS3 类

def qis3_bmincut_batch(J, init_x, init_y, num_iters, branch_depth=1, popsize=10, lambda_balance=1.0, device='cpu'):
    """
    QIS3 for balanced mincut.
    Returns:
        energies: torch.Tensor shape (batch_size, num_iters) - not used here, kept for compatibility
        solutions: torch.Tensor shape (batch_size, n) - best solutions per batch? Actually QIS3 returns single best.
        cut_value: torch.Tensor shape (batch_size,) - best cut value for each batch (if batch_size>1, we run QIS3 multiple times)
        imbalance: torch.Tensor shape (batch_size,) - corresponding imbalance
    """
    batch_size = init_x.shape[0]
    n = J.shape[0]
    device = J.device

    ones = torch.ones(n, device=device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)
    best_cuts = torch.zeros(batch_size, device=device)
    best_imbalances = torch.zeros(batch_size, device=device)
    dummy_energies = torch.zeros(batch_size, num_iters, device=device)
    dummy_solutions = torch.zeros(batch_size, n, device=device)
    
    for b in range(batch_size):
        init_spins = torch.sign(init_x[b])  
        solver = QIS3(J_balanced, branch_depth=branch_depth, popsize=popsize, num_iters=num_iters, device=device)
        best_spin, _ = solver.solve() 
        best_spin_t = torch.tensor(best_spin, device=device)
        orig_J = -J   
        xJx = torch.einsum('i,ij,j->', best_spin_t, -J, best_spin_t)
        cut_value = 0.25 * (torch.sum(-J) - xJx) 
        imbalance = (torch.sum(best_spin_t).float() / n).item()  
        imbalance = torch.sum(best_spin_t)
        
        best_cuts[b] = cut_value
        best_imbalances[b] = imbalance
        dummy_solutions[b] = best_spin_t
    
    dummy_energies[:, -1] = best_cuts
    return dummy_energies, dummy_solutions, best_cuts, best_imbalances