# qis3_mincut.py
import torch
from qis3.qis3 import QIS3   # 之前定义的 QIS3 类

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
    
    # 预先计算平衡惩罚需要的常数
    ones = torch.ones(n, device=device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)  # 用于内部能量计算，但最终 cut 仍用原 J
    
    best_cuts = torch.zeros(batch_size, device=device)
    best_imbalances = torch.zeros(batch_size, device=device)
    # 为了兼容原有返回格式，我们伪造 energies 和 solutions
    dummy_energies = torch.zeros(batch_size, num_iters, device=device)
    dummy_solutions = torch.zeros(batch_size, n, device=device)
    
    for b in range(batch_size):
        # 对每个 batch 独立运行 QIS3（因为分支定界等内部有随机性，可运行多次取最佳）
        # 这里为了快速演示，只用初始的 init_x[b] 作为种子
        init_spins = torch.sign(init_x[b])  # 从初始连续值得到初始自旋
        # QIS3 需要 Ising 耦合矩阵 J (original, not balanced)？我们直接用 J_balanced 作为能量矩阵
        solver = QIS3(J_balanced, branch_depth=branch_depth, popsize=popsize, num_iters=num_iters, device=device)
        # 但我们希望 QIS3 内部用 J_balanced 计算能量，最后输出 cut 和 imbalance。
        # 简便方法：修改 QIS3 的 _compute_energy 以返回 cut 和 imbalance，或者单独计算。
        # 这里简化为：运行 solver 得到最终自旋，然后计算 cut 和 imbalance。
        best_spin, _ = solver.solve()  # best_spin 是 numpy array
        best_spin_t = torch.tensor(best_spin, device=device)
        # 计算 cut_value: H = -0.5 * sum_{i<j} J_ij s_i s_j (original J, not balanced)
        # 注意：J 是原图邻接矩阵（正值），cut = 0.5 * sum_{i<j} w_ij (1 - s_i s_j)
        # 我们可以直接用 bsb_bmincut_batch 内部的 cut_value 计算公式：
        # 在 bsb_bmincut_batch 中，cut_value = 0.25 * (torch.sum(orig_J) - xJx)  where orig_J = -J, xJx = s^T (-J) s
        orig_J = -J   # 因为 bsb_bmincut_batch 中传入的 J 是原图邻接矩阵，然后内部取 orig_J = -J 来计算 cut
        # 但这里 J 就是原图邻接矩阵，所以 orig_J = -J
        xJx = torch.einsum('i,ij,j->', best_spin_t, -J, best_spin_t)
        cut_value = 0.25 * (torch.sum(-J) - xJx)   # 公式来源于 bsb_bmincut_batch
        # 或者更直观：
        # cut = 0.5 * sum_{i<j} J_ij (1 - s_i s_j)
        # n = len(best_spin_t)
        # cut = 0.5 * sum_{i<j} J[i,j] * (1 - best_spin_t[i]*best_spin_t[j])
        imbalance = (torch.sum(best_spin_t).float() / n).item()  # 简单 imbalance = |sum(s)|/n，越小越平衡
        # 注意：bsb_bmincut_batch 中 imbalance 返回的是 sum_x（即 sum(s)），而不是绝对值，用户后续可能自己处理。
        # 为了和用户代码一致，我们也返回 sum_x
        imbalance = torch.sum(best_spin_t)
        
        best_cuts[b] = cut_value
        best_imbalances[b] = imbalance
        dummy_solutions[b] = best_spin_t
    
    # 伪造的 energies 最后一行设置为 cut_value（或能量，用户不一定用）
    dummy_energies[:, -1] = best_cuts
    return dummy_energies, dummy_solutions, best_cuts, best_imbalances