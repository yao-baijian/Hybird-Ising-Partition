# import sys
# import os

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# import numpy as np
# import torch
# from src.sbm.sbm import bsb_bmincut_batch
# from src.sbm.utils import load_data
# from src.qis3.qis3 import QIS3                # 假设 qis3.py 中有 QIS3 类

def qis3_bmincut_batch(J, init_x, init_y, num_iters, branch_depth, popsize, lambda_balance, device='cpu', qis3_batch_size=1):
    """
    QIS3 wrapper for balanced mincut, mimicking the interface of bsb_bmincut_batch.
    
    Args:
        qis3_batch_size: independent batch size for QIS3 (default 1, since QIS3 internal
                         already uses popsize for diversity, no need for large batch).
    
    Returns:
        energies: torch.Tensor shape (batch_size, num_iters)   (dummy, last column is total energy)
        solutions: torch.Tensor shape (batch_size, n)          (dummy, best solution per batch)
        cut_values: torch.Tensor shape (batch_size,)           (actual cut weight)
        imbalances: torch.Tensor shape (batch_size,)           (sum of spins)
    """
    batch_size = qis3_batch_size
    n = J.shape[0]
    device = J.device
    
    # Balanced Ising matrix used in BSB (same as bmincut_to_bsb)
    ones = torch.ones(n, device=device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)
    
    best_cuts = torch.zeros(batch_size, device=device)
    best_imbalances = torch.zeros(batch_size, device=device)
    dummy_energies = torch.zeros(batch_size, num_iters, device=device)
    dummy_solutions = torch.zeros(batch_size, n, device=device)
    
    for b in range(batch_size):
        # 为每个 batch 独立运行 QIS3
        solver = QIS3(
            J_balanced,
            branch_depth=branch_depth,
            popsize=popsize,
            num_iters=num_iters,
            device=device
        )
        best_spin, _ = solver.solve()   # best_spin is numpy array of +/-1
        best_spin_t = torch.tensor(best_spin, device=device, dtype=torch.float32)
        
        # 计算 cut 和 imbalance（使用原始邻接矩阵 J）
        orig_J = -J
        xJx = torch.einsum('i,ij,j->', best_spin_t, orig_J, best_spin_t)
        cut_value = 0.25 * (torch.sum(orig_J) - xJx)
        imbalance = torch.sum(best_spin_t)   # sum of spins (used in penalty)
        
        best_cuts[b] = cut_value
        best_imbalances[b] = imbalance
        total_energy = cut_value + lambda_balance * (imbalance ** 2)
        dummy_energies[b, -1] = total_energy
        dummy_solutions[b] = best_spin_t
    
    return dummy_energies, dummy_solutions, best_cuts, best_imbalances

def compare_qis3_vs_bsb(device='cpu'):
    # BSB 扫描 dt
    dt_values = np.arange(0.3, 1 + 0.1/2, 0.5).tolist()
    # QIS3 扫描超参数
    branch_depths = [0, 1, 2]
    popsizes = [8, 16, 32]
    num_iter = 1000
    batch_size = 5
    qis3_batch_size = 1
    
    graph_indices = [5, 10, 34]   # G6, G11, G35
    lambda_values = [0.05, 0.1, 0.2]
    
    instance_dir = "../partition/gset/"
    
    total_lambda = len(lambda_values)
    total_graphs = len(graph_indices)
    total_dt = len(dt_values)
    total_qis3_configs = len(branch_depths) * len(popsizes)
    
    for li, lambda_balance in enumerate(lambda_values):
        print(f"\n{'='*100}")
        print(f"[Progress] lambda_balance={lambda_balance}  ({li+1}/{total_lambda})")
        print(f"{'='*100}")
        print(f"{'graph':<6} | {'bsb dt':<6} | {'bsb cut':<9} | {'bsb imb':<9} | "
              f"{'qis3 config':<12} | {'qis3 cut':<9} | {'qis3 imb':<9}")
        print("-" * 90)
        
        for gi, graph_idx in enumerate(graph_indices):
            graph_name = f"G{graph_idx+1}"
            print(f"\n  >>> [{li+1}/{total_lambda}, {gi+1}/{total_graphs}] Loading graph {graph_name}...")
            sys.stdout.flush()
            
            set_name = instance_dir + f'G{graph_idx + 1}'        
            J = load_data(set_name)
            J = (J.T + J)   # 对称化
            
            # 固定随机种子以保证可比性
            np.random.seed(42)
            init_x_batch = np.random.uniform(-0.1, 0.1, (batch_size, J.shape[0]))
            init_y_batch = np.random.uniform(-0.1, 0.1, (batch_size, J.shape[0]))
            init_x_tensor = torch.tensor(init_x_batch, dtype=torch.float32, device=device)
            init_y_tensor = torch.tensor(init_y_batch, dtype=torch.float32, device=device)
            J_tensor = torch.tensor(J, dtype=torch.float32, device=device)
            
            # ---------- BSB: 扫描 dt ----------
            print(f"    BSB: scanning {total_dt} dt values...")
            sys.stdout.flush()
            best_bsb_energy = float('inf')
            best_bsb_dt = None
            best_bsb_cut = None
            best_bsb_imb = None
            for di, dt in enumerate(dt_values):
                bsb_energy, _, bsb_cut, bsb_imb = bsb_bmincut_batch(
                    J_tensor, init_x_tensor, init_y_tensor,
                    num_iter, dt, lambda_balance=lambda_balance
                )
                final_energies = bsb_energy[:, -1]
                min_idx = torch.argmin(final_energies)
                cur_energy = final_energies[min_idx].item()
                if cur_energy < best_bsb_energy:
                    best_bsb_energy = cur_energy
                    best_bsb_dt = dt
                    best_bsb_cut = bsb_cut[min_idx].item()
                    best_bsb_imb = bsb_imb[min_idx].item()
            print(f"    BSB done. Best dt={best_bsb_dt:.2f}, cut={best_bsb_cut:.0f}")
            sys.stdout.flush()
            
            # ---------- QIS3: 扫描 branch_depth & popsize ----------
            print(f"    QIS3: scanning {total_qis3_configs} configs (branch_depth x popsize)...")
            sys.stdout.flush()
            best_qis3_energy = float('inf')
            best_qis3_config = None
            best_qis3_cut = None
            best_qis3_imb = None
            for bd in branch_depths:
                for ps in popsizes:
                    q3_energy, _, q3_cut, q3_imb = qis3_bmincut_batch(
                        J_tensor, init_x_tensor, init_y_tensor,
                        num_iter, branch_depth=bd, popsize=ps,
                        lambda_balance=lambda_balance, device=device,
                        qis3_batch_size=qis3_batch_size
                    )
                    final_energies = q3_energy[:, -1]
                    min_idx = torch.argmin(final_energies)
                    cur_energy = final_energies[min_idx].item()
                    if cur_energy < best_qis3_energy:
                        best_qis3_energy = cur_energy
                        best_qis3_config = f"d={bd},p={ps}"
                        best_qis3_cut = q3_cut[min_idx].item()
                        best_qis3_imb = q3_imb[min_idx].item()
            print(f"    QIS3 done. Best config={best_qis3_config}, cut={best_qis3_cut:.0f}")
            sys.stdout.flush()
            
            # 输出结果
            print(f"  Result: G{graph_idx+1:<5} | {best_bsb_dt:<6.2f} | {best_bsb_cut:<9.0f} | "
                  f"{best_bsb_imb:<8.4f} | {best_qis3_config:<12} | {best_qis3_cut:<9.0f} | {best_qis3_imb:<8.4f}")
            sys.stdout.flush()
    
    print(f"\n{'='*100}")
    print(f"Experiment complete!")
    print(f"{'='*100}")

def direct_qis3():
    # 直接运行 QIS3，使用默认超参数，比较与 BSB 的结果
    import numpy as np
    import TuringQHeuristic as TQ
    J  = np.random.randn(70,70) # 待解问题的矩阵，以一个随机矩阵为例
    J = J.T + J ; J*=0.5 # 对称化矩阵。
    J -= np.diag(np.diag(J)) # 去对角元矩阵。
    h  = np.random.randn(70)
    
    Q  = np.random.randn(70,70) # 待解问题的矩阵，以一个随机矩阵为例
    Q = Q.T + Q ; Q*=0.5

    model1= TQ.Model()
    model1.set_J_h(J=J,h=h)
    model1.timeout=10
    model1.batchsize=16
    model1.n_iter = 1000
    model1.fileout=False
    model1.monitor=False
    model1.auto_complete_init()

    print("best mode:",model1.mode)
    model1.mode=model1.mode  # 指定模式
    model1.timeout=10 # 指定时间
    result1 = model1.optimize() 

    model2= TQ.Model()
    model2.set_Q(Q=Q)
    model2.timeout=10
    model2.batchsize=16
    model2.n_iter = 1000
    model2.fileout=False
    model2.monitor=False
    model2.auto_complete_init()

    print("best mode:",model2.mode)
    model2.mode=model2.mode  # 指定模式
    model2.timeout=10 # 指定时间
    result2 = model2.optimize() 

    print(model1.best_energy)
    print(model1.best_state)
    print(model1.best_state@J@model1.best_state+h@model1.best_state)

    print(model2.best_energy)
    print(model2.best_state)
    print((model2.best_state+1)/2@Q@(model2.best_state+1)/2)

if __name__ == "__main__":
    # compare_qis3_vs_bsb(device='cpu')
    direct_qis3()
    pass