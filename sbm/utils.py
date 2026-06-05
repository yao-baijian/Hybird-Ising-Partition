import numpy as np
from typing import Tuple
import torch
from scipy.sparse import lil_matrix, csr_matrix

def load_data(name='data/Gset/G30'):
    file = open(name, 'r')
    for (idx, line) in enumerate(file):
        if idx == 0:
            N = int(line.split(' ')[0])
            J = np.zeros([N,N])
        else:
            J[int(line.split(' ')[0])-1][int(line.split(' ')[1])-1] = (line.split(' ')[2])
    file.close()
    tor_arr = -J
    return tor_arr

def load_dimacs10_data(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        while first_line == '':
            first_line = file.readline().strip()
        N = int(first_line.split()[0])
        J = np.zeros((N, N), dtype=float)   # 默认权重为1，也可以改为int

        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0]) - 1          # 转为0索引
            for v_str in parts[1:]:
                v = int(v_str) - 1
                J[u][v] = 1.0
                J[v][u] = 1.0             # 无向图对称

    tor_arr = -J
    return tor_arr

def load_qplib_data(file_path):
    """
    Revised QPLIB loader based on official documentation:
    Objective: 1/2 x^T Q^0 x + b^0 x + q^0
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    num_vars = int(lines[3].split('#')[0].strip())
    objective_sense = lines[2].lower() # minimize or maximize
    
    Q = np.zeros((num_vars, num_vars))
    b = np.zeros(num_vars)
    
    for idx, line in enumerate(lines):
        if 'number of quadratic terms in objective' in line:
            num_terms = int(line.split('#')[0][0:idx].strip()) if '#' in line else int(line.split()[0])
            # The line itself may contain the count, but we've already split it. 
            # Re-parsing with split logic for robustness
            num_terms = int(line.split()[0])

            for i in range(1, num_terms + 1):
                parts = lines[idx + i].split()
                v1, v2, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                Q[v1, v2] = val
                # print(f"Parsed quadratic term: Q[{v1}, {v2}] = {val}")
        elif 'default value for linear coefficients in objective' in line:
            default_b = float(line.split()[0])
            b.fill(default_b)
        elif 'number of non-default linear coefficients in objective' in line:
            num_terms = int(line.split()[0])
            for i in range(1, num_terms + 1):
                parts = lines[idx + i].split()
                v1, val = int(parts[0]) - 1, float(parts[1])
                b[v1] = val
                # print(f"Parsed linear term: b[{v1}] = {val}")
        
    return Q, b, num_vars, objective_sense

def read_tsplib_data(filename: str) -> Tuple[int, np.ndarray, str]:
    coordinates = []
    dimension = 0
    name = ""
    reading_coords = False
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.startswith("EOF"):
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    # 忽略第一列（节点编号），读取坐标
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append([x, y])
    
    return dimension, np.array(coordinates), name

def maxcut_to_bsb(J):
    J_balanced = -0.5 * J
    return J_balanced

def bmincut_to_bsb(J, lambda_balance=1.0):
    N = J.shape[0]
    ones = torch.ones(N, device=J.device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)
    return J_balanced

def bmincut_to_kway_qubo(J, k, node_weights=None, lambda_balance=1.0,sparse_threshold=5000):
    """
    将 k-way 平衡图割转化为 QUBO 矩阵 (或 Ising 耦合矩阵)，适用于 SBM 等求解器。

    参数:
        J (torch.Tensor): 邻接矩阵，形状 (N, N)，可稀疏或稠密。
        k (int): 分区数 (≥2)。
        node_weights (torch.Tensor, optional): 节点权重，形状 (N,)。默认全为1。
        lambda_balance (float): 平衡惩罚系数。
        sparse_threshold (int): 当变量数 (N*k) 超过此值时使用稀疏存储。

    返回:
        Q (torch.Tensor or scipy.sparse.csr_matrix): QUBO 矩阵，形状 (N*k, N*k)。
        offset (float): 常数项，可忽略。
    """
    if J.is_sparse:
        J = J.coalesce()
        edges = J.indices().t().cpu().numpy()
        weights = J.values().cpu().numpy()
    else:
        N = J.shape[0]
        # 转为稀疏格式避免稠密遍历
        J_sp = J.to_sparse().coalesce()
        edges = J_sp.indices().t().cpu().numpy()
        weights = J_sp.values().cpu().numpy()

    N = J.shape[0]
    if node_weights is None:
        node_weights = torch.ones(N, dtype=torch.float32)
    else:
        node_weights = node_weights.float().flatten()

    total_weight = node_weights.sum().item()
    target = total_weight / k                # 每个分区的理想总权重

    n_vars = N * k
    # 选择存储格式
    use_sparse = n_vars > sparse_threshold
    if use_sparse:
        Q = lil_matrix((n_vars, n_vars), dtype=np.float64)
    else:
        Q = torch.zeros((n_vars, n_vars), dtype=torch.float64)

    def idx(i, p):
        return i * k + p

    # ========== 1. 切割目标 ==========
    # cut = Σ_{(i,j)} w_ij * (1 - Σ_p x_{i,p} x_{j,p})
    # 展开后得到二次项: -w_ij * x_{i,p} x_{j,p} 以及一次项: +w_ij * 1 (常数)
    for (i, j), w in zip(edges, weights):
        if i == j:
            continue
        # 对每个分区 p
        for p in range(k):
            vi = idx(i, p)
            vj = idx(j, p)
            if use_sparse:
                Q[vi, vj] += -w
                Q[vj, vi] += -w
            else:
                Q[vi, vj] += -w
                Q[vj, vi] += -w
        # 常数项将在最后汇总，此处跳过

    # ========== 2. 平衡惩罚项 ==========
    # λ * Σ_p ( Σ_i c_i x_{i,p} - target )²
    # 展开: λ Σ_p ( (Σ_i c_i x_{i,p})² - 2 target Σ_i c_i x_{i,p} + target² )
    # 忽略常数 target²
    for p in range(k):
        # 线性项: -2 λ target c_i
        for i in range(N):
            vi = idx(i, p)
            coeff = -2.0 * lambda_balance * target * node_weights[i]
            if use_sparse:
                Q[vi, vi] += coeff
            else:
                Q[vi, vi] += coeff
        # 二次项: 2 λ c_i c_j   for i<j 同分区
        for i in range(N):
            vi = idx(i, p)
            for j in range(i+1, N):
                vj = idx(j, p)
                coeff = 2.0 * lambda_balance * node_weights[i] * node_weights[j]
                if use_sparse:
                    Q[vi, vj] += coeff
                    Q[vj, vi] += coeff
                else:
                    Q[vi, vj] += coeff
                    Q[vj, vi] += coeff

    # ========== 3. 独热约束 (每个节点恰好选一个分区) ==========
    # 使用强惩罚 M 乘以 ( Σ_p x_{i,p} - 1 )²
    M = 100.0 * max(weights.max(), lambda_balance * total_weight**2) if len(weights) > 0 else 100.0
    for i in range(N):
        # 线性项: -2M
        for p in range(k):
            vi = idx(i, p)
            if use_sparse:
                Q[vi, vi] += -2.0 * M
            else:
                Q[vi, vi] += -2.0 * M
        # 二次项: 2M for p != q
        for p in range(k):
            vi = idx(i, p)
            for q in range(p+1, k):
                vj = idx(i, q)
                if use_sparse:
                    Q[vi, vj] += 2.0 * M
                    Q[vj, vi] += 2.0 * M
                else:
                    Q[vi, vj] += 2.0 * M
                    Q[vj, vi] += 2.0 * M

    # 收集常数项 (来自切割目标的 w_ij 和平衡惩罚的 +λ target²)
    constant = 0.0
    for (i, j), w in zip(edges, weights):
        if i != j:
            constant += w
    constant += lambda_balance * k * (target ** 2)
    # 独热约束的常数项 +M (每个节点贡献 M, 因为 Σ_p x_{i,p}=1 时 (1-1)²=0, 展开得到 +M)
    constant += N * M

    if use_sparse:
        Q = Q.tocsr()
    else:
        # 确保对称 (通常已是)
        pass

    return Q, constant

def kway_qubo_to_bsb(J, k, node_weights=None, lambda_balance=1.0):
    Q, const = bmincut_to_kway_qubo(J, k, node_weights, lambda_balance)
    from scipy.sparse import csr_matrix
    if isinstance(Q, csr_matrix):
        Q = Q.toarray()
    else:
        Q = Q.numpy()
    J_ising = -0.5 * Q
    h = -0.5 * Q.sum(axis=1)
    return J_ising, h, const

def qblib_to_bsb(Q_orig, b_orig, num_vars, device):
    Q = torch.tensor(Q_orig, dtype=torch.float32, device=device)
    Q_sym = 0.5 * (Q + Q.T) 
    
    b = torch.tensor(b_orig, dtype=torch.float32, device=device)
    ones = torch.ones(num_vars, dtype=torch.float32, device=device)

    J_ising = 0.125 * Q_sym
    h_ising = 0.25 * torch.matmul(Q_sym, ones) + 0.5 * b
    
    J_tensor = torch.zeros((num_vars + 1, num_vars + 1), device=device)
    J_tensor[:num_vars, :num_vars] = J_ising
    J_tensor[:num_vars, num_vars] = 0.7 * h_ising
    J_tensor[num_vars, :num_vars] = 0.7 * h_ising
    num_vars += 1
    
    return J_tensor, num_vars

