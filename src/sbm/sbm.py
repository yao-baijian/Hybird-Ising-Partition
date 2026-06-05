import torch
import math
import numpy as np

from .utils import *

def sb(sb_type, J, init_x, init_y, num_iters, dt):
    N = J.shape[0]
    x_comp = (init_x.copy()) # position
    y_comp = (init_y.copy()) # momentum
    xi = (0.7 / math.sqrt(N)) # xi = 0.5
    sol = np.sign(x_comp)
    energies = []
    e = - 1 / 2 * sol.T @ J @ sol
    alpha = (np.linspace(0, 1, num_iters)) 
    for i in range(num_iters):
        
        # bsb
        if (sb_type == "bsb"):
            y_comp += ((-1 + alpha[i]) * x_comp + xi * (J @ x_comp)) * dt 
            x_comp += y_comp * dt 
            y_comp[np.abs(x_comp) > 1] = 0.
            x_comp = np.clip(x_comp,-1, 1)
        elif (sb_type == "dsb"):
        # dsb
            y_comp += ((-1 + alpha[i]) * x_comp + xi * (J @ x_comp.sign())) * dt
            x_comp += y_comp * dt
            y_comp[x_comp.abs() > 1] = 0.
            x_comp.clamp_(-1, 1)
        elif (sb_type == "sb"):    
        # sb
            y_comp += xi * (J @ x_comp) * dt # y momentum
            # for j in range(M):
            #     y_comp += ((-1 + alpha[i]) * x_comp - x_comp ** 3) * dt / M # alpha equals to alpha
            #     x_comp += y_comp * dt / M #

        sol = np.sign(x_comp)
        e = - 1 / 2 * sol.T @ J @ sol
        energy = -1/4 * J.sum() - 1/2 * e
        energies.append(energy)

    return energies

def bsb_torch(J, init_x, init_y, num_iters, dt):
    x_comp = init_x.clone()
    y_comp = init_y.clone()
    
    N = J.shape[0]
    # xi = torch.tensor(0.7 / (math.sqrt(N) * (torch.std(J) + 1e-8)))
    xi = 0.035 / (math.sqrt(N) * (torch.std(J) + 1e-8))
    # print(f'std J: {torch.std(J):.4f}, xi: {xi:.4f}, xi / std(J): {xi / (torch.std(J) + 1e-8):.4f}')
    energies = []
    alpha = torch.linspace(0, 1, num_iters)
    
    for i in range(num_iters):
        y_comp += ((-1 + alpha[i]) * x_comp + xi * torch.matmul(J, x_comp)) * dt
        x_comp += y_comp * dt
        
        boundary_mask = torch.abs(x_comp) > 1
        y_comp[boundary_mask] = 0.
        x_comp = torch.clamp(x_comp, -1, 1)

        if (i == num_iters - 1):
            sol = torch.sign(x_comp)
            e = -0.5 * torch.matmul(sol, torch.matmul(J, sol))
            energy = -0.25 * torch.sum(J) - 0.5 * e
            energies.append(energy.item())
    
    return energies, sol

def _bsb_step(x_comp, y_comp, J, alpha_i, xi, dt):
    """Single SB iteration step, extracted for optional torch.compile."""
    Jx = torch.matmul(x_comp, J.T)
    y_comp = y_comp + ((-1 + alpha_i) * x_comp + xi * Jx) * dt
    x_comp = x_comp + y_comp * dt
    boundary_mask = torch.abs(x_comp) > 1
    y_comp = torch.where(boundary_mask, torch.zeros_like(y_comp), y_comp)
    x_comp = torch.clamp(x_comp, -1, 1)
    return x_comp, y_comp


def bsb_torch_batch(J, init_x, init_y, num_iters, dt, best_known=None, max_iters=5000, use_compile=False):
    N = J.shape[0]
    batch_size = init_x.shape[0]
    x_comp = init_x.clone()
    y_comp = init_y.clone()
    xi = 4.0 / (torch.sqrt(torch.tensor(N, device=J.device, dtype=torch.float32)))
    use_convergence_mode = best_known is not None
    
    if use_convergence_mode:
        total_iterations = max_iters
        target_energy = best_known * 0.99
        steps_to_converge = torch.full((batch_size,), max_iters, dtype=torch.int32, device=J.device)
    else:
        total_iterations = num_iters

    alpha = torch.linspace(0, 1, total_iterations, device=J.device)
    energies = torch.zeros(batch_size, total_iterations, device=J.device)
    es = torch.zeros(batch_size, total_iterations, device=J.device)
    
    # Optionally compile the step function
    step_fn = torch.compile(_bsb_step, dynamic=True) if use_compile else _bsb_step
    
    for i in range(total_iterations):
        x_comp, y_comp = step_fn(x_comp, y_comp, J, alpha[i], xi, dt)
        
        if use_convergence_mode:
            sol = torch.sign(x_comp)
            J_sol = torch.matmul(sol, J.T)
            e = -0.5 * torch.sum(sol * J_sol, dim=1)
            energy = -0.25 * torch.sum(J) - 0.5 * e
            reached_target = (energy >= target_energy)
            
            if reached_target.any():
                return i + 1, False

        elif i == total_iterations - 1:
            sol = torch.sign(x_comp)
            J_sol = torch.matmul(sol, J.T)
            e = -0.5 * torch.sum(sol * J_sol, dim=1)
            es[:, i] = e
            energy = -0.25 * torch.sum(J) - 0.5 * e
            energies[:, i] = energy
    
    final_solutions = torch.sign(x_comp)
    
    if use_convergence_mode:
        return max_iters, True
    else:
        return energies, final_solutions, es

def bsb_bmincut_batch(J, init_x, init_y, num_iters, dt, lambda_balance=1.0, use_compile=False):
    N = J.shape[0]
    batch_size = init_x.shape[0]
    
    x_comp = init_x.clone()  # [batch_size, n]
    y_comp = init_y.clone()  # [batch_size, n]
    xi = 0.7 / torch.sqrt(torch.tensor(N, device=J.device, dtype=torch.float32))
    
    alpha = torch.linspace(0, 1, num_iters, device=J.device)
    energies = torch.zeros(batch_size, num_iters, device=J.device)
    ones = torch.ones(N, device=J.device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)
    
    step_fn = torch.compile(_bsb_step, dynamic=True) if use_compile else _bsb_step
    
    for i in range(num_iters):
        x_comp, y_comp = step_fn(x_comp, y_comp, J_balanced, alpha[i], xi, dt)
    
    orig_J = -J
    sol = torch.sign(x_comp)
    xJx = torch.einsum('bi,ij,bj->b', sol, orig_J, sol)
    cut_value = 0.25 * (torch.sum(orig_J) - xJx)
    sum_x = torch.sum(sol, dim=1)  # [batch_size]
    balance_term = lambda_balance * sum_x**2  # [batch_size]
    energy = cut_value + balance_term
    energies[:, i] = energy

    return energies, sol, cut_value, sum_x

def qsb(J, init_x, init_y, num_iters, dbg_iter, best_known = 0, factor = [6, 4 ,4], qtz_type = 'scaleup', dbg_option = 'OFF'):

    energies    = []
    scl1        = 2 ** factor[0] - 1
    scl2        = 2 ** 7 - 1

    N           = J.shape[0]
    xi          = 0.75 / (math.sqrt(N) * torch.std(J) + 1e-8)
    # xi          = 1.0 / math.sqrt(N)
    JX_dbg      = []
    alpha       = np.linspace(0, 1, num_iters)
    step        = num_iters
    acc_reach   = 0
    x_comp      = scale_up(np.array(init_x.copy()), scl1)
    y_comp      = scale_up(np.array(init_y.copy()), scl1)
    
    x_comp_init = x_comp.copy() 
    y_comp_init = y_comp.copy() 
    
    x_comp_dbg  = []
    y_comp_dbg  = []
    
    for i in range(num_iters):
        '''
        Note:
        1. All scale up to match with x_comp, the intuitive is to avoid generate any decimal during calculation, 
            and keep Lagrange unchanged
            
        2. Only scale up x_comp and y_comp, scale down when calculating. This will generate decimal during calculation
        
        3. Uniformly quantization, dont scale up x_comp and y_comp, just match them to quantized decimal.
        
            y_comp += ((-1 + alpha[i]) * x_comp + xi * (J @ x_comp)) * dt
            x_comp += y_comp * dt
            y_comp[np.abs(x_comp) > 1] = 0.
            x_comp = np.clip(x_comp,-1, 1)
        
        '''
        
        if i == dbg_iter:
            JX_dbg = (J @ x_comp).astype(int)
            result_sub  = J[0:512, 0:64] @ x_comp[0:64]
            result_sub2 = J[0:512, 64:128] @ x_comp[64:128]
        
        if (qtz_type == 'scaleup'):
            y_comp_div_dt = (-scl2 + alpha[i] * scl2) * x_comp + scale_up((J @ x_comp) * xi, scl2)
            y_comp = y_comp + scale_down(y_comp_div_dt, factor[1])
            x_comp = x_comp + scale_down(y_comp, factor[2])
        elif(qtz_type == 'unscale'):
            y_comp_div_dt = (-1 + alpha[i]) * x_comp + (J @ x_comp) * xi
            y_comp = y_comp + scale_down(y_comp_div_dt,  factor[1])
            x_comp = x_comp + scale_down(y_comp, factor[2])
            
        y_comp[np.abs(x_comp) > scl1] = 0.
        x_comp = np.clip(x_comp, -scl1, scl1)
        
        if i == dbg_iter:
            x_comp_dbg = x_comp.copy()
            y_comp_dbg = y_comp.copy()
        
        sol = np.sign(x_comp)
        e = - 1 / 2 * sol.T @ J @ sol  #
        energy = -1/4 * J.sum() - 1/2 * e
        if (energy > best_known * 0.99) and acc_reach == 0:
            acc_reach = 1
            step = i
        energies.append(energy)
        
    if dbg_option == 'ON':
        print(','.join(map(str, x_comp_init)))
        print(','.join(map(str, y_comp_init)))
        print(','.join(map(str, JX_dbg)))
        print(','.join(map(str, x_comp_dbg)))
        print(','.join(map(str, y_comp_dbg)))
        print(','.join(map(str, result_sub)))
        print(','.join(map(str, result_sub2)))
    
    return np.array(energies), step

def qsb_torch(J, init_x, init_y, num_iters):

    scl = 2 ** 7 - 1         # 用于中间计算的缩放
    scl2 = 4
    
    N = J.shape[0]
    xi = 0.7 / math.sqrt(N)
    
    x_comp = scaleup_torch(init_x.clone(), scl)
    y_comp = scaleup_torch(init_y.clone(), scl)
    
    energies = []
    alpha = torch.linspace(0, 1, num_iters)
    scl2_tensor = torch.tensor(scl, dtype=torch.int32)
    
    for i in range(num_iters):
        Jx = torch.matmul(J.float(), x_comp.float()).to(torch.int32)
        alpha_term = (-scl2_tensor + scaleup_torch(alpha[i], scl)) * x_comp
        # y_update = alpha_term + Jx * xi
        y_update = alpha_term + Jx
        y_comp = y_comp + shiftdown_torch(y_update, scl2)
        x_comp = x_comp + shiftdown_torch(y_comp, scl2)
        
        boundary_mask = torch.abs(x_comp) > scl
        y_comp[boundary_mask] = 0
        x_comp = torch.clamp(x_comp, -scl, scl)
        
        sol = torch.sign(x_comp.float())
        e = -0.5 * torch.matmul(sol, torch.matmul(J, sol))
        energy = -0.25 * torch.sum(J) - 0.5 * e
        energies.append(energy.item())
    
    return torch.tensor(energies)

def qsb_torch_batch(J, init_x, init_y, num_iters, dt_tuned = 16, best_known=None, max_iters=5000):
    scl = 2 ** 7 - 1
    
    N = J.shape[0]
    batch_size = init_x.shape[0]
    # xi = 100 / (math.sqrt(N) * (torch.std(J) + 1e-8))
    xi = 3.5 / (math.sqrt(N) * (torch.std(J) + 1e-8))
    
    # xi = 160 / (math.sqrt(N))
    
    # print(160 / (math.sqrt(N)))
    # print( 2.4 / (math.sqrt(N) * (torch.std(J) + 1e-8)))
    

    x_comp = scaleup_torch(init_x.clone(), scl)
    y_comp = scaleup_torch(init_y.clone(), scl)
    
    alpha = torch.linspace(0, 1, num_iters, device=J.device)
    scl2_tensor = torch.tensor(scl, dtype=torch.int32, device=J.device)

    use_convergence_mode = best_known is not None
    
    if use_convergence_mode:
        total_iterations = max_iters
        target_energy = best_known * 0.99
        steps_to_converge = torch.full((batch_size,), max_iters, dtype=torch.int32, device=J.device)
        
    else:
        total_iterations = num_iters

    energies = torch.zeros(batch_size, num_iters, device=J.device)
    es = torch.zeros(batch_size, num_iters, device=J.device)
    
    for i in range(total_iterations):
        Jx = torch.matmul(x_comp.float(), J.T.float()).to(torch.int32)
        
        alpha_term = (-scl2_tensor + scaleup_torch(alpha[i], scl)) * x_comp
        # y_update = alpha_term + Jx * xi
        y_update = alpha_term + Jx * xi

        y_comp += scaledown_torch(y_update, dt_tuned)
        x_comp += scaledown_torch(y_comp, dt_tuned)

        # y_comp += shiftdown_torch(y_update, scl2)
        # x_comp += shiftdown_torch(y_comp, scl2)
        
        boundary_mask = torch.abs(x_comp) > scl
        y_comp[boundary_mask] = 0
        x_comp = torch.clamp(x_comp, -scl, scl)

        if use_convergence_mode:
            sol = torch.sign(x_comp.float())  
            J_sol = torch.matmul(sol, J.T)  
            e = -0.5 * torch.sum(sol * J_sol, dim=1)  
            energy = -0.25 * torch.sum(J) - 0.5 * e  
            reached_target = (energy >= target_energy)
            
            if reached_target.any():
                return i + 1, False
        
        elif (i == num_iters - 1):
            sol = torch.sign(x_comp.float())  
            J_sol = torch.matmul(sol, J.T)  
            e = -0.5 * torch.sum(sol * J_sol, dim=1)  
            es[:, i] = e
            energy = -0.25 * torch.sum(J) - 0.5 * e  
            energies[:, i] = energy
    
    if use_convergence_mode:
        return max_iters, True

    final_solutions = torch.sign(x_comp.float())
    return energies, final_solutions, es

def qsb_bmincut_torch_batch(J, init_x, init_y, num_iters, dt_tuned=16, lambda_balance=1.0):
    scl = 2 ** 7 - 1
    
    N = J.shape[0]
    batch_size = init_x.shape[0]
    xi = 0.7 / math.sqrt(N)

    x_comp = scaleup_torch(init_x.clone(), scl)
    y_comp = scaleup_torch(init_y.clone(), scl)
    
    alpha = torch.linspace(0, 1, num_iters, device=J.device)
    scl2_tensor = torch.tensor(scl, dtype=torch.int32, device=J.device)
    
    energies = torch.zeros(batch_size, num_iters, device=J.device)
    ones = torch.ones(N, device=J.device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)
    
    for i in range(num_iters):
        Jx = torch.matmul(x_comp.float(), J_balanced.T.float()).to(torch.int32)
        
        alpha_term = (-scl2_tensor + scaleup_torch(alpha[i], scl)) * x_comp
        y_update = alpha_term + Jx

        # 更新动量
        y_comp += scaledown_torch(y_update, dt_tuned)
        x_comp += scaledown_torch(y_comp, dt_tuned)
        
        # 边界处理
        boundary_mask = torch.abs(x_comp) > scl
        y_comp[boundary_mask] = 0
        x_comp = torch.clamp(x_comp, -scl, scl)
    
    orig_J = -J # 恢复原始J矩阵
    sol = torch.sign(x_comp.float())
    xJx = torch.einsum('bi,ij,bj->b', sol, orig_J, sol)
    cut_value = 0.25 * (torch.sum(orig_J) - xJx)
    sum_x = torch.sum(sol, dim=1)
    balance_term = lambda_balance * sum_x**2
    energy = cut_value + balance_term
    energies[:, i] = energy

    return energies, sol, cut_value, sum_x

def qsb_torch_no_tuned_batch(J, init_x, init_y, num_iters, dt):
    scl = 2 ** 7 - 1
    N = J.shape[0]
    xi = 0.7 / math.sqrt(N)
    batch_size = init_x.shape[0]
    
    x_comp = scaleup_torch(init_x.clone(), scl)
    y_comp = scaleup_torch(init_y.clone(), scl)
    
    alpha = torch.linspace(0, 1, num_iters, device=J.device)
    energies = torch.zeros(batch_size, num_iters, device=J.device)
    
    for i in range(num_iters):
        Jx = torch.matmul(x_comp.float(), J.T).to(torch.int32)
        
        alpha_current = (alpha[i] - 1) * scl
        y_update = (alpha_current * x_comp + xi * Jx) * dt
        y_comp += y_update.to(torch.int32)
        x_comp += (y_comp * dt).to(torch.int32)
        
        # 边界处理
        boundary_mask = torch.abs(x_comp) > scl
        y_comp[boundary_mask] = 0
        x_comp = torch.clamp(x_comp, -scl, scl)
        
        if (i == num_iters - 1):
            sol = torch.sign(x_comp.float())
            # 批量计算所有解的能量
            J_sol = torch.matmul(sol, J.T)
            e = -0.5 * torch.sum(sol * J_sol, dim=1)
            energy = -0.25 * torch.sum(J) - 0.5 * e
            energies[:, i] = energy
    
    return energies, torch.sign(x_comp.float())

def scale_up(targets, factor):
    rescaled_targets = []
    for target in targets:
        rescaled_targets.append(int(target * factor))
    return np.array(rescaled_targets)

def scale_down(targets, factor):
    rescaled_targets = []
    for target in targets:
        rescaled_targets.append(np.ceil(target/factor))
    return np.array(rescaled_targets)

def scaleup_torch(x, scale_factor):
    return torch.round(x * scale_factor).to(torch.int32)

def shiftdown_torch(x, shift_bits):
    return (x >> shift_bits).to(torch.int32)

def scaledown_torch(x, scale):
    return (x / scale).to(torch.int32)