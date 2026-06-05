import numpy as np
import torch
from typing import Optional, Tuple
from src.sbm.sbm import bsb_torch_batch

class QIS3:
    """
    QIS3: Quantum-Inspired Solver v3
    Combines Simulated Bifurcation (SB) with Branch & Bound and adaptive perturbation.
    """
    def __init__(
        self,
        J: torch.Tensor,                # Ising coupling matrix (symmetric, diag=0)
        sb_type: str = 'bsb',           # 'bsb' or 'dsb'
        num_iters: int = 1000,
        dt: float = 0.1,
        branch_depth: int = 2,
        popsize: int = 10,
        adaptive: bool = True,
        device: str = 'cpu'
    ):
        self.J = J.to(device)
        self.sb_type = sb_type
        self.num_iters = num_iters
        self.dt = dt
        self.branch_depth = branch_depth
        self.popsize = popsize
        self.adaptive = adaptive
        self.device = device
        self.n = J.shape[0]
        self.best_energy = float('inf')
        self.best_solution = None

    def solve(self, timeout_sec: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Main solve entry.
        Returns: (best_spins (np.ndarray of +-1), best_energy)
        """
        # Step 1: initial population and SB
        init_x = torch.randint(0, 2, (self.popsize, self.n), device=self.device).float() * 2 - 1
        init_y = torch.zeros_like(init_x)
        energies, solutions, _ = bsb_torch_batch(
            self.J, init_x, init_y, self.num_iters, self.dt
        )
        best_idx = torch.argmin(energies[:, -1])
        self.best_solution = solutions[best_idx].cpu().numpy()
        self.best_energy = energies[best_idx, -1].item()

        # Step 2: Branch & Bound
        if self.branch_depth > 0:
            self._branch_and_bound(fixed_vars={}, depth=0)

        # Step 3: Adaptive perturbation
        if self.adaptive:
            self._adaptive_perturbation()

        return self.best_solution, self.best_energy

    def _branch_and_bound(self, fixed_vars: dict, depth: int):
        """Recursive branch and bound."""
        if depth >= self.branch_depth:
            sub_energy, sub_sol = self._solve_subproblem(fixed_vars)
            if sub_energy < self.best_energy:
                self.best_energy = sub_energy
                self.best_solution = sub_sol
            return

        # Select variable to branch (most uncertain)
        var = self._select_branching_variable(fixed_vars)
        if var is None:
            return

        for val in [-1, 1]:
            new_fixed = fixed_vars.copy()
            new_fixed[var] = val
            # Pruning: compute lower bound (simple, can be improved)
            lb = self._estimate_lower_bound(new_fixed)
            if lb >= self.best_energy:
                continue
            self._branch_and_bound(new_fixed, depth + 1)

    def _solve_subproblem(self, fixed_vars: dict) -> Tuple[float, np.ndarray]:
        """Solve reduced problem with fixed spins."""
        fixed_indices = list(fixed_vars.keys())
        free_indices = [i for i in range(self.n) if i not in fixed_indices]
        n_free = len(free_indices)
        if n_free == 0:
            # all fixed
            full_sol = np.zeros(self.n)
            for idx, val in fixed_vars.items():
                full_sol[idx] = val
            e = self._compute_energy(full_sol)
            return e, full_sol

        # Build reduced J matrix
        J_sub = torch.zeros((n_free, n_free), device=self.device, dtype=self.J.dtype)
        # Constant term from fixed-fixed and fixed-free interactions
        const = 0.0
        for i, fi in enumerate(free_indices):
            for j, fj in enumerate(free_indices):
                J_sub[i, j] = self.J[fi, fj]
            # linear term from fixed spins
            for fix_idx, fix_val in fixed_vars.items():
                const += fix_val * self.J[fix_idx, fi] * 0.5  # factor depends on Ising energy formula

        # Run SB on subproblem
        init_x = torch.randint(0, 2, (self.popsize, n_free), device=self.device).float() * 2 - 1
        init_y = torch.zeros_like(init_x)
        energies, solutions, _ = bsb_torch_batch(
            J_sub, init_x, init_y, self.num_iters, self.dt
        )
        best_idx = torch.argmin(energies[:, -1])
        best_sub_energy = energies[best_idx, -1].item()
        best_sub_sol = solutions[best_idx].cpu().numpy()

        # Reconstruct full solution
        full_sol = np.zeros(self.n)
        for i, idx in enumerate(free_indices):
            full_sol[idx] = best_sub_sol[i]
        for idx, val in fixed_vars.items():
            full_sol[idx] = val

        total_energy = best_sub_energy + const
        return total_energy, full_sol

    def _select_branching_variable(self, fixed_vars: dict) -> Optional[int]:
        free = [i for i in range(self.n) if i not in fixed_vars]
        if not free:
            return None
        if self.best_solution is not None:
            # choose variable with smallest |spin| (most uncertain)
            cand = [(i, abs(self.best_solution[i])) for i in free]
            return min(cand, key=lambda x: x[1])[0]
        else:
            return free[0]

    def _estimate_lower_bound(self, fixed_vars: dict) -> float:
        """Very loose lower bound: ignore all quadratic terms among free variables."""
        # For now, return -inf so no pruning. Can be improved with semidefinite relaxation.
        return -float('inf')

    def _adaptive_perturbation(self):
        """Flip random bits and re-run SB if improvement found."""
        for _ in range(5):
            new_sol = self.best_solution.copy()
            flip_mask = np.random.rand(self.n) < 0.2
            new_sol[flip_mask] *= -1
            init_x = torch.tensor(new_sol, dtype=torch.float32, device=self.device).unsqueeze(0)
            init_y = torch.zeros_like(init_x)
            energies, solutions, _ = bsb_torch_batch(
                self.J, init_x, init_y, self.num_iters // 2, self.dt
            )
            new_energy = energies[0, -1].item()
            if new_energy < self.best_energy:
                self.best_energy = new_energy
                self.best_solution = solutions[0].cpu().numpy()
                break

    def _compute_energy(self, spins: np.ndarray) -> float:
        s = torch.tensor(spins, dtype=torch.float32, device=self.device)
        e = -0.5 * torch.matmul(s, torch.matmul(self.J, s))
        return (-0.25 * torch.sum(self.J) - 0.5 * e).item()