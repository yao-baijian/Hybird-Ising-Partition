import torch
import math

def digcim_bmincut_batch(J, init_x, init_y, num_iters, dt, lambda_balance=1.0, use_compile=False):
    """
    Batch version of the digitized Coherent Ising Machine (digCIM) for graph partitioning.
    
    The dynamics follow the first-order ODE:
        dx_i/dt = a * φ(x_i) + ξ * Σ_j (J_balanced)_{ij} * sign(x_j)
    where φ(x) = x if |x|<=1 else sign(x), and a is annealed from -1 to 0.
    
    Args:
        J (torch.Tensor): Coupling matrix of shape (N, N). For graph partitioning,
                          this is typically the adjacency matrix A (or a precomputed Ising matrix).
        init_x (torch.Tensor): Initial positions of shape (batch_size, N).
        init_y (torch.Tensor): Initial momenta (unused in digCIM, kept for API compatibility).
        num_iters (int): Number of integration steps.
        dt (float): Time step size.
        lambda_balance (float): Lagrange multiplier for partition balance constraint.
        use_compile (bool): Whether to torch.compile the step function (for speed).
    
    Returns:
        energies (torch.Tensor): Energy (cut+balance) history, shape (batch_size, num_iters).
                                 Only the last column is non-zero (compatible with bsb_bmincut_batch).
        sol (torch.Tensor): Final binary solution, shape (batch_size, N), values in {-1, 1}.
        cut_value (torch.Tensor): Edge cut value for each batch element, shape (batch_size,).
        sum_x (torch.Tensor): Sum of spins (imbalance) for each batch element, shape (batch_size,).
    """
    N = J.shape[0]
    batch_size = init_x.shape[0]
    device = J.device
    dtype = torch.float32

    # Make a working copy; initial values are assumed to be in [-1, 1] range already
    x = init_x.clone().to(dtype)
    # y is unused but kept for signature compatibility
    # (init_y is ignored in digCIM)

    # Build the effective coupling matrix that includes the balance penalty term.
    # Same construction as in bsb_bmincut_batch.
    ones = torch.ones(N, device=device, dtype=dtype)
    # The balanced matrix used in dynamics: J_balanced = -0.5 * J - 2λ * 𝟙𝟙ᵀ
    # Note: we keep the original J for final cut evaluation (orig_J = -J)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)

    # Normalization factor ξ as suggested in the paper:
    # ξ = 0.5 * sqrt( (N-1) / Σ_{ij} J_ij^2 )
    frob_norm_sq = torch.sum(J_balanced ** 2)
    xi = 0.5 * math.sqrt(max(1, N - 1) / (frob_norm_sq + 1e-12))

    # Annealing schedule for the control parameter a(t):
    # a starts at -1 (strong negative) and linearly increases to 0 over the iterations.
    alpha = torch.linspace(0, 1, num_iters, device=device, dtype=dtype)
    a = -1.0 + alpha  # a from -1 to 0

    # Pre-allocate energy history (only last column will be set)
    energies = torch.zeros(batch_size, num_iters, device=device, dtype=dtype)

    # Define the phi function (clipping to [-1,1] with identity in the linear region)
    def phi(v):
        # v: arbitrary tensor
        return torch.where(torch.abs(v) <= 1.0, v, torch.sign(v))

    # Single step for digCIM (can be compiled)
    def _step(x, a_val, xi, J_bal, dt):
        # Compute sign of current positions
        sign_x = torch.sign(x)
        # Compute the coupling term: (J_balanced @ sign(x))   [batch matrix multiply]
        J_sign = torch.matmul(sign_x, J_bal.T)  # shape (batch, N)
        # Dynamics: dx/dt = a*φ(x) + ξ * (J @ sign(x))
        dxdt = a_val * phi(x) + xi * J_sign
        x_new = x + dt * dxdt
        # Optional clipping to maintain numeric stability (many CIM implementations clip)
        x_new = torch.clamp(x_new, -1.0, 1.0)
        return x_new

    # Optionally compile the step function
    if use_compile:
        _step = torch.compile(_step, dynamic=True)

    # Main iteration loop
    for i in range(num_iters):
        x = _step(x, a[i], xi, J_balanced, dt)

        # On the final iteration, compute the cut value, balance term, and energy
        if i == num_iters - 1:
            sol = torch.sign(x)                     # binary decision
            # Original coupling matrix for cut evaluation (note: orig_J = -J)
            orig_J = -J
            # Compute xJx = σᵀ J σ  (using original J)
            xJx = torch.einsum('bi,ij,bj->b', sol, orig_J, sol)
            cut_value = 0.25 * (torch.sum(orig_J) - xJx)
            sum_x = torch.sum(sol, dim=1)
            balance_term = lambda_balance * (sum_x ** 2)
            energy = cut_value + balance_term
            energies[:, i] = energy

    # The returned energies contain zeros except for the last column, exactly
    # matching the behavior of bsb_bmincut_batch.
    return energies, sol, cut_value, sum_x


# Optional: expose a simpler alias if desired
digcim_bmincut_batch.__doc__ = """
Batch digCIM solver for graph partitioning / Ising problems.

Implements the digitized Coherent Ising Machine (digCIM) dynamics with a
balanced penalty term for two-way partitioning.

Parameters:
    J: square coupling matrix (N x N)
    init_x: initial analog positions (batch_size x N) in [-1, 1]
    init_y: placeholder for API compatibility (unused)
    num_iters: number of integration steps
    dt: time step size
    lambda_balance: strength of the partition balance constraint
    use_compile: if True, torch.compile is applied to the step kernel

Returns:
    energies: history of energy values (batch_size x num_iters), non-zero only at last step
    sol: final binary solution (batch_size x N) in {-1, +1}
    cut_value: edge cut for each sample (batch_size,)
    sum_x: sum of spins (imbalance) for each sample (batch_size,)
"""