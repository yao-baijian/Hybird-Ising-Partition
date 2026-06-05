import sys
sys.path.append('.')
sys.path.append('tests')
from src.fem import FEM
import torch
import time
import numpy as np
import warnings
import os
import csv
from datetime import datetime
from utils import simple_kaffpa, coarsen_graph_by_matching, expand_coarse_labels, call_pymetis_with_part
from src.fem.problem import infer_bmincut
from src.fem.cyclic_expansion import cyclic_expansion_refine, adjacency_from_sparse

try:
    import pymetis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    warnings.warn("pymetis is not installed. METIS mode will fail.")

num_trials = 1
num_steps = 1000
dev = 'cpu'
anneal = 'inverse'
manual_grad = False
runs_per_method = 1

partition_methods = [
    'direct_fem',
    'kaffpa',
    'coarse_fem_refine_kaffpa',
    'coarse_kaffpa_refine_fem'
]

# normal graph instances
instances = [
    'tests/test_instances/G1.txt',
    '../partition/gset/G2',
]
# instance = '../partition/data/ash219/ash219.mtx'
case_type = 'bmincut'
q_values = [2,4,8]  # Number of partitions

# Enable multi-level coarsening for kaffpa (and FEM+KaFFPa uses coarsening
# by design). Set to False to run vanilla KaFFPa on the full graph.
enable_multilevel_coarsen_for_kaffpa = True
coarsen_to = 500

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
build_dir = 'build'
os.makedirs(build_dir, exist_ok=True)
csv_path = os.path.join(build_dir, f'bmincut_results_best_{timestamp}.csv')
fieldnames = [
    'instance',
    'q',
    'partition_method',
    'cut_value',
    'imbalance',
    'total_time_s',
    'coarsen_time_s',
    'init_partition_time_s',
    'refine_time_s',
]
best_rows = []
for instance in instances:
    # Use FEM parser to easily load the normal graph
    case_bmincut = FEM.from_file(case_type, instance, index_start=1)
    for q in q_values:
        for partition_method in partition_methods:
            p = None
            best_config = None
            best_row = None

            # Print table header once before first method (fixed-width columns)
            if partition_method == partition_methods[0]:
                col_w = (24, 4, 22, 10, 12, 10)  # instance, q, method, time, cut, imbalance
                header_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:<{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}} {{:>{col_w[5]}}}"
                sep = ' '.join(['-' * w for w in col_w])
                print(header_fmt.format('instance', 'q', 'method', 'time(s)', 'cut', 'imbalance'))
                print(sep)

            for run_idx in range(runs_per_method):
                coarsen_time_s = 0.0
                init_partition_time_s = 0.0
                refine_time_s = 0.0
                start_time = time.perf_counter()

                # print(f'\n====== Evaluating Method: {partition_method} ======')
                if partition_method == 'direct_fem':
                    case_bmincut.set_up_solver(num_trials, num_steps, anneal=anneal, dev=dev, q=q, manual_grad=manual_grad)
                    config, result = case_bmincut.solve()
                    
                    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
                    p = config[optimal_inds[0]]
                    fem_eval_cut = result.min().item()

                elif partition_method == 'metis':
                    if not HAS_METIS:
                        raise ImportError("pymetis is required for 'metis' partition method")
                    init_start = time.perf_counter()
                    
                    # We construct the adjacency dict/list for pymetis using the FEM sparse tensor
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    
                    n = J.shape[0]
                    adjacency_list = [[] for _ in range(n)]
                    
                    indices = J.indices()
                    # It's an unweighted graph typically for normal METIS or we can pass xadj/adjncy.
                    # pymetis.part_graph accepts adjacency list of lists
                    for idx in range(indices.shape[1]):
                        r = int(indices[0, idx])
                        c = int(indices[1, idx])
                        if r != c:  # no self loops
                            adjacency_list[r].append(c)

                    # metis
                    edgecuts, parts = pymetis.part_graph(q, adjacency=adjacency_list)
                    # suppressed intermediate prints
                    init_partition_time_s = time.perf_counter() - init_start
                    
                    # evaluate METIS assignment cut with FEM traditional bmincut cut
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(parts):
                        p[i, p_group] = 1.0
                        
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                    fem_eval_cut = fem_eval_cut.item()
                    
                    # suppressed intermediate prints

                elif partition_method == 'coarse_fem_refine_metis':
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
                    
                    # 1. Multi-level coarsening
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J,
                        node_weights=torch.ones(n, dtype=torch.float32),
                        coarsen_to=500,
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    
                    num_coarse_nodes = coarse_graph.shape[0]
                    
                    # 2. FEM solver on coarse graph
                    init_start = time.perf_counter()
                    case_bmincut_coarse = FEM.from_couplings(
                        'bmincut',
                        num_coarse_nodes,
                        int(coarse_graph._nnz() // 2),
                        coarse_graph,
                        node_weights=coarse_node_weights,
                    )
                    case_bmincut_coarse.set_up_solver(num_trials, num_steps, dev=dev, q=q)
                    config, result = case_bmincut_coarse.solve()
                    
                    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
                    best_config = config[optimal_inds[0]]
                    coarse_assignment = best_config.argmax(dim=1).cpu().numpy()
                    init_partition_time_s = time.perf_counter() - init_start
                    
                    # 3. Projection to original graph
                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
                    
                    # 4. METIS refinement step
                    if not HAS_METIS:
                        raise ImportError("pymetis is required for 'coarse_fem_refine_metis' partition method")
                    
                    refine_start = time.perf_counter()
                    adjacency_list = [[] for _ in range(n)]
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r = int(indices[0, idx])
                        c = int(indices[1, idx])
                        if r != c:  
                            adjacency_list[r].append(c)

                    # pymetis: call wrapper that passes `part` if supported, otherwise
                    # emits a clear warning and calls without initial partition.
                    edgecuts, parts = call_pymetis_with_part(q, adjacency_list, part=initial_assignment.tolist())
                    refine_time_s = time.perf_counter() - refine_start

                    # suppressed intermediate prints
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(parts):
                        p[i, p_group] = 1.0
                        
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                    # suppressed intermediate prints

                elif partition_method == 'coarse_fem_refine_kahypar':
                    try:
                        import kahypar
                    except ImportError:
                        raise ImportError("kahypar is required for 'coarse_fem_refine_kahypar' partition method")
                    
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
                    
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=500
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]
                    
                    # Prefer FEM-based Ising/QUBO initial partition on the coarse graph
                    # implemented in FEM.initial_partition.fem_initial_partition (k=2).
                    # Fall back to the previous FEM-on-coarse solver if anything fails.
                    init_start = time.perf_counter()
                    try:
                        from src.fem.initial_partition import fem_initial_partition

                        # Convert sparse coarse_graph to dense numpy adjacency for the QUBO builder
                        try:
                            coarse_adj_np = coarse_graph.to_dense().cpu().numpy()
                        except Exception:
                            # If coarse_graph is already dense tensor
                            coarse_adj_np = coarse_graph.cpu().numpy()

                        c_np = coarse_node_weights.cpu().numpy().reshape(-1)
                        coarse_assignment = fem_initial_partition(
                            coarse_adj_np,
                            None,
                            None,
                            c_np,
                            k=2,
                            lambda_penalty=1.0,
                            num_trials=num_trials,
                            num_steps=num_steps,
                            dev=dev,
                        )
                    except Exception:
                        # Fallback: run previous coarse FEM solver
                        case_bmincut_coarse = FEM.from_couplings(
                            'bmincut', num_coarse_nodes, int(coarse_graph._nnz() // 2), coarse_graph, node_weights=coarse_node_weights
                        )
                        case_bmincut_coarse.set_up_solver(num_trials, num_steps, dev=dev, q=q)
                        config, result = case_bmincut_coarse.solve()
                        coarse_assignment = config[torch.argwhere(result==result.min()).reshape(-1)[0]].argmax(dim=1).cpu().numpy()
                    init_partition_time_s = time.perf_counter() - init_start
                    
                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
                    
                    # Kahypar Refinement
                    refine_start = time.perf_counter()
                    hyperedges = []
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r, c = int(indices[0, idx]), int(indices[1, idx])
                        if r < c:
                            hyperedges.append([r, c])
                            
                    num_hyperedges = len(hyperedges)
                    hyperedge_indices = []
                    hyperedge_indices_ptrs = [0]
                    for he in hyperedges:
                        hyperedge_indices.extend(he)
                        hyperedge_indices_ptrs.append(len(hyperedge_indices))
                        
                    hypergraph = kahypar.Hypergraph(n, num_hyperedges, hyperedge_indices, hyperedge_indices_ptrs, q, [1]*num_hyperedges, [1]*n)
                    for i in range(n):
                        hypergraph.setNodePart(i, int(initial_assignment[i]))
                        
                    context = kahypar.Context()
                    try:
                        context.loadINIconfiguration("kahypar_config.ini")
                    except:
                        pass # use defaults
                    context.setK(q)
                    context.setEpsilon(0.05)
                    
                    # Improve partition based on the initial block assignments
                    kahypar.improvePartition(hypergraph, context)
                    part = [hypergraph.blockID(i) for i in range(n)]
                    refine_time_s = time.perf_counter() - refine_start

                    # suppressed intermediate prints
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(part):
                        p[i, p_group] = 1.0
                        
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                    # suppressed intermediate prints

                elif partition_method == 'coarse_fem_refine_kaffpa':
                    import kahip

                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]

                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=500
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]

                    # Use FEM to produce a q-way coarse initial partition so KaFFPa only refines
                    from src.fem.initial_partition import fem_initial_partition_kway
                    init_start = time.perf_counter()
                    try:
                        # Convert sparse coarse_graph to dense numpy for the helper
                        try:
                            coarse_adj_np = coarse_graph.to_dense().cpu().numpy()
                        except Exception:
                            coarse_adj_np = coarse_graph.cpu().numpy()

                        c_np = coarse_node_weights.cpu().numpy().reshape(-1)
                        coarse_assignment = fem_initial_partition_kway(
                            coarse_adj_np,
                            None,
                            None,
                            c_np,
                            k=q,
                            lambda_penalty=1.0,
                            num_trials=num_trials,
                            num_steps=num_steps,
                            dev=dev,
                        )
                    except Exception as e:
                        # Let exceptions propagate to surface FEM issues (no silent fallback)
                        raise
                    init_partition_time_s = time.perf_counter() - init_start

                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

                    refine_start = time.perf_counter()
                    adjacency_list = [[] for _ in range(n)]
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r, c = int(indices[0, idx]), int(indices[1, idx])
                        if r != c:  
                            adjacency_list[r].append(c)

                    xadj = [0]
                    adjncy = []
                    for r in range(n):
                        adjncy.extend(adjacency_list[r])
                        xadj.append(len(adjncy))

                    vwgt = [1] * n
                    adjcwgt = [1] * len(adjncy)

                    # Use local simple refinement (KL/FM-like) on top of FEM initial partition
                    edgecut, part = simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, part=initial_assignment.tolist(), max_passes=10)
                    refine_time_s = time.perf_counter() - refine_start

                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(part):
                        p[i, p_group] = 1.0

                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))

                elif partition_method == 'kaffpa':
                    try:
                        import kahip
                    except ImportError:
                        raise ImportError("kahip is required for 'kaffpa' partition method")
                    
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
                    
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=200
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]
                    
                    coarse_adj = [[] for _ in range(num_coarse_nodes)]
                    c_indices = coarse_graph.indices()
                    c_values = coarse_graph.values()
                    
                    for idx in range(c_indices.shape[1]):
                        r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
                        if r != c:  
                            coarse_adj[r].append((c, int(c_values[idx].item())))
                    
                    c_xadj = [0]
                    c_adjncy = []
                    c_adjcwgt = []
                    for r in range(num_coarse_nodes):
                        for c, w in coarse_adj[r]:
                            c_adjncy.append(c)
                            c_adjcwgt.append(w)
                        c_xadj.append(len(c_adjncy))
                        
                    c_vwgt = coarse_node_weights.int().cpu().numpy().tolist()
                    
                    init_start = time.perf_counter()
                    edgecut, coarse_assignment = simple_kaffpa(c_vwgt, c_xadj, c_adjcwgt, c_adjncy, q, epsilon=0.05, max_passes=10)
                    coarse_assignment = np.array(coarse_assignment)
                    init_partition_time_s = time.perf_counter() - init_start
                    
                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
                    
                    adjacency_list = [[] for _ in range(n)]
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r, c = int(indices[0, idx]), int(indices[1, idx])
                        if r != c:  
                            adjacency_list[r].append(c)
                            
                    xadj = [0]
                    adjncy = []
                    for r in range(n):
                        adjncy.extend(adjacency_list[r])
                        xadj.append(len(adjncy))
                        
                    vwgt = [1] * n
                    adjcwgt = [1] * len(adjncy)
                    
                    # Use local simple refinement (KL/FM-like) on top of FEM initial partition
                    refine_start = time.perf_counter()
                    edgecut, part = simple_kaffpa(vwgt, xadj, adjcwgt, adjncy, q, epsilon=0.05, part=initial_assignment.tolist(), max_passes=10)
                    refine_time_s = time.perf_counter() - refine_start

                    # suppressed intermediate prints
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(part):
                        p[i, p_group] = 1.0

                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                    # suppressed intermediate prints

                elif partition_method == 'coarse_metis_refine_fem':
                    # Coarsen, run METIS on coarse graph, then refine with FEM cyclic expansion
                    if not HAS_METIS:
                        raise ImportError("pymetis is required for 'coarse_metis_refine_fem' partition method")
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]

                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]

                    # Build adjacency list for coarse METIS
                    c_indices = coarse_graph.indices()
                    c_values = coarse_graph.values()
                    coarse_adj_list = [[] for _ in range(num_coarse_nodes)]
                    for idx in range(c_indices.shape[1]):
                        r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
                        if r != c:
                            coarse_adj_list[r].append(c)

                    init_start = time.perf_counter()
                    edgecuts, coarse_parts = pymetis.part_graph(num_coarse_nodes and q or q, adjacency=coarse_adj_list)
                    coarse_assignment = np.array(coarse_parts)
                    init_partition_time_s = time.perf_counter() - init_start

                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

                    # Run cyclic expansion FEM refinement on the full graph
                    refine_start = time.perf_counter()
                    adjacency = adjacency_from_sparse(J)
                    refined_assignment = cyclic_expansion_refine(
                        adjacency,
                        initial_assignment,
                        q,
                        max_iterations=50,
                        max_candidates=60,
                        num_trials=num_trials,
                        num_steps=num_steps,
                        dev=dev,
                        patience=10,
                        verbose=True,
                        allow_nonadjacent=True,
                    )
                    refine_time_s = time.perf_counter() - refine_start

                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i in range(n):
                        p[i, refined_assignment[i]] = 1.0

                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))

                elif partition_method == 'coarse_kahypar_refine_fem':
                    # Coarsen, run KaHyPar on coarse graph, then refine with FEM cyclic expansion
                    try:
                        import kahypar
                    except ImportError:
                        raise ImportError("kahypar is required for 'coarse_kahypar_refine_fem' partition method")
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]

                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]

                    # Build hyperedges for coarse graph
                    c_indices = coarse_graph.indices()
                    hyperedges = []
                    for idx in range(c_indices.shape[1]):
                        r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
                        if r < c:
                            hyperedges.append([r, c])

                    num_hyperedges = len(hyperedges)
                    hyperedge_indices = []
                    hyperedge_indices_ptrs = [0]
                    for he in hyperedges:
                        hyperedge_indices.extend(he)
                        hyperedge_indices_ptrs.append(len(hyperedge_indices))

                    init_start = time.perf_counter()
                    hypergraph = kahypar.Hypergraph(num_coarse_nodes, num_hyperedges, hyperedge_indices, hyperedge_indices_ptrs, q, [1]*num_hyperedges, [1]*num_coarse_nodes)
                    context = kahypar.Context()
                    try:
                        context.loadINIconfiguration("kahypar_config.ini")
                    except:
                        pass
                    context.setK(q)
                    context.setEpsilon(0.05)

                    kahypar.partition(hypergraph, context)
                    coarse_parts = [hypergraph.blockID(i) for i in range(num_coarse_nodes)]
                    coarse_assignment = np.array(coarse_parts)
                    init_partition_time_s = time.perf_counter() - init_start

                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

                    refine_start = time.perf_counter()
                    adjacency = adjacency_from_sparse(J)
                    refined_assignment = cyclic_expansion_refine(
                        adjacency,
                        initial_assignment,
                        q,
                        max_iterations=50,
                        max_candidates=60,
                        num_trials=num_trials,
                        num_steps=num_steps,
                        dev=dev,
                        patience=10,
                        verbose=True,
                        allow_nonadjacent=True,
                    )
                    refine_time_s = time.perf_counter() - refine_start

                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i in range(n):
                        p[i, refined_assignment[i]] = 1.0
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))

                elif partition_method == 'coarse_kaffpa_refine_fem':
                    import kahip
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]

                    # Stage 1: Multi-level coarsening using matching-based coarsening
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse, _ = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]

                    # Build adjacency for coarse graph for KaFFPa
                    coarse_adj = [[] for _ in range(num_coarse_nodes)]
                    c_indices = coarse_graph.indices()
                    c_values = coarse_graph.values()

                    for idx in range(c_indices.shape[1]):
                        r, c = int(c_indices[0, idx]), int(c_indices[1, idx])
                        if r != c:
                            coarse_adj[r].append((c, int(c_values[idx].item())))

                    c_xadj = [0]
                    c_adjncy = []
                    c_adjcwgt = []
                    for r in range(num_coarse_nodes):
                        for c, w in coarse_adj[r]:
                            c_adjncy.append(c)
                            c_adjcwgt.append(w)
                        c_xadj.append(len(c_adjncy))

                    c_vwgt = coarse_node_weights.int().cpu().numpy().tolist()

                    # Stage 2: Run KaFFPa on coarse graph to get initial q-way partition
                    init_start = time.perf_counter()
                    edgecut, coarse_assignment = simple_kaffpa(c_vwgt, c_xadj, c_adjcwgt, c_adjncy, q, epsilon=0.05, max_passes=10)
                    coarse_assignment = np.array(coarse_assignment)
                    init_partition_time_s = time.perf_counter() - init_start

                    # Stage 3: Project coarse partition back to original graph
                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)

                    # Stage 4: Cyclic Expansion QUBO refinement using FEM
                    # Convert sparse coupling matrix to adjacency list format
                    refine_start = time.perf_counter()
                    adjacency = adjacency_from_sparse(J)
                    
                    num_steps_cyclic = 100

                    # Run Cyclic Expansion refinement
                    refined_assignment = cyclic_expansion_refine(
                        adjacency,
                        initial_assignment,
                        q,
                        max_iterations=50,
                        max_candidates=60,
                        num_trials=num_trials,
                        num_steps=num_steps_cyclic,
                        dev=dev,
                        patience=10,
                        verbose=False,
                        allow_nonadjacent = True
                    )
                    refine_time_s = time.perf_counter() - refine_start

                    # Build output tensor
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i in range(n):
                        p[i, refined_assignment[i]] = 1.0

                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))

                elif partition_method == 'kahypar':
                    try:
                        import kahypar
                    except ImportError:
                        raise ImportError("kahypar is required for 'kahypar' partition method")
                    init_start = time.perf_counter()
                    
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
                    hyperedges = []
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r, c = int(indices[0, idx]), int(indices[1, idx])
                        if r < c:
                            hyperedges.append([r, c])
                            
                    num_hyperedges = len(hyperedges)
                    hyperedge_indices = []
                    hyperedge_indices_ptrs = [0]
                    for he in hyperedges:
                        hyperedge_indices.extend(he)
                        hyperedge_indices_ptrs.append(len(hyperedge_indices))
                        
                    hypergraph = kahypar.Hypergraph(n, num_hyperedges, hyperedge_indices, hyperedge_indices_ptrs, q, [1]*num_hyperedges, [1]*n)
                    context = kahypar.Context()
                    try:
                        context.loadINIconfiguration("kahypar_config.ini")
                    except:
                        pass
                    context.setK(q)
                    context.setEpsilon(0.05)
                    
                    kahypar.partition(hypergraph, context)
                    part = [hypergraph.blockID(i) for i in range(n)]
                    init_partition_time_s = time.perf_counter() - init_start

                    # suppressed intermediate prints
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(part):
                        p[i, p_group] = 1.0
                        
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                    # suppressed intermediate prints

                else:
                    raise ValueError(f"Unknown partition method: {partition_method}")

                J = case_bmincut.problem.coupling_matrix
                n = J.shape[0]
                final_assignment = p.argmax(dim=1).cpu().numpy()
                counts = np.bincount(final_assignment, minlength=q)
                ideal = n / q
                imbalance = float(np.max(np.abs(counts - ideal) / ideal))

                # Evaluate cut value via FEM's infer_bmincut to ensure consistent metric
                try:
                    cut_value = float(fem_eval_cut.item())
                except Exception:

                    cut_value = float(fem_eval_cut)

                total_time_s = time.perf_counter() - start_time
                row = {
                    'instance': os.path.basename(instance),
                    'q': q,
                    'partition_method': partition_method,
                    'cut_value': cut_value,
                    'imbalance': imbalance,
                    'total_time_s': total_time_s,
                    'coarsen_time_s': coarsen_time_s,
                    'init_partition_time_s': init_partition_time_s,
                    'refine_time_s': refine_time_s,
                }

                if best_row is None:
                    best_row = row
                else:
                    if row['cut_value'] < best_row['cut_value']:
                        best_row = row
                    elif row['cut_value'] == best_row['cut_value'] and row['total_time_s'] < best_row['total_time_s']:
                        best_row = row

            best_rows.append(best_row)
            col_w = (24, 4, 22, 10, 12, 10)
            row_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:<{col_w[2]}}} {{:>{col_w[3]}.4f}} {{:>{col_w[4]}.1f}} {{:>{col_w[5]}.4f}}"
            print(row_fmt.format(best_row['instance'], best_row['q'], best_row['partition_method'], best_row['total_time_s'], best_row['cut_value'], best_row['imbalance']))

with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in best_rows:
        writer.writerow(row)

print(f"Saved best results to: {csv_path}")
