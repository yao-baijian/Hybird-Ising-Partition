from test_bmincut_base import *

num_trials = 10
num_steps = 1000
dev = 'cpu'
anneal = 'lin'
manual_grad = False
runs_per_method = 1
enable_multilevel_coarsen_for_kaffpa = True
case_type = 'bmincut'

partition_methods = [
    'direct_fem',
    'kaffpa',
    'coarse_fem_refine_kaffpa',
]

instance_dir = '../partition/gset/'
instances = [f'G{i}' for i in range(1, 2)]
q_values = [2,4,8,16]  # Number of partitions
coarsen_list = [20, 50, 100, 200, 500]

best_rows = []

print_header()

for instance in instances:    
    for q in q_values:
        for partition_method in partition_methods:
            p = None
            best_config = None
            best_row = None
            for coarsen_to in coarsen_list:
                no_coarsen = False
                coarsen_time_s = 0.0
                init_partition_time_s = 0.0
                refine_time_s = 0.0
                start_time = time.perf_counter()
    
                if partition_method == 'direct_fem':
                    case_bmincut = FEM.from_file(case_type, instance_dir + instance, index_start=1)
                    case_bmincut.set_up_solver(num_trials, num_steps, anneal=anneal, dev=dev, q=q, manual_grad=manual_grad)
                    config, result = case_bmincut.solve()
                    
                    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
                    p = config[optimal_inds[0]]
                    fem_eval_cut = result.min().item()
                    no_coarsen = True
    
                elif partition_method == 'metis':
                    init_start = time.perf_counter()
                    case_bmincut = FEM.from_file(case_type, instance_dir + instance, index_start=1)
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    
                    n = J.shape[0]
                    adjacency_list = [[] for _ in range(n)]
                    
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r = int(indices[0, idx])
                        c = int(indices[1, idx])
                        if r != c:  # no self loops
                            adjacency_list[r].append(c)
    
    
                    edgecuts, parts = pymetis.part_graph(q, adjacency=adjacency_list)
                    init_partition_time_s = time.perf_counter() - init_start
                    
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(parts):
                        p[i, p_group] = 1.0
                        
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
                    fem_eval_cut = fem_eval_cut.item()
                    no_coarsen = True
                    
                elif partition_method == 'coarse_fem_refine_metis':
                    case_bmincut = FEM.from_file(case_type, instance_dir + instance, index_start=1)
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
                    
                    # 1. Multi-level coarsening
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
                        J,
                        node_weights=torch.ones(n, dtype=torch.float32),
                        coarsen_to=coarsen_to,
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
                    
                    initial_assignment = expand_coarse_labels(coarse_groups, coarse_assignment, n)
                    
                    refine_start = time.perf_counter()
                    adjacency_list = [[] for _ in range(n)]
                    indices = J.indices()
                    for idx in range(indices.shape[1]):
                        r = int(indices[0, idx])
                        c = int(indices[1, idx])
                        if r != c:  
                            adjacency_list[r].append(c)
    
                    edgecuts, parts = call_pymetis_with_part(q, adjacency_list, part=initial_assignment.tolist())
                    refine_time_s = time.perf_counter() - refine_start
                    p = torch.zeros((n, q), dtype=J.dtype, device=J.device)
                    for i, p_group in enumerate(parts):
                        p[i, p_group] = 1.0
                        
                    _, fem_eval_cut = infer_bmincut(J, p.unsqueeze(0))
    
                elif partition_method == 'coarse_fem_refine_kaffpa':
                    case_bmincut = FEM.from_file(case_type, instance_dir + instance, index_start=1)
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
    
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
                    )
                    coarsen_time_s = time.perf_counter() - coarsen_start
                    num_coarse_nodes = coarse_graph.shape[0]
    
                    # Use FEM to produce a q-way coarse initial partition so KaFFPa only refines

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
                    case_bmincut = FEM.from_file(case_type, instance_dir + instance, index_start=1)
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
                    
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
                        J, node_weights=torch.ones(n, dtype=torch.float32), coarsen_to=coarsen_to
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
                    case_bmincut = FEM.from_file(case_type, instance_dir + instance, index_start=1)
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
    
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
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
    
                elif partition_method == 'coarse_kaffpa_refine_fem':
                    import kahip
                    J = case_bmincut.problem.coupling_matrix
                    if not J.is_sparse:
                        J = J.to_sparse()
                    J = J.coalesce()
                    n = J.shape[0]
    
                    # Stage 1: Multi-level coarsening using matching-based coarsening
                    coarsen_start = time.perf_counter()
                    coarse_graph, coarse_node_weights, coarse_groups, original_to_coarse = coarsen_graph_by_matching(
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
                    'instance': instance,
                    'q': q,
                    'partition_method': partition_method,
                    'coarsen_to': coarsen_to if not no_coarsen else 0,
                    'cut_value': cut_value,
                    'imbalance': imbalance,
                    'total_time_s': total_time_s,
                    'coarsen_time_s': coarsen_time_s,
                    'init_partition_time_s': init_partition_time_s,
                    'refine_time_s': refine_time_s,
                }
                
                best_rows.append(row)
                print_row(row)
                if no_coarsen:
                    break

save_to_csv(best_rows)

