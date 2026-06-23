import torch, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def parse_file(problem_type, filename, index_start=0, map_type = 'normal'):
    if problem_type in ['maxcut', 'bmincut', 'modularity', 'vertexcover']:
        n, m, J = read_graph(filename, index_start)
    elif problem_type == 'hyperbmincut':
        if map_type == 'normal':
            n, m, J = read_hypergraph(filename, index_start)
        elif map_type == 'star':
            n, m, J = read_hypergraph_star(filename, index_start)
        elif map_type == 'bisecgraph':
            n, m, J = read_hypergraph_bisecgraph(filename, index_start)
    elif problem_type == 'maxksat':
        n, m, J = read_cnf(filename)
    return n, m, J

def load_matrix(path:'str',numer_package:'str',store_format:'str') -> 'float':
    """"
    Load the coupling matrix of the Graph instance from '.txt' file to python matrix.
    
    Parameters:

    :param path - The file path of the graph instance, with format of '.txt';
    :param numer_package - Choose the preferred python sci-package, choices = 'scipy' and 'torch';
    :param store_format - The output matrix will be with the store format of 'store_format', choices = 'csr', 'csc' and 'dense'.
    
 
    Returns:
    
    The output coupling matrix of the instance graph.
    
    """ 
    with open(path, "r") as f:
        l = f.readline()
        N, edges = [int(x) for x in l.split(" ") if x != "\n"]
        
    G = pd.read_csv(path,sep=' ',skiprows=[0],index_col=False, header=None,names=['node1','node2','weight'])
    G.fillna({'weight':int(1)},inplace=True)
    shift = G.iloc[0,0]
    ori_graph = np.array([list(np.concatenate([G.iloc[:,0]-shift,G.iloc[:,1]- shift])),
                 list(np.concatenate([G.iloc[:,1]-shift,G.iloc[:,0]- shift])),
                 list(np.concatenate([G.iloc[:,-1],G.iloc[:,-1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1,:],ori_graph[0,:])).tolist()].T
    if numer_package == 'scipy':
        J = coo_matrix((ori_graph[2,:].tolist(),
                             (ori_graph[0,:].tolist(),ori_graph[1,:].tolist())), shape=(N, N))
        if J.shape[0] != N:
            print("The shape of J does not match N!")
        if J.data.shape[0]/2 != edges:
            print("The number of elements in J does not match edges!")
        if store_format == 'csr':
            J = csr_matrix(J)
        elif store_format == 'csc':
            J = csc_matrix(J)
        elif store_format == 'dense':
            J = J.todense()
        else:
            print("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    elif numer_package == 'torch':
        J = torch.sparse_coo_tensor([ori_graph[0,:].tolist(),
                                 ori_graph[1,:].tolist()], 
                                    ori_graph[2,:].tolist(),(N, N))
        if J.shape[0] != N:
            print("The shape of J does not match N!")
        if J._values().shape[0]/2 != edges:
            print("The number of elements in J does not match edges!")
        
        if store_format == 'csr':
            J = J.to_sparse_csr()
        elif store_format == 'csc':
            J = J.to_sparse_csc()
        elif store_format == 'dense':
            J = J.to_dense()
        else:
            print("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    else:
        print("Error: Input wrong 'numer_package'! Please choose from ['scipy', 'torch'].")
    return J

def load_gset(instance):
    """
    load the weight matrix of Gset, modified from code of Zisong Shen
    """
    # print('loading Gset',instance,'...')
    path = './Gset/' + instance
    G = pd.read_csv(path, sep=' ')
    n_v = int(G.columns[0])
    ori_graph = np.array([list(np.concatenate([G.iloc[:,0]-1,G.iloc[:,1]-1])), list(np.concatenate([G.iloc[:,1]-1,G.iloc[:,0]-1])), list(np.concatenate([G.iloc[:,-1],G.iloc[:,-1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1,:],ori_graph[0,:])).tolist()].T
    J = torch.sparse_coo_tensor([ori_graph[0,:].tolist(), ori_graph[1,:].tolist()], ori_graph[2,:].tolist(),(n_v, n_v)).to_sparse_csr() 
    """ using sparse column here """
    with open('targetvalue.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    result = re.findall(".*"+instance+" (.*).*", content)
    target_value = int(result[0]) if result else 0
    # print('N=%d'%(J.shape[0])," c=%.2f"%(torch.count_nonzero(J.to_dense())*2/J.shape[0]),"best_cut:"+ target_value)
    return J.to_dense(), target_value

def read_graph(file, index_start=0):
    """
    function for reading graph files
    the specific format should be n m in the first line, and m following lines
    represent source end weight
    Parameters:
        file: string, the filenmae of the graph to be readed
        index_start: int, specify which is the start index of the graph
    """
    with open(file,"r") as f:
        l = f.readline()
        n, m = [int(x) for x in l.split(" ") if x!="\n"]
        J = torch.zeros([n, n])
        neighbors = [[] for i in range(n)]
        for k in range(m):
            l = f.readline()
            l_split = l.split()
            i, j = [int(x) for x in l_split[:2]]
            if len(l_split) == 2:
                w = 1.0
            elif len(l_split) == 3:
                w = float(l_split[2])
            else:
                raise ValueError("Unkown graph file format")
            i -= index_start
            j -= index_start
            J[i, j], J[j, i] = w, w
            neighbors[i].append(j)
            neighbors[j].append(i)
    return n, m, J

def read_hypergraph(file, index_start=1):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    all_pairs = []
    all_weights = []
    for vertices in hyperedges:
        if len(vertices) > 1:
            pairs = torch.combinations(torch.tensor(vertices), 2)
            all_pairs.append(pairs)
            pair_weight = 1 / (len(vertices) - 1)
            weights = torch.full((pairs.shape[0],), pair_weight)
            all_weights.append(weights)    

    if len(all_pairs) == 0:
        J = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32), (n, n))
        return n, m, J

    indices = torch.cat(all_pairs, dim=0)
    weights_tensor = torch.cat(all_weights, dim=0)

    indices_symmetric = torch.cat([indices, indices.flip(1)], dim=0)
    weights_symmetric = torch.cat([weights_tensor, weights_tensor], dim=0)

    J_sparse = torch.sparse_coo_tensor(indices_symmetric.t(), weights_symmetric, (n, n)).coalesce()
    max_val = torch.max(torch.abs(J_sparse.values()))
    if max_val > 0:
        J_sparse = torch.sparse_coo_tensor(
            J_sparse.indices(),
            J_sparse.values() / max_val,
            J_sparse.shape,
        ).coalesce()

    return n, m, J_sparse

def hyperedge_list_to_coupling(hyperedges, num_nodes, map_type='clique'):
    """Convert a list of hyperedges to a pairwise coupling matrix.

    Parameters
    ----------
    hyperedges : list of list of int
        Each inner list contains vertex ids belonging to one hyperedge.
    num_nodes : int
        Number of vertices in the hypergraph.
    map_type : str
        ``'clique'`` — all-pairs expansion with 1/(k-1) weight (default).
        ``'star'``   — one auxiliary node per hyperedge, all edges weight 1.

    Returns
    -------
    J : torch.Tensor
        Coupling matrix of shape ``(N, N)`` (clique, dense) or
        ``(N + M, N + M)`` (star, sparse COO).
    """
    if map_type == 'clique':
        return _clique_expansion(hyperedges, num_nodes)
    elif map_type == 'star':
        return _star_expansion(hyperedges, num_nodes)
    else:
        raise ValueError(f"Unknown map_type: {map_type}")


def _clique_expansion(hyperedges, num_nodes):
    """All-pairs clique expansion — each hyperedge becomes a clique with 1/(k-1) weight."""
    all_pairs = []
    all_weights = []
    for vertices in hyperedges:
        if len(vertices) > 1:
            pairs = torch.combinations(torch.tensor(vertices, dtype=torch.long), 2)
            all_pairs.append(pairs)
            pair_weight = 1.0 / (len(vertices) - 1)
            all_weights.append(torch.full((pairs.shape[0],), pair_weight))

    if not all_pairs:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    indices = torch.cat(all_pairs, dim=0)
    weights_tensor = torch.cat(all_weights, dim=0)
    indices_sym = torch.cat([indices, indices.flip(1)], dim=0)
    weights_sym = torch.cat([weights_tensor, weights_tensor], dim=0)

    sparse = torch.sparse_coo_tensor(
        indices_sym.t(), weights_sym, (num_nodes, num_nodes),
    ).coalesce()
    max_val = torch.max(torch.abs(sparse.values()))
    if max_val > 0:
        sparse = torch.sparse_coo_tensor(
            sparse.indices(), sparse.values() / max_val, sparse.shape,
        ).coalesce()
    return sparse.to_dense()


def _star_expansion(hyperedges, num_nodes):
    """Star expansion — one auxiliary node per hyperedge, all edges weight 1.

    Returns a sparse COO tensor to avoid OOM when there are many hyperedges.
    """
    m = len(hyperedges)
    new_n = num_nodes + m
    edge_pairs = []
    for he_idx, vertices in enumerate(hyperedges):
        center = num_nodes + he_idx
        for v in vertices:
            edge_pairs.append((v, center))
            edge_pairs.append((center, v))

    if not edge_pairs:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            (new_n, new_n),
        ).coalesce()

    edges_tensor = torch.tensor(edge_pairs, dtype=torch.long)
    values = torch.ones(edges_tensor.shape[0], dtype=torch.float32)
    return torch.sparse_coo_tensor(
        edges_tensor.t(), values, (new_n, new_n),
    ).coalesce()


def read_hypergraph_star(file, index_start=1):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    new_n = n + m
    all_pairs = []
    
    for he_idx, vertices in enumerate(hyperedges):
        center_node_id = n + he_idx
        
        for node in vertices:
            pair = torch.tensor([center_node_id, node])
            all_pairs.append(pair.unsqueeze(0))
            pair_reverse = torch.tensor([node, center_node_id])
            all_pairs.append(pair_reverse.unsqueeze(0))
    
    indices = torch.cat(all_pairs, dim=0)
    values = torch.ones(indices.shape[0])
    
    J_sparse = torch.sparse_coo_tensor(indices.t(), values, (new_n, new_n))
    J = J_sparse.to_dense()
    
    return new_n, m, J

def read_hypergraph_bisecgraph(file, index_start=1):
    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)
    
    n_new = n + m
    
    # 使用列表推导式批量生成边
    edge_pairs = []
    for he_idx, vertices in enumerate(hyperedges):
        hyperedge_node_id = n + he_idx
        # 为每个原节点生成双向边
        for node in vertices:
            edge_pairs.extend([(node, hyperedge_node_id), (hyperedge_node_id, node)])
    
    # 转换为张量（批量操作）
    edges_tensor = torch.tensor(edge_pairs, dtype=torch.long)
    indices = edges_tensor.t()
    values = torch.ones(indices.shape[1])
    
    J_sparse = torch.sparse_coo_tensor(indices, values, (n_new, n_new))
    J = J_sparse.to_dense()
    
    
    # 使用集合来避免重复，然后排序
    neighbor_sets = [set() for _ in range(n_new)]
    for u, v in edge_pairs:
        neighbor_sets[u].add(v)
    
    return n_new, m, J

def hypergraph_to_cycle(file, index_start=1):

    with open(file, "r") as f:
        l = f.readline()
        m, n = [int(x) for x in l.split(" ") if x != "\n"]
        
        hyperedges = []
        for _ in range(m):
            l = f.readline()
            vertices = [int(x) - index_start for x in l.split() if x != "\n"]
            hyperedges.append(vertices)

    # n, m, hyperedges = read_hypergraph(file, index_start)
    
    # 构建邻接矩阵
    J = torch.zeros((n, n))
    
    for vertices in hyperedges:
        if len(vertices) < 2:
            continue
            
        # 将顶点连接成环
        k = len(vertices)
        for i in range(k):
            u = vertices[i]
            v = vertices[(i + 1) % k]  # 环连接
            weight = 1.0 / k  # 均匀分配权重
            
            J[u, v] += weight
            J[v, u] += weight
    
    return n, m, J

def read_cnf(path):
    with open(path,'r') as f:
        lines = f.readlines()
    k_length_sat_table = {}
    for line in lines:
        l = line.split()
        if l[0] == 'c':  # comment line
            pass
        elif l[0] == 'p': # problem line
            N, M = map(int, l[2:])
        else:
            clause = list(map(int, l[:-1]))
            k = len(clause)
            if k not in k_length_sat_table:
                k_length_sat_table[k] = []
            k_length_sat_table[k].append([])
            k_length_sat_table[k][-1].append(list(map(abs, clause)))
            q_states = []
            for i in range(k):
                if clause[i] > 0:
                    q_states.append(0)    # postive literal
                else:
                    q_states.append(1)     # negative literal
            k_length_sat_table[k][-1].append(q_states)
    sat_table = []
    minimum_index = []
    maximum_index = []
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key] = np.array(k_length_sat_table[key])
        minimum_index.append(np.min(k_length_sat_table[key][:,0,:]))
        maximum_index.append(np.max(k_length_sat_table[key][:,0,:]))
    max_idx = max(maximum_index)
    min_idx = min(minimum_index)
    assert max_idx - min_idx + 1  == N
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key][:,0,:] -= min_idx
        k_length_sat_table[key] = k_length_sat_table[key].tolist()
        sat_table += k_length_sat_table[key]
    real_M = len(sat_table)
    assert real_M  == M
    max_k = max(k_length_sat_table.keys())
    min_k = min(k_length_sat_table.keys())
    if max_k != min_k:
        raise ValueError("This is not a max-ksat instances.")
    mask_tensor = clause_mask_tensor(N, M, sat_table)
    return N, M, mask_tensor

def clause_mask_tensor(N, M, sat_table):
    clause = []
    for ii in range(M):
        k = len(sat_table[ii][0])
        clause_m = torch.sparse_coo_tensor(
            [sat_table[ii][0], sat_table[ii][1]], 
            [1] * k,
            (N, 2)
        ).to_dense().unsqueeze(0)
        clause.append(clause_m.unsqueeze(0))
    clause_batch = torch.cat(clause, dim=0)    #[M, batch, N, q]  # sparse tensor values = k * M * batch
    return clause_batch 

def plot_free_energy(history):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = history['step']
    
    # 1. Free Energy 主图
    ax1.plot(steps, history['free_energy'], 'b-', linewidth=2, label='Free Energy')
    ax1.set_xlabel('Iteration Step')
    ax1.set_ylabel('Free Energy')
    ax1.set_title('Free Energy Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    