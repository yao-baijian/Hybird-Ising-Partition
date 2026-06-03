import sys
sys.path.append('.')
sys.path.append('tests')
from fem import FEM
import torch
import time
import numpy as np
import warnings
import os
import csv
from datetime import datetime
from utils import simple_kaffpa, coarsen_graph_by_matching, expand_coarse_labels, call_pymetis_with_part
from fem.problem import infer_bmincut
from fem.cyclic_expansion import cyclic_expansion_refine, adjacency_from_sparse
from fem.initial_partition import fem_initial_partition_kway

try:
    import pymetis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    warnings.warn("pymetis is not installed.")

# try:
#     import metis
#     HAS_METIS = True
# except ImportError:
#     HAS_METIS = False
#     warnings.warn("metis is not installed.")

try:
    import kahip
    HAS_KAHIP = True
except ImportError:
    HAS_KAHIP = False
    warnings.warn("kahip is not installed.")

try:
    import kahypar
    HAS_KAHYPAR = True
except ImportError:
    HAS_KAHYPAR = False
    warnings.warn("kahypar is not installed.")
    
# ==========================================
# Select the partition method to run:
# 'direct_fem'                : Original FEM applied directly to normal graph
# 'coarse_fem_refine_metis'   : Multi-level coarsening + FEM coarse opt + METIS fine opt
# 'coarse_fem_refine_kahypar' : Multi-level coarsening + FEM coarse opt + KaHyPar fine opt
# 'coarse_fem_refine_kaffpa'  : Multi-level coarsening + FEM coarse opt + KaFFPa fine opt
# 'coarse_kaffpa_refine_fem'  : Multi-level coarsening + KaFFPa coarse opt + Cyclic Expansion FEM fine opt
# 'metis'                     : PyMetis graph partitioner alone
# 'kahypar'                   : KaHyPar partitioner alone
# 'kaffpa'                    : KaFFPa partitioner alone
# ==========================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
build_dir = 'build'
os.makedirs(build_dir, exist_ok=True)
csv_path = os.path.join(build_dir, f'bmincut_results_best_{timestamp}.csv')

fieldnames = [
    'instance',
    'q',
    'partition_method',
    'coarsen_to',
    'cut_value',
    'imbalance',
    'total_time_s',
    'coarsen_time_s',
    'init_partition_time_s',
    'refine_time_s',
]

col_w = (24, 4, 22, 10, 10, 12, 10)  # instance, q, method, coarsen_to, time, cut, imbalance

def print_header():
    header_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:<{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}}} {{:>{col_w[5]}}} {{:>{col_w[6]}}}"
    sep = ' '.join(['-' * w for w in col_w])
    print(header_fmt.format('instance', 'q', 'method', 'coarsen_to', 'time(s)', 'cut', 'imbalance'))
    print(sep)

def print_row(best_row):
    row_fmt = f"{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:<{col_w[2]}}} {{:>{col_w[3]}}} {{:>{col_w[4]}.4f}} {{:>{col_w[5]}.1f}} {{:>{col_w[6]}.4f}}"
    print(row_fmt.format(best_row['instance'], best_row['q'], best_row['partition_method'], best_row['coarsen_to'], best_row['total_time_s'], best_row['cut_value'], best_row['imbalance']))

def save_to_csv(best_rows):
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)
    print(f"Saved best results to: {csv_path}")


# FEM option



