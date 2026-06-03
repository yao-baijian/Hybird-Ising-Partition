import sys
sys.path.append('.')
import subprocess
import pandas as pd
import re
import os
import csv

num_trials = 200
num_steps = 1000
dev = 'cuda' # if you do not have gpu in your computing devices, then choose 'cpu' here

case_type = 'hyperbmincut'
map_types = ['normal', 'star', 'clique', 'weighted_clique', 'bisecgraph']
# epsilons = np.linspace(0.01, 0.05, 5)
epsilons = [0.02]
instance_root_dir = '../partition/data/hypergraph_set/'
grad_options = [True, False]
instance_list = ['bibd_49_3.mtx.hgr',
                 'Pd_rhs.mtx.hgr',
                 'dac2012_superblue19.hgr',
                 'ISPD98_ibm07.hgr',
                 'G2_circuit.mtx.hgr']
q_values = [2, 4, 8, 16, 32, 64]

results = []
total_experiments = len(instance_list) * len(map_types) * len(epsilons) * len(grad_options) * len(q_values)
current_experiment = 0

csv_filename = 'kahypar_results.csv'

hypergraph_stats = {}
if os.path.exists('../partition/data/hypergraph_statistics.csv'):
    stats_df = pd.read_csv('../partition/data/hypergraph_statistics.csv')
    for _, row in stats_df.iterrows():
        hypergraph_stats[row['graph']] = {
            'avgHEsize': row['avgHEsize'],
            'HNs': row['HNs'],
            'HEs': row['HEs'],
            'density': row['density']
        }

processed_instances = set()

kahyper_on = True
if kahyper_on:

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Avg_HE_Size', 'Q', 'Epsilon', 'Hyperedge_Cut', 'Total_Time_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 打印表头（只需要一次）
        print(f"{'Instance':<20} {'AvgHE':<8} {'Q':<4} {'Epsilon':<8} {'Hyperedge Cut':<15} {'Total Time (s)':<15}")
        print("-" * 75)

        for instance in instance_list:
            if not os.path.exists(instance_root_dir + instance): 
                continue
            for q in q_values:
                for epsilon in epsilons:
                    kahypar_cmd = [
                        '../kahypar/build/kahypar/application/KaHyPar',
                        '-h', instance_root_dir + instance,
                        '-k', str(q),
                        '-e', str(epsilon),
                        '-o', 'km1',
                        '-m', 'direct',
                        '-p', '../kahypar/config/km1_kKaHyPar_sea20.ini'
                    ]

                    result = subprocess.run(kahypar_cmd, capture_output=True, text=True)
                    # print("KaHyPar output:")
                    # print(result.stdout)
                    # if result.stderr:
                    #     print("KaHyPar errors:")
                    #     print(result.stderr)
                        # 解析输出，提取Hyperedge Cut值和总时间
                    hyperedge_cut = "N/A"
                    total_time = "N/A"
                    
                    for line in result.stdout.split('\n'):
                        if 'Hyperedge Cut' in line:
                            # 提取数字部分
                            parts = line.split('=')
                            if len(parts) > 1:
                                hyperedge_cut = parts[1].strip()
                        elif 'Partition time' in line:
                            # 提取总时间，格式: "Partition time = 10.4691 s"
                            time_match = re.search(r'Partition time\s*=\s*([\d.]+)\s*s', line)
                            if time_match:
                                total_time = time_match.group(1)
                    
                    stats_info = hypergraph_stats[instance.replace('.hgr', '')]
                    
                    # 判断是否是第一次运行该实例
                    is_first_run = instance not in processed_instances

                    avg_he_size = ''
                    if is_first_run:
                        avg_he_size = stats_info['avgHEsize']
                        processed_instances.add(instance)

                    # 表格化输出到控制台
                    instance_short = instance[:18] + ".." if len(instance) > 18 else instance
                    print(f"{instance_short:<20} {avg_he_size:<8} {q:<4} {epsilon:<8.4f} {hyperedge_cut:<15} {total_time:<15}")
                    
                    # 写入CSV文件
                    writer.writerow({
                        'Instance': instance,
                        'Avg_HE_Size': avg_he_size,
                        'Q': q,
                        'Epsilon': epsilon,
                        'Hyperedge_Cut': hyperedge_cut,
                        'Total_Time_s': total_time
                    })
                    
                    if result.stderr:
                        print(f"Error for {instance}, q={q}, epsilon={epsilon}:")
                        print(result.stderr)

    print("-" * 65)
    print(f"Results saved to {csv_filename}")
    print("KaHyPar experiments completed.")