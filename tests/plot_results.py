import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 120


def read_rows(csv_files):
    rows = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        'instance': row['instance'],
                        'q': int(row['q']),
                        'partition_method': row['partition_method'],
                        'cut_value': float(row['cut_value']),
                        'imbalance': float(row['imbalance']),
                        'total_time_s': float(row['total_time_s']),
                        'coarsen_time_s': float(row.get('coarsen_time_s', 0.0) or 0.0),
                        'init_partition_time_s': float(row.get('init_partition_time_s', 0.0) or 0.0),
                        'refine_time_s': float(row.get('refine_time_s', 0.0) or 0.0),
                    }
                )
    return rows


def deduplicate_best(rows):
    best = {}
    for row in rows:
        key = (row['instance'], row['q'], row['partition_method'])
        if key not in best:
            best[key] = row
            continue
        if row['cut_value'] < best[key]['cut_value']:
            best[key] = row
        elif row['cut_value'] == best[key]['cut_value'] and row['total_time_s'] < best[key]['total_time_s']:
            best[key] = row
    return list(best.values())


def pastel_colors(n):
    cmap = plt.get_cmap('Pastel1')
    return [cmap(i % cmap.N) for i in range(n)]


def save_cut_plot(rows_q, q, out_dir):
    instances = sorted({r['instance'] for r in rows_q})
    methods = sorted({r['partition_method'] for r in rows_q})

    cut_by_key = {(r['instance'], r['partition_method']): r['cut_value'] for r in rows_q}

    x = np.arange(len(instances), dtype=float)
    width = 0.8 / max(1, len(methods))
    colors = pastel_colors(len(methods))

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    for j, method in enumerate(methods):
        heights = [cut_by_key.get((ins, method), np.nan) for ins in instances]
        xpos = x - 0.4 + width * (j + 0.5)
        ax.bar(
            xpos,
            heights,
            width=width,
            label=method,
            color=colors[j],
            edgecolor='#666666',
            linewidth=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=20, ha='right')
    ax.set_ylabel('Cut Value')
    ax.set_xlabel('Instance')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', frameon=True)

    out_file = out_dir / f'cut_comparison_q{q}.png'
    ax.set_title(out_file.name)
    fig.tight_layout()
    fig.savefig(out_file, dpi=350, bbox_inches='tight')
    plt.close(fig)


def aggregate_runtime_by_method(rows_q):
    grouped = defaultdict(list)
    for r in rows_q:
        grouped[r['partition_method']].append(r)

    methods = sorted(grouped.keys())
    avg_coarsen = []
    avg_init = []
    avg_refine = []

    for method in methods:
        values = grouped[method]
        c = float(np.mean([v['coarsen_time_s'] for v in values]))
        i = float(np.mean([v['init_partition_time_s'] for v in values]))
        r = float(np.mean([v['refine_time_s'] for v in values]))
        t = float(np.mean([v['total_time_s'] for v in values]))

        if method == 'direct_fem' and (c + i + r) <= 1e-12:
            r = t

        avg_coarsen.append(c)
        avg_init.append(i)
        avg_refine.append(r)

    return methods, np.array(avg_coarsen), np.array(avg_init), np.array(avg_refine)


def save_runtime_plot(rows_q, q, out_dir):
    methods, coarsen, initp, refine = aggregate_runtime_by_method(rows_q)

    x = np.arange(len(methods), dtype=float)
    width = 0.65

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    stage_colors = ['#d7e8f7', '#c7dfd4', '#efe5c8']
    edge = '#666666'

    ax.bar(x, coarsen, width=width, label='coarsen', color=stage_colors[0], edgecolor=edge, linewidth=0.9)
    ax.bar(x, initp, width=width, bottom=coarsen, label='init partition', color=stage_colors[1], edgecolor=edge, linewidth=0.9)
    ax.bar(x, refine, width=width, bottom=coarsen + initp, label='refine', color=stage_colors[2], edgecolor=edge, linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Partition Method')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', frameon=True)

    out_file = out_dir / f'time_comparison_q{q}.png'
    ax.set_title(out_file.name)
    fig.tight_layout()
    fig.savefig(out_file, dpi=350, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot bmincut benchmark results from CSV.')
    parser.add_argument(
        '--input-glob',
        default='build/bmincut_results_best_*.csv',
        help='Glob pattern for input CSV files.',
    )
    parser.add_argument(
        '--out-dir',
        default='build',
        help='Output directory for generated figures.',
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(Path('.').glob(args.input_glob))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files matched pattern: {args.input_glob}')

    configure_style()

    rows = read_rows(csv_files)
    best_rows = deduplicate_best(rows)

    q_values = sorted({r['q'] for r in best_rows})
    for q in q_values:
        rows_q = [r for r in best_rows if r['q'] == q]
        if not rows_q:
            continue
        save_cut_plot(rows_q, q, out_dir)
        save_runtime_plot(rows_q, q, out_dir)

    print(f'Loaded {len(csv_files)} CSV files.')
    print(f'Generated plots for q values: {q_values}')
    print(f'Output directory: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
