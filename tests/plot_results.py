import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def configure_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
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


def _detect_baseline_method(rows_q):
    """Pick the baseline method for computing improvement ratios.
    
    Prefers 'kahip' over 'kaffpa' over any available method.
    """
    methods = {r['partition_method'] for r in rows_q}
    for preferred in ('kahip', 'kaffpa'):
        if preferred in methods:
            return preferred
    return sorted(methods)[0]


def save_cut_plot(rows_q, q, out_dir, baseline_method=None):
    """Output improvement ratio compared to a baseline method instead of raw cut value.
    
    For each instance, computes: (baseline_cut - method_cut) / |baseline_cut| * 100.
    Positive values mean the method beats the baseline (lower cut = better).
    The baseline method itself is not plotted (always 0%).
    """
    if baseline_method is None:
        baseline_method = _detect_baseline_method(rows_q)

    instances = sorted({r['instance'] for r in rows_q})
    methods = sorted({r['partition_method'] for r in rows_q})

    cut_by_key = {(r['instance'], r['partition_method']): r['cut_value'] for r in rows_q}

    # Only plot non-baseline methods
    plot_methods = [m for m in methods if m != baseline_method]
    if not plot_methods:
        return

    x = np.arange(len(instances), dtype=float)
    n_methods = len(plot_methods)
    width = min(0.8 / n_methods, 0.25)
    colors = pastel_colors(n_methods)

    fig, ax = plt.subplots(figsize=(12, 3))

    for j, method in enumerate(plot_methods):
        ratios = []
        for ins in instances:
            base_cut = cut_by_key.get((ins, baseline_method), None)
            method_cut = cut_by_key.get((ins, method), None)
            if base_cut is not None and method_cut is not None and abs(base_cut) > 1e-12:
                # Positive = method improves over baseline
                ratio = (base_cut - method_cut) / abs(base_cut) * 100.0
            else:
                ratio = np.nan
            ratios.append(ratio)

        xpos = x - (n_methods - 1) * width / 2 + j * width
        ax.bar(
            xpos,
            ratios,
            width=width,
            label=method,
            color=colors[j],
            edgecolor='#666666',
            linewidth=0.9,
        )

    ax.axhline(y=0, color='#333333', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=20, ha='right')
    ax.set_ylabel(f'Cut Improvement vs {baseline_method} (%)')
    ax.set_xlabel('Instance')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(loc='best', frameon=True)

    out_file = out_dir / f'cut_improvement_q{q}.png'
    fig.tight_layout()
    fig.savefig(out_file, dpi=350, bbox_inches='tight')
    plt.close(fig)


def save_runtime_plot(rows_q, q, out_dir):
    """Runtime plot showing all instances × all methods in one figure.
    
    Produces grouped stacked bars: x-axis = instances, each group of bars = methods,
    each bar is stacked with coarsen/init/refine time components.
    Different hatch patterns identify each method; legend shows method ↔ pattern.
    """
    instances = sorted({r['instance'] for r in rows_q})
    methods = sorted({r['partition_method'] for r in rows_q})

    n_instances = len(instances)
    n_methods = len(methods)

    # Collect runtime breakdown per (instance, method)
    time_by_key: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for r in rows_q:
        key = (r['instance'], r['partition_method'])
        c = r['coarsen_time_s']
        i = r['init_partition_time_s']
        ref = r['refine_time_s']
        if r['partition_method'] == 'direct_fem' and (c + i + ref) <= 1e-12:
            ref = r['total_time_s']
        time_by_key[key] = (c, i, ref)

    fig, ax = plt.subplots(figsize=(max(12, n_instances * n_methods * 0.5), 4))

    # Phase colors (same for all methods)
    stage_colors = {'coarsen': '#4C72B0', 'init': '#55A868', 'refine': '#DD8452',
                     'single': '#AAAAAA'}
    # Distinct hatch patterns per method (cycle if more methods than patterns)
    hatch_patterns = ['', '//', '\\\\', 'xx', '++', '..', 'oo', '**']
    edge = '#666666'
    group_width = 0.8
    bar_width = group_width / n_methods

    # Check which stages each method actually uses
    has_phase = {}
    for method in methods:
        c_any = any(time_by_key.get((ins, method), (0, 0, 0))[0] > 1e-12 for ins in instances)
        i_any = any(time_by_key.get((ins, method), (0, 0, 0))[1] > 1e-12 for ins in instances)
        r_any = any(time_by_key.get((ins, method), (0, 0, 0))[2] > 1e-12 for ins in instances)
        n_phases = sum([c_any, i_any, r_any])
        has_phase[method] = (c_any, i_any, r_any, n_phases)

    for m_idx, method in enumerate(methods):
        hatch = hatch_patterns[m_idx % len(hatch_patterns)]
        coarsen_vals = []
        init_vals = []
        refine_vals = []
        for ins in instances:
            c, i, ref = time_by_key.get((ins, method), (0.0, 0.0, 0.0))
            coarsen_vals.append(c)
            init_vals.append(i)
            refine_vals.append(ref)

        xpos = np.arange(n_instances, dtype=float) - group_width / 2 + bar_width * (m_idx + 0.5)
        c_any, i_any, r_any, n_phases = has_phase[method]

        if n_phases <= 1:
            total_vals = [c + i + ref for c, i, ref in zip(coarsen_vals, init_vals, refine_vals)]
            ax.bar(xpos, total_vals, width=bar_width, label=method,
                   color=stage_colors['single'], edgecolor=edge, linewidth=0.9,
                   hatch=hatch)
        else:
            ax.bar(xpos, coarsen_vals, width=bar_width,
                   color=stage_colors['coarsen'], edgecolor=edge, linewidth=0.9,
                   hatch=hatch)
            ax.bar(xpos, init_vals, width=bar_width, bottom=coarsen_vals,
                   color=stage_colors['init'], edgecolor=edge, linewidth=0.9,
                   hatch=hatch)
            ax.bar(xpos, refine_vals, width=bar_width,
                   bottom=[c + i for c, i in zip(coarsen_vals, init_vals)],
                   color=stage_colors['refine'], edgecolor=edge, linewidth=0.9,
                   hatch=hatch)

    ax.set_xticks(np.arange(n_instances, dtype=float))
    ax.set_xticklabels(instances, rotation=20, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Instance')
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # Legend: one entry per method, showing its hatch + representative color
    from matplotlib.patches import Patch
    method_legend = []
    for m_idx, method in enumerate(methods):
        hatch = hatch_patterns[m_idx % len(hatch_patterns)]
        c_any, i_any, r_any, n_phases = has_phase[method]
        # Use the dominant stage color for the legend patch
        base_color = stage_colors['single'] if n_phases <= 1 else stage_colors['init']
        method_legend.append(
            Patch(facecolor=base_color, edgecolor=edge, hatch=hatch, label=method)
        )
    # Add phase color explanation
    phase_legend = [
        Patch(facecolor=stage_colors['coarsen'], edgecolor=edge, label='coarsen'),
        Patch(facecolor=stage_colors['init'], edgecolor=edge, label='init'),
        Patch(facecolor=stage_colors['refine'], edgecolor=edge, label='refine'),
    ]
    if any(n == 1 for _, _, _, n in has_phase.values()):
        phase_legend.append(
            Patch(facecolor=stage_colors['single'], edgecolor=edge, label='single phase')
        )
    legend1 = ax.legend(handles=method_legend, loc='upper left', frameon=True,
                        fontsize=8, title='Method')
    ax.add_artist(legend1)
    ax.legend(handles=phase_legend, loc='upper right', frameon=True, fontsize=8,
              title='Phase')

    out_file = out_dir / f'time_comparison_q{q}.png'
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
