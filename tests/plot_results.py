import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from test_bmincut_base import METHOD_NAME_MAP


# ── Shared bar style constants ──────────────────────────────────────────────
BAR_EDGE_COLOR = "#807E7E"
BAR_LINE_WIDTH = 1.2
BAR_WIDTH_FRAC = 0.8      # fraction of group width consumed by bars
BAR_WIDTH_MAX = 0.1       # max width per individual bar
GROUP_WIDTH = 0.8          # width of each instance group on x-axis

# ── Unified color palette (single source for both plots) ───────────────────
COLOR_CYCLE = plt.get_cmap('Pastel1').colors  # 9 pastel categorical colors

# Phase colors shared by the runtime stacked bars
STAGE_COLORS = {
    'coarsen': COLOR_CYCLE[0],  # blue
    'init':    COLOR_CYCLE[2],  # green
    'refine':  COLOR_CYCLE[3],  # red
    'single':  '#AAAAAA',
}


def configure_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Calibri']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.dpi'] = 120


def read_rows(csv_files):
    rows = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
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
                if 'device' in row:
                    entry['device'] = row['device']
                rows.append(entry)
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
    width = min(BAR_WIDTH_FRAC / n_methods, BAR_WIDTH_MAX)
    colors = [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(n_methods)]

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

        paper_name = METHOD_NAME_MAP.get(method, method)
        xpos = x - (n_methods - 1) * width / 2 + j * width
        ax.bar(
            xpos,
            ratios,
            width=width,
            label=paper_name,
            color=colors[j],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINE_WIDTH,
        )

    ax.axhline(y=0, color='#333333', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=20, ha='right')
    ax.set_ylabel(f'Cut Improvement (%)', fontweight='bold')
    ax.set_xlabel('Instance', fontweight='bold')
    ax.grid(axis='y', linewidth=1.4, alpha=0.5, linestyle='-')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=True, fontsize=9)

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

    fig, ax = plt.subplots(figsize=(12, 3))

    # Phase colors (same for all methods)
    # Distinct hatch patterns per method (cycle if more methods than patterns)
    hatch_patterns = ['', '//', '\\\\', 'xx', '++', '..', 'oo', '**']
    bar_width = min(GROUP_WIDTH / n_methods, BAR_WIDTH_MAX)

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

        xpos = np.arange(n_instances, dtype=float) - GROUP_WIDTH / 2 + bar_width * (m_idx + 0.5)
        c_any, i_any, r_any, n_phases = has_phase[method]

        if n_phases <= 1:
            total_vals = [c + i + ref for c, i, ref in zip(coarsen_vals, init_vals, refine_vals)]
            ax.bar(xpos, total_vals, width=bar_width, label=method,
                   color=STAGE_COLORS['single'], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_LINE_WIDTH,
                   hatch=hatch)
        else:
            ax.bar(xpos, coarsen_vals, width=bar_width, color=STAGE_COLORS['coarsen'], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_LINE_WIDTH,
                   hatch=hatch)
            ax.bar(xpos, init_vals, width=bar_width, bottom=coarsen_vals,
                   color=STAGE_COLORS['init'], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_LINE_WIDTH,
                   hatch=hatch)
            ax.bar(xpos, refine_vals, width=bar_width,
                   bottom=[c + i for c, i in zip(coarsen_vals, init_vals)],
                   color=STAGE_COLORS['refine'], edgecolor=BAR_EDGE_COLOR, linewidth=BAR_LINE_WIDTH,
                   hatch=hatch)

    ax.set_xticks(np.arange(n_instances, dtype=float))
    ax.set_xticklabels(instances, rotation=20, ha='right')
    ax.set_ylabel('Time (s)', fontweight='bold')
    ax.set_xlabel('Instance', fontweight='bold')
    ax.grid(axis='y', linewidth=1.4, alpha=0.5, linestyle='-')
    ax.set_axisbelow(True)
    
    # Legend: one entry per method, showing its hatch + representative color
    from matplotlib.patches import Patch
    method_legend = []
    for m_idx, method in enumerate(methods):
        paper_name = METHOD_NAME_MAP.get(method, method)
        hatch = hatch_patterns[m_idx % len(hatch_patterns)]
        c_any, i_any, r_any, n_phases = has_phase[method]
        # Use the dominant stage color for the legend patch
        base_color = STAGE_COLORS['single'] if n_phases <= 1 else STAGE_COLORS['init']
        method_legend.append(
            Patch(facecolor=base_color, edgecolor=BAR_EDGE_COLOR, hatch=hatch, label=paper_name)
        )
    # Add phase color explanation
    phase_legend = [
        Patch(facecolor=STAGE_COLORS['coarsen'], edgecolor=BAR_EDGE_COLOR, label='coarsen'),
        Patch(facecolor=STAGE_COLORS['init'], edgecolor=BAR_EDGE_COLOR, label='init'),
        Patch(facecolor=STAGE_COLORS['refine'], edgecolor=BAR_EDGE_COLOR, label='refine'),
    ]
    if any(n == 1 for _, _, _, n in has_phase.values()):
        phase_legend.append(
            Patch(facecolor=STAGE_COLORS['single'], edgecolor=BAR_EDGE_COLOR, label='single phase')
        )
    legend1 = ax.legend(handles=method_legend, loc='upper left', frameon=True,
                        fontsize=9, title='Method')
    ax.add_artist(legend1)
    ax.legend(handles=phase_legend, loc='upper right', frameon=True, fontsize=8,
              title='Phase')

    out_file = out_dir / f'time_comparison_q{q}.png'
    fig.tight_layout()
    fig.savefig(out_file, dpi=350, bbox_inches='tight')
    plt.close(fig)


def save_gpu_boost_plot(rows, out_dir):
    """Plot 3: GPU speedup (cpu_total_time / gpu_total_time) per method.

    Line plot with one figure per method; different q values as separate lines.
    """
    rows_with_device = [r for r in rows if 'device' in r]
    if not rows_with_device:
        return

    methods = sorted({r['partition_method'] for r in rows_with_device})
    q_values = sorted({r['q'] for r in rows_with_device})
    instances = sorted({r['instance'] for r in rows_with_device})

    for method in methods:
        fig, ax = plt.subplots(figsize=(8, 3))
        ls_cycle = ['-', '--', '-.', ':']
        for qi, q in enumerate(q_values):
            speedups = []
            for ins in instances:
                cpu_row = next(
                    (r for r in rows_with_device
                     if r['instance'] == ins and r['q'] == q
                     and r['partition_method'] == method and r['device'] == 'cpu'),
                    None
                )
                gpu_row = next(
                    (r for r in rows_with_device
                     if r['instance'] == ins and r['q'] == q
                     and r['partition_method'] == method and r['device'] == 'cuda'),
                    None
                )
                if cpu_row is not None and gpu_row is not None and gpu_row['total_time_s'] > 0:
                    speedups.append(cpu_row['total_time_s'] / gpu_row['total_time_s'])
                else:
                    speedups.append(np.nan)
            paper_name = METHOD_NAME_MAP.get(method, method)
            ax.plot(instances, speedups, marker='o', linestyle=ls_cycle[qi % len(ls_cycle)],
                    label=f'q={q}', linewidth=1.8, markersize=5)

        ax.axhline(y=1, color='#999999', linewidth=0.8, linestyle='--')
        ax.set_ylabel('Speedup (CPU / GPU)', fontweight='bold')
        ax.set_xlabel('Instance', fontweight='bold')
        # ax.set_title(f'{paper_name} — GPU Speedup', fontweight='bold')
        ax.grid(axis='y', linewidth=1.4, alpha=0.5, linestyle='-')
        ax.set_axisbelow(True)
        ax.legend(loc='best', frameon=True, fontsize=10)

        out_file = out_dir / f'gpu_boost_{method}.png'
        fig.tight_layout()
        fig.savefig(out_file, dpi=350, bbox_inches='tight')
        plt.close(fig)


def save_gpu_ratio_plot(rows, out_dir):
    """Plot 4: GPU init fraction (gpu_init_time / gpu_total_time) per method.

    Line plot with one figure per method; different q values as separate lines.
    Shows how much of the runtime is spent on the GPU-accelerated phase.
    """
    rows_gpu = [r for r in rows if 'device' in r and r['device'] == 'cuda']
    if not rows_gpu:
        print('No GPU rows found for Plot 4.')
        return

    methods = sorted({r['partition_method'] for r in rows_gpu})
    q_values = sorted({r['q'] for r in rows_gpu})
    instances = sorted({r['instance'] for r in rows_gpu})

    for method in methods:
        fig, ax = plt.subplots(figsize=(8, 3))
        ls_cycle = ['-', '--', '-.', ':']
        for qi, q in enumerate(q_values):
            ratios = []
            for ins in instances:
                row = next(
                    (r for r in rows_gpu
                     if r['instance'] == ins and r['q'] == q
                     and r['partition_method'] == method),
                    None
                )
                if row is not None and row['total_time_s'] > 0:
                    ratios.append(row['init_partition_time_s'] / row['total_time_s'])
                else:
                    ratios.append(np.nan)
            paper_name = METHOD_NAME_MAP.get(method, method)
            ax.plot(instances, ratios, marker='s', linestyle=ls_cycle[qi % len(ls_cycle)],
                    label=f'q={q}', linewidth=1.8, markersize=5)

        ax.set_ylabel('GPU Init Fraction', fontweight='bold')
        ax.set_xlabel('Instance', fontweight='bold')
        # ax.set_title(f'{paper_name} — GPU Runtime Ratio', fontweight='bold')
        ax.grid(axis='y', linewidth=1.4, alpha=0.5, linestyle='-')
        ax.set_axisbelow(True)
        ax.legend(loc='best', frameon=True, fontsize=10)

        out_file = out_dir / f'gpu_ratio_{method}.png'
        fig.tight_layout()
        fig.savefig(out_file, dpi=350, bbox_inches='tight')
        plt.close(fig)


# ── Plot 5: Coarsening sensitivity (merged from plot_coarsening_sensitivity.py) ─

def save_coarsening_sensitivity_plot(data_q, out_dir):
    """Plot 5: Coarsening target sensitivity — cut value + runtime vs coarsen_to."""
    for (ins, q), vals in data_q.items():
        idx = np.argsort(vals['coarsen_to'])
        x = np.array(vals['coarsen_to'])[idx]
        y_cut = np.array(vals['cut_value'])[idx]
        y_time = np.array(vals['total_time_s'])[idx]

        fig, ax1 = plt.subplots(figsize=(6, 4))

        color_cut = '#4c72b0'
        color_time = '#dd8452'

        ax1.set_xlabel('Coarsen Target Nodes')
        ax1.set_ylabel('Cut Value', color=color_cut)
        line1, = ax1.plot(x, y_cut, marker='o', color=color_cut, label='Cut Value',
                          linewidth=1.5, markersize=5)
        ax1.tick_params(axis='y', labelcolor=color_cut)
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Runtime (s)', color=color_time)
        line2, = ax2.plot(x, y_time, marker='s', color=color_time, label='Runtime',
                          linewidth=1.5, markersize=5)
        ax2.tick_params(axis='y', labelcolor=color_time)

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', frameon=True)

        out_name = f'sensitivity_{ins}_q{q}.png'
        ax1.set_title(out_name)

        fig.tight_layout()
        plt.savefig(out_dir / out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_dir / out_name}")


def read_sensitivity_data(csv_path):
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['instance'], row['q'])
            data[key]['coarsen_to'].append(int(row['coarsen_to']))
            data[key]['cut_value'].append(float(row['cut_value']))
            data[key]['total_time_s'].append(float(row['total_time_s']))
    return data


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
    parser.add_argument(
        '--plot-number',
        type=int,
        nargs='+',
        default=[1, 2],
        help='Plot types to generate: 1=cut_improvement, 2=time_comparison, '
             '3=gpu_boost, 4=gpu_ratio, 5=coarsening_sensitivity. Default: 1 2',
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(Path('.').glob(args.input_glob))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files matched pattern: {args.input_glob}')

    configure_style()

    rows = read_rows(csv_files)

    plot_numbers = set(args.plot_number)

    # Plots 1 & 2 use deduplicated best rows (no device distinction)
    if 1 in plot_numbers or 2 in plot_numbers:
        best_rows = deduplicate_best(rows)
        q_values = sorted({r['q'] for r in best_rows})
        for q in q_values:
            rows_q = [r for r in best_rows if r['q'] == q]
            if not rows_q:
                continue
            if 1 in plot_numbers:
                save_cut_plot(rows_q, q, out_dir)
            if 2 in plot_numbers:
                save_runtime_plot(rows_q, q, out_dir)
        print(f'Plots 1/2 — q values: {q_values}')

    # Plots 3 & 4 use detailed rows with device labels
    if 3 in plot_numbers:
        save_gpu_boost_plot(rows, out_dir)
        print('Plot 3 — GPU boost charts generated.')
    if 4 in plot_numbers:
        save_gpu_ratio_plot(rows, out_dir)

    # Plot 5: coarsening sensitivity — reads separate CSV files
    if 5 in plot_numbers:
        sensitivity_csvs = sorted(Path('.').glob('build/bmincut_cfrk_sensitivity_*.csv'))
        if not sensitivity_csvs:
            print('Plot 5 — no sensitivity CSV files found (pattern: build/bmincut_cfrk_sensitivity_*.csv)')
        else:
            all_sens_data = {}
            for csv_file in sensitivity_csvs:
                data = read_sensitivity_data(csv_file)
                all_sens_data.update(data)
            save_coarsening_sensitivity_plot(all_sens_data, out_dir)
            print(f'Plot 5 — sensitivity charts generated from {len(sensitivity_csvs)} CSV file(s).')

    print(f'Loaded {len(csv_files)} CSV files.')
    print(f'Output directory: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
