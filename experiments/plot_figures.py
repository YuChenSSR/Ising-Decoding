#!/usr/bin/env python3
"""Generate IEEE CAL paper figures from parsed experiment results."""
import os
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# IEEE style
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.figsize': (3.5, 2.5),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 0.8,
})


def load_csv(name):
    path = os.path.join(PARSED_DIR, name)
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def safe_float(val, default=float('nan')):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def extract_distance(filename):
    m = re.search(r'_d(\d+)', filename)
    return int(m.group(1)) if m else 0


def fig1_backend_comparison():
    """Figure 1: Deployment backend comparison (bar chart)."""
    data = load_csv("exp1_results.csv")
    if not data:
        return

    labels = []
    ler_vals = []
    latency_vals = []
    speedup_vals = []

    label_map = {
        'pytorch_nocompile': 'PyTorch',
        'pytorch_compile': 'PyTorch\n+compile',
        'trt_fp16': 'TRT\nFP16',
        'trt_int8': 'TRT\nINT8',
    }

    for row in data:
        fn = row['file']
        for key, label in label_map.items():
            if key in fn:
                labels.append(label)
                ler_vals.append(safe_float(row.get('ler_predecoder')))
                latency_vals.append(safe_float(row.get('pm_latency_predecoder_us')))
                speedup_vals.append(safe_float(row.get('pm_speedup')))
                break

    if not labels:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.25))
    x = np.arange(len(labels))
    w = 0.5

    # Left: LER comparison
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    ax1.bar(x, ler_vals, w, color=colors[:len(labels)], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_title('(a) LER by Backend', fontsize=8)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))

    # Right: PyMatching speedup
    ax2.bar(x, speedup_vals, w, color=colors[:len(labels)], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('PyMatching Speedup (×)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_title('(b) PyMatching Speedup', fontsize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_backend_comparison.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig1_backend_comparison.png"), bbox_inches='tight')
    print("Saved: fig1_backend_comparison")
    plt.close()


def fig2_quantization_tradeoff():
    """Figure 2: Quantization accuracy-performance Pareto (scatter)."""
    data = load_csv("exp2_results.csv")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(3.15, 2.45))

    markers = {'model1': 'o', 'model4': 's'}
    colors_map = {'fp32': '#2196F3', 'trt_fp16': '#4CAF50', 'trt_int8': '#F44336'}
    label_map = {'fp32': 'FP32', 'trt_fp16': 'TRT-FP16', 'trt_int8': 'TRT-INT8'}

    for row in data:
        fn = row['file']
        mid = 'model1' if 'model1' in fn else 'model4'
        for qkey in ['trt_int8', 'trt_fp16', 'fp32']:
            if qkey in fn:
                quant = qkey
                break

        ler = safe_float(row.get('ler_predecoder'))
        latency = safe_float(row.get('pm_latency_predecoder_us'))
        marker = markers[mid]
        color = colors_map[quant]
        label = f"{mid.upper()} {label_map[quant]}"
        ax.scatter(latency, ler, marker=marker, c=color, s=48,
                   edgecolors='black', linewidth=0.5, label=label, zorder=3)

    ax.set_xlabel('PyMatching Latency (µs/round)')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('Quantization: Accuracy vs Latency', fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=5.5, loc='upper right',
              handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_quantization_tradeoff.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig2_quantization_tradeoff.png"), bbox_inches='tight')
    print("Saved: fig2_quantization_tradeoff")
    plt.close()


def fig3_distance_scaling():
    """Figure 3: Latency vs code distance scaling."""
    data = load_csv("exp3_results.csv")
    if not data:
        return

    distances = []
    baseline_lat = []
    predecoder_lat = []

    for row in sorted(data, key=lambda r: extract_distance(r['file'])):
        d = extract_distance(row['file'])
        bl = safe_float(row.get('pm_latency_baseline_us'))
        pl = safe_float(row.get('pm_latency_predecoder_us'))
        if d > 0 and not np.isnan(bl):
            distances.append(d)
            baseline_lat.append(bl)
            predecoder_lat.append(pl)

    if not distances:
        return

    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    ax.semilogy(distances, baseline_lat, 'o-', color='#F44336', label='Baseline (PyMatching only)', markersize=4)
    ax.semilogy(distances, predecoder_lat, 's-', color='#4CAF50', label='After Pre-decoder', markersize=4)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0, label='1 µs/round budget')
    ax.set_xlabel('Code Distance (d)')
    ax.set_ylabel('PyMatching Latency (µs/round)')
    ax.set_title('Decoding Latency vs Code Distance', fontsize=8)
    ax.set_xticks(distances)
    ax.legend(fontsize=5.5, handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_distance_scaling.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig3_distance_scaling.png"), bbox_inches='tight')
    print("Saved: fig3_distance_scaling")
    plt.close()


def fig4_realtime_feasibility():
    """Figure 4: Total decoding pipeline latency vs real-time budget."""
    data3 = load_csv("exp3_results.csv")
    data4 = load_csv("exp4_results.csv")
    all_data = data3 + data4

    if not all_data:
        return

    # Group by distance, collect different backends
    from collections import defaultdict
    by_d = defaultdict(dict)
    for row in all_data:
        fn = row['file']
        d = extract_distance(fn)
        bl = safe_float(row.get('pm_latency_baseline_us'))
        pl = safe_float(row.get('pm_latency_predecoder_us'))
        fwd = safe_float(row.get('model_forward_s'))
        total_rounds = safe_float(row.get('total_rounds', '0'))

        if d > 0 and not np.isnan(pl):
            if 'int8' in fn:
                by_d[d]['int8_pm'] = pl
                if total_rounds > 0 and not np.isnan(fwd):
                    by_d[d]['int8_nn_us'] = fwd / total_rounds * 1e6
            elif 'fp16' in fn or 'exp3' in fn:
                by_d[d]['fp16_pm'] = pl
                by_d[d]['baseline'] = bl
                if total_rounds > 0 and not np.isnan(fwd):
                    by_d[d]['fp16_nn_us'] = fwd / total_rounds * 1e6

    distances = sorted(by_d.keys())
    if not distances:
        return

    fig, ax = plt.subplots(figsize=(3.15, 2.45))

    # Baseline
    bl_vals = [by_d[d].get('baseline', np.nan) for d in distances]
    ax.semilogy(distances, bl_vals, 'x--', color='#9E9E9E', label='PyMatching only', markersize=5)

    # FP16: NN + PM
    fp16_total = []
    for d in distances:
        nn = by_d[d].get('fp16_nn_us', 0)
        pm = by_d[d].get('fp16_pm', np.nan)
        fp16_total.append(nn + pm)
    ax.semilogy(distances, fp16_total, 'o-', color='#4CAF50', label='TRT-FP16 (NN+PM)', markersize=4)

    # INT8: NN + PM
    int8_total = []
    int8_d = []
    for d in distances:
        if 'int8_pm' in by_d[d]:
            nn = by_d[d].get('int8_nn_us', 0)
            pm = by_d[d]['int8_pm']
            int8_total.append(nn + pm)
            int8_d.append(d)
    if int8_d:
        ax.semilogy(int8_d, int8_total, 's-', color='#F44336', label='TRT-INT8 (NN+PM)', markersize=4)

    ax.axhline(y=1.0, color='#FF9800', linestyle='--', linewidth=1.5, label='1 µs/round budget')
    ax.set_xlabel('Code Distance (d)')
    ax.set_ylabel('Total Decoding Latency (µs/round)')
    ax.set_title('Real-Time Feasibility Analysis', fontsize=8)
    ax.set_xticks(distances)
    ax.legend(fontsize=5.5, loc='upper left', handletextpad=0.3,
              borderpad=0.25, labelspacing=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_realtime_feasibility.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig4_realtime_feasibility.png"), bbox_inches='tight')
    print("Saved: fig4_realtime_feasibility")
    plt.close()


if __name__ == "__main__":
    fig1_backend_comparison()
    fig2_quantization_tradeoff()
    fig3_distance_scaling()
    fig4_realtime_feasibility()
    print(f"\nAll figures saved to: {FIG_DIR}/")
