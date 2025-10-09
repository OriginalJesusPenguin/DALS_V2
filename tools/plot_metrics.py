import os
import re
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def parse_metrics_file(fname):
    metrics = OrderedDict()
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Accept formats like: "Chamfer: 12.34" or "Chamfer 12.34" or "Chamfer=12.34"
            parts = re.split(r'[:=\t ]+', line)
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            val = parts[1].strip().rstrip(',')
            if is_number(val):
                metrics[key] = float(val)
    return metrics


def collect_all_metrics(root_dir):
    experiments = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'metrics.txt' in filenames:
            exp_name = os.path.basename(dirpath)
            fname = os.path.join(dirpath, 'metrics.txt')
            metrics = parse_metrics_file(fname)
            if len(metrics) == 0:
                continue
            experiments[exp_name] = metrics
    return experiments


def intersect_metric_keys(experiments):
    keys = None
    for m in experiments.values():
        k = set(m.keys())
        keys = k if keys is None else (keys & k)
    return [] if keys is None else sorted(keys)


def plot_metrics(experiments, out_dir, title_prefix='Metrics'):
    os.makedirs(out_dir, exist_ok=True)
    metric_keys = intersect_metric_keys(experiments)
    if len(metric_keys) == 0:
        print('No common numeric metrics found across experiments.')
        return

    exp_names = sorted(experiments.keys())
    # Single PNG table for all metrics x all experiments
    fig_h = max(4, 0.5 * len(metric_keys))
    fig_w = max(6, 0.8 * len(exp_names))
    plt.figure(figsize=(fig_w, fig_h))
    cell_text = []
    for key in metric_keys:
        row = [f'{experiments[e][key]:.4g}' for e in exp_names]
        cell_text.append(row)
    table = plt.table(cellText=cell_text,
                      rowLabels=metric_keys,
                      colLabels=exp_names,
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    plt.axis('off')
    out_all = os.path.join(out_dir, 'all_metrics.png')
    plt.savefig(out_all, dpi=220, bbox_inches='tight')
    plt.close()
    print('Saved:', out_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/ralbe/DALS/mesh_autodecoder/output',
                        help='Root directory containing experiment subfolders with metrics.txt')
    parser.add_argument('--out', type=str,
                        default='/home/ralbe/DALS/mesh_autodecoder/output/plots',
                        help='Directory to save plots')
    parser.add_argument('--title', type=str, default='Metrics',
                        help='Title prefix for figures')
    args = parser.parse_args()

    experiments = collect_all_metrics(args.root)
    if len(experiments) == 0:
        print('No metrics.txt files found under', args.root)
        return

    plot_metrics(experiments, args.out, title_prefix=args.title)


if __name__ == '__main__':
    main()


