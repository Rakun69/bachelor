#!/usr/bin/env python3
"""
Plot real_crossover_table_100.csv as an overview figure (times, sizes, ratio).

Inputs:
  results_summary/real_crossover_table_100.csv

Outputs:
  data/visualizations/real_crossover_overview_100.png (combined)
  data/visualizations/real_times_100.png            (separate)
  data/visualizations/real_sizes_100.png            (separate)
  data/visualizations/real_ratio_100.png            (separate)

Usage:
  PYTHONPATH=. python scripts/plot_crossover_table_100.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    x = df['IoT Readings']
    std_total = df['Standard Total (s)']
    nova_total = df['Nova Total (s)']
    std_size = df['Standard Size (KB)']
    nova_size = df['Nova Size (KB)']

    # Time ratio: Standard / Nova (>1 => Nova besser)
    ratio = std_total / nova_total.replace(0, pd.NA)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('REAL crossover overview (from CSV: real_crossover_table_100.csv)')

    # Times plot
    ax = axes[0]
    ax.plot(x, std_total, 'r-', label='Standard total (s)')
    ax.plot(x, nova_total, 'b-', label='Nova total (s)')
    ax.set_xlabel('IoT Readings')
    ax.set_ylabel('Total time (s)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Sizes plot
    ax = axes[1]
    ax.plot(x, std_size, 'r-', label='Standard size (KB)')
    ax.plot(x, nova_size, 'b-', label='Nova size (KB)')
    ax.set_xlabel('IoT Readings')
    ax.set_ylabel('Total proof size (KB)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Ratio plot
    ax = axes[2]
    ax.plot(x, ratio, 'g-', label='Standard / Nova (time)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('IoT Readings')
    ax.set_ylabel('Time ratio (>1 Nova better)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved: {out_path}")


def _compute_crossover_index(x: pd.Series, ratio: pd.Series) -> int | None:
    for i, (n, r) in enumerate(zip(x, ratio)):
        try:
            if float(r) > 1.0:
                return int(n)
        except Exception:
            continue
    return None


def plot_separate(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    x = df['IoT Readings']
    std_total = df['Standard Total (s)']
    nova_total = df['Nova Total (s)']
    std_size = df['Standard Size (KB)']
    nova_size = df['Nova Size (KB)']
    ratio = std_total / nova_total.replace(0, pd.NA)

    out_dir.mkdir(parents=True, exist_ok=True)
    cross = _compute_crossover_index(x, ratio)

    # Times
    fig = plt.figure(figsize=(16, 6))
    plt.plot(x, std_total, 'r-', linewidth=2, label='Standard total (s)')
    plt.plot(x, nova_total, 'b-', linewidth=2, label='Nova total (s)')
    if cross is not None:
        plt.axvline(cross, color='green', linestyle='--', alpha=0.6, label=f'Crossover ~ {cross}')
    plt.xlabel('IoT Readings')
    plt.ylabel('Total time (s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    times_path = out_dir / f'real_times_{tag}.png'
    fig.tight_layout()
    fig.savefig(times_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Sizes
    fig = plt.figure(figsize=(16, 6))
    plt.plot(x, std_size, 'r-', linewidth=2, label='Standard size (KB)')
    plt.plot(x, nova_size, 'b-', linewidth=2, label='Nova size (KB)')
    plt.xlabel('IoT Readings')
    plt.ylabel('Total proof size (KB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    sizes_path = out_dir / f'real_sizes_{tag}.png'
    fig.tight_layout()
    fig.savefig(sizes_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Ratio
    fig = plt.figure(figsize=(16, 6))
    plt.plot(x, ratio, 'g-', linewidth=2, label='Standard / Nova (time)')
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    if cross is not None:
        plt.axvline(cross, color='green', linestyle='--', alpha=0.6)
    plt.xlabel('IoT Readings')
    plt.ylabel('Time ratio (>1 Nova better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    ratio_path = out_dir / f'real_ratio_{tag}.png'
    fig.tight_layout()
    fig.savefig(ratio_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved: {times_path}")
    print(f"✅ Saved: {sizes_path}")
    print(f"✅ Saved: {ratio_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot real_crossover_table_100.csv')
    p.add_argument('--csv', type=str, default='results_summary/real_crossover_table_100.csv')
    p.add_argument('--out', type=str, default='data/visualizations/real_crossover_overview_100.png')
    p.add_argument('--separate', action='store_true', help='Also write three separate, larger plots')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_table(csv_path)
    plot_overview(df, Path(args.out))

    if args.separate:
        # derive tag from filename suffix, e.g., ..._100.csv -> 100
        m = re.search(r'_(\d+)\.csv$', csv_path.name)
        tag = m.group(1) if m else 'table'
        plot_separate(df, Path(args.out).parent, tag)


if __name__ == '__main__':
    main()


