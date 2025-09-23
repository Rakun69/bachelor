#!/usr/bin/env python3
"""
Plot REAL crossover results from JSON into an overview visualization.

Outputs (default):
- data/visualizations/real_crossover_overview.png

Usage:
  PYTHONPATH=. python scripts/plot_crossover_json.py \
    --json data/real_measurements/crossover_results.json \
    --out-dir data/visualizations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    results = sorted(results, key=lambda r: r.get("iot_readings", 0))
    return results


def extract_series(results: List[Dict[str, Any]]):
    readings: List[int] = []
    std_total: List[float] = []
    nova_total: List[float] = []
    std_size_kb: List[float] = []
    nova_size_kb: List[float] = []
    ratio: List[float] = []

    for row in results:
        n = int(row.get("iot_readings", 0))
        std = row.get("standard", {})
        nova = row.get("nova", {})

        st = float(std.get("total_time_s", 0.0))
        nt = float(nova.get("total_time_s", 0.0))
        ssz = float(std.get("total_proof_size_bytes", 0) / 1024.0)
        nsz = float(nova.get("proof_size_bytes", 0) / 1024.0)

        readings.append(n)
        std_total.append(st)
        nova_total.append(nt)
        std_size_kb.append(ssz)
        nova_size_kb.append(nsz)
        ratio.append(st / nt if nt > 0 else 0.0)

    return readings, std_total, nova_total, std_size_kb, nova_size_kb, ratio


def compute_crossovers(readings, std_total, nova_total, std_size_kb, nova_size_kb) -> Tuple[int | None, int | None]:
    time_cross = None
    for n, st, nt in zip(readings, std_total, nova_total):
        if nt > 0 and st / nt > 1.0:
            time_cross = n
            break

    size_cross = None
    for n, ssz, nsz in zip(readings, std_size_kb, nova_size_kb):
        if ssz > nsz:
            size_cross = n
            break

    return time_cross, size_cross


def plot_overview(json_path: Path, out_dir: Path) -> Path:
    results = load_results(json_path)
    readings, std_total, nova_total, std_size_kb, nova_size_kb, ratio = extract_series(results)
    time_cross, size_cross = compute_crossovers(readings, std_total, nova_total, std_size_kb, nova_size_kb)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Times
    ax = axes[0]
    ax.plot(readings, std_total, 'r-o', label='Standard total (s)')
    ax.plot(readings, nova_total, 'b-o', label='Nova total (s)')
    if time_cross is not None:
        ax.axvline(time_cross, color='green', linestyle='--', alpha=0.7)
        ax.text(time_cross, max(max(std_total), max(nova_total)) * 0.8,
                f'Time crossover ~ {time_cross}', color='green')
    ax.set_xlabel('IoT Readings')
    ax.set_ylabel('Total time (s)')
    ax.set_title('Total time vs readings')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Sizes
    ax = axes[1]
    ax.plot(readings, std_size_kb, 'r-o', label='Standard size (KB)')
    ax.plot(readings, nova_size_kb, 'b-o', label='Nova size (KB)')
    if size_cross is not None:
        ax.axvline(size_cross, color='orange', linestyle='--', alpha=0.7)
        ax.text(size_cross, max(max(std_size_kb), max(nova_size_kb)) * 0.8,
                f'Size crossover ~ {size_cross}', color='orange')
    ax.set_xlabel('IoT Readings')
    ax.set_ylabel('Total proof size (KB)')
    ax.set_title('Total size vs readings')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Ratio
    ax = axes[2]
    ax.plot(readings, ratio, 'g-o', label='Standard / Nova (time)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    if time_cross is not None:
        ax.axvline(time_cross, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('IoT Readings')
    ax.set_ylabel('Time ratio (>1 Nova better)')
    ax.set_title('Efficiency ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle('REAL crossover overview (from JSON)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'real_crossover_overview.png'
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved: {out_file}")
    print(f"   Time crossover: {time_cross}")
    print(f"   Size crossover: {size_cross}")
    return out_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot crossover overview from JSON')
    p.add_argument('--json', type=str, default='data/real_measurements/crossover_results.json')
    p.add_argument('--out-dir', type=str, default='data/visualizations')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    plot_overview(json_path, out_dir)


if __name__ == '__main__':
    main()


