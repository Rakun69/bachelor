#!/usr/bin/env python3
"""
Plot Nova + Recursive BatchSize1 + Recursive BatchSize20 from CSV files.

Usage:
  PYTHONPATH=. python scripts/plot_crossover_csv.py \
    --nova data/visualizations/md_warm_nr6/crossover_summary.csv \
    --rec1 data/visualizations/md_warm_nr11/crossover_summary.csv \
    --rec20 data/visualizations/md_warm_nr8/crossover_summary.csv \
    --out-dir data/visualizations
"""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_csv(path: Path):
    readings = []
    std_time = []
    nova_time = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            readings.append(int(row["IoT Readings"]))
            std_time.append(float(row["Standard Zeit (s)"]))
            nova_time.append(float(row["Nova Zeit (s)"]))

    return readings, std_time, nova_time


def plot_three(nova_csv: Path, rec1_csv: Path, rec20_csv: Path, out_dir: Path):
    n_read, n_std, n_nova = load_csv(nova_csv)
    r1_read, r1_std, r1_nova = load_csv(rec1_csv)
    r20_read, r20_std, r20_nova = load_csv(rec20_csv)

    # Recursive times = Nova column in respective CSVs
    rec1_time = r1_nova
    rec20_time = r20_nova

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(n_read, n_nova, 'b-o', label="Nova")
    plt.plot(r1_read, rec1_time, 'r-o', label="Recursive (BatchSize=1)")
    plt.plot(r20_read, rec20_time, 'g-o', label="Recursive (BatchSize=20)")

    plt.xlabel("IoT Readings")
    plt.ylabel("Total Proof Time (s)")
    plt.title("Nova vs Recursive (BS=1) vs Recursive (BS=20)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_file = out_dir / "nova_recursive_comparison.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out_file}")
    return out_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nova", type=str, required=True, help="CSV with Nova results")
    p.add_argument("--rec1", type=str, required=True, help="CSV with Recursive BS=1")
    p.add_argument("--rec20", type=str, required=True, help="CSV with Recursive BS=20")
    p.add_argument("--out-dir", type=str, default="data/visualizations")
    return p.parse_args()


def main():
    args = parse_args()
    plot_three(
        nova_csv=Path(args.nova),
        rec1_csv=Path(args.rec1),
        rec20_csv=Path(args.rec20),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
