#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_column(path: Path, col_name: str):
    readings = []
    values = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            readings.append(int(row["IoT Readings"]))
            values.append(float(row[col_name]))
    return readings, values


def plot_three(nr11_csv: Path, nr6_csv: Path, out_dir: Path):

    # Load data
    read_11, nova_11 = load_column(nr11_csv, "Nova Zeit (s)")
    read_6,  nova_6 = load_column(nr6_csv,  "Nova Zeit (s)")
    _,       std_6  = load_column(nr6_csv,  "Standard Zeit (s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # ================================================
    # COLORS (peppig & professionell)
    # ================================================
    red      = "#e63946"   # peppiges Rot
    blue     = "#1d4ed8"   # kräftiges Blau
    lightblu = "#3cb4e5"   # dein Hellblau bleibt

    green    = "#2b9348"   # crossover-Linien

    # ================================================
    # Plot curves
    # ================================================
    plt.plot(read_6, std_6,
             color=red, marker='o', linewidth=2.6,
             label="Standard total (s) (Batch-Size 1)")

    plt.plot(read_6, nova_6,
             color=blue, marker='o', linewidth=2.6,
             label="Nova total (s) (Batch-Size 1)")

    plt.plot(read_11, nova_11,
             color=lightblu, marker='o', linewidth=2.6,
             label="Nova total (s) (Batch-Size 20)")

    # ================================================
    # Crossover LINES (like your old plot)
    # ================================================
    max_y = max(max(std_6), max(nova_6), max(nova_11))

    # BS1 crossover at x=600
    plt.axvline(600, color=green, linestyle="--", linewidth=1.8, alpha=0.9)
    plt.text(600 + 10, max_y * 0.70, "Crossover ~ 600", color=green,
             fontsize=10)

    # BS20 crossover at x=300
    plt.axvline(300, color=green, linestyle="--", linewidth=1.8, alpha=0.9)
    plt.text(300 + 10, max_y * 0.78, "Crossover ~ 300", color=green,
             fontsize=10)

    # ================================================
    # Labels
    # ================================================
    plt.xlabel("IoT Readings")
    plt.ylabel("Total time (s)")
    plt.title("Total time vs readings")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ================================================
    # Save high DPI
    # ================================================
    out_file = out_dir / "comparison_nr11_nr6.png"
    plt.savefig(out_file, dpi=220, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {out_file}")
    return out_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nr11", type=str, required=True)
    p.add_argument("--nr6",  type=str, required=True)
    p.add_argument("--out-dir", type=str, default="data/visualizations")
    return p.parse_args()


def main():
    args = parse_args()
    plot_three(
        nr11_csv=Path(args.nr11),
        nr6_csv=Path(args.nr6),
        out_dir=Path(args.out_dir)
    )


if __name__ == "__main__":
    main()
