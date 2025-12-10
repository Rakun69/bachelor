#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_csv(path: Path):
    readings, std_time, nova_time = [], [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            readings.append(int(row["IoT Readings"]))
            std_time.append(float(row["Standard Zeit (s)"]))
            nova_time.append(float(row["Nova Zeit (s)"]))
    return readings, std_time, nova_time


def plot_nr6(csv_path: Path, out_dir: Path):
    read, std, nova = load_csv(csv_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Farben (peppig)
    red = "#e63946"   # kräftiges Rot
    blue = "#1d4ed8"  # kräftiges Blau
    green = "#2b9348" # crossover Farbe

    # Plots
    plt.plot(read, std,  color=red,  marker='o', linewidth=2.6, label="Standard total (s)")
    plt.plot(read, nova, color=blue, marker='o', linewidth=2.6, label="Nova total (s)")

    # ------------------------------
    # HARD-CODED CROSSOVER POINT
    # ------------------------------
    CROSS_X = 600

    max_y = max(max(std), max(nova))

    plt.axvline(CROSS_X, color=green, linestyle="--", linewidth=1.8, alpha=0.9)
    plt.text(
        CROSS_X + 15,
        max_y * 0.75,
        f"Crossover ~ {CROSS_X}",
        color=green,
        fontsize=10
    )

    # Layout
    plt.xlabel("IoT Readings")
    plt.ylabel("Total time (s)")
    plt.title("Total time vs readings")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save
    out_file = out_dir / "nr6_plot.png"
    plt.savefig(out_file, dpi=220, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True,
                   help="CSV with Standard + Nova results for nr6")
    p.add_argument("--out-dir", type=str, default="data/visualizations")
    return p.parse_args()


def main():
    args = parse_args()
    plot_nr6(Path(args.csv), Path(args.out_dir))


if __name__ == "__main__":
    main()
