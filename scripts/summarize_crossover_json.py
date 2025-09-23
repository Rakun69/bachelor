#!/usr/bin/env python3
"""
Summarize data/real_measurements/crossover_results.json into:
- results_summary/real_crossover_table.md (Markdown table)
- results_summary/real_crossover_table.csv (flat CSV)

Usage:
  PYTHONPATH=. python scripts/summarize_crossover_json.py \
    --json data/real_measurements/crossover_results.json \
    --out-md results_summary/real_crossover_table.md \
    --out-csv results_summary/real_crossover_table.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    results = sorted(results, key=lambda r: r.get("iot_readings", 0))
    return results


def write_markdown_table(results: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("### REAL Crossover Results (from JSON)")
    lines.append("")
    lines.append("| IoT Readings | Standard Total (s) | Standard Size (KB) | Nova Total (s) | Nova Size (KB) | Time Advantage (Std/Nova) | Winner |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")

    for row in results:
        n = int(row.get("iot_readings", 0))
        std = row.get("standard", {})
        nova = row.get("nova", {})
        std_total = float(std.get("total_time_s", 0.0))
        std_size_kb = float(std.get("total_proof_size_bytes", 0) / 1024.0)
        nova_total = float(nova.get("total_time_s", 0.0))
        nova_size_kb = float(nova.get("proof_size_bytes", 0) / 1024.0)
        ratio = float(row.get("time_advantage", 0.0))
        winner = str(row.get("winner", "-") )
        lines.append(
            f"| {n} | {std_total:.2f} | {std_size_kb:.1f} | {nova_total:.2f} | {nova_size_kb:.1f} | {ratio:.2f}x | {winner} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flat_csv(results: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("IoT Readings,Standard Total (s),Standard Size (KB),Nova Total (s),Nova Size (KB),Time Advantage (Std/Nova),Winner\n")
        for row in results:
            n = int(row.get("iot_readings", 0))
            std = row.get("standard", {})
            nova = row.get("nova", {})
            std_total = float(std.get("total_time_s", 0.0))
            std_size_kb = float(std.get("total_proof_size_bytes", 0) / 1024.0)
            nova_total = float(nova.get("total_time_s", 0.0))
            nova_size_kb = float(nova.get("proof_size_bytes", 0) / 1024.0)
            ratio = float(row.get("time_advantage", 0.0))
            winner = str(row.get("winner", "-") )
            f.write(f"{n},{std_total:.2f},{std_size_kb:.1f},{nova_total:.2f},{nova_size_kb:.1f},{ratio:.2f}x,{winner}\n")


def compute_insights(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    time_crossover = None
    for row in results:
        if float(row.get("time_advantage", 0.0)) > 1.0:
            time_crossover = int(row.get("iot_readings", 0))
            break

    size_crossover = None
    for row in results:
        std_size = float(row.get("standard", {}).get("total_proof_size_bytes", 0) / 1024.0)
        nova_size = float(row.get("nova", {}).get("proof_size_bytes", 0) / 1024.0)
        if std_size > nova_size:
            size_crossover = int(row.get("iot_readings", 0))
            break

    return {"time_crossover": time_crossover, "size_crossover": size_crossover}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize crossover JSON into table files")
    p.add_argument("--json", type=str, default="data/real_measurements/crossover_results.json")
    p.add_argument("--out-md", type=str, default="results_summary/real_crossover_table.md")
    p.add_argument("--out-csv", type=str, default="results_summary/real_crossover_table.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    results = load_results(json_path)
    write_markdown_table(results, out_md)
    write_flat_csv(results, out_csv)
    insights = compute_insights(results)

    print("\n✅ Summary written:")
    print(f"  Markdown: {out_md}")
    print(f"  CSV:      {out_csv}")
    print(f"  Time crossover: {insights['time_crossover']}")
    print(f"  Size crossover: {insights['size_crossover']}")


if __name__ == "__main__":
    main()


