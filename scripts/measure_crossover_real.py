from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple
from statistics import mean


import numpy as np

# Local imports
from src.proof_systems.snark_manager import SNARKManager


LOGGER = logging.getLogger("measure_crossover_real")


def load_iot_data(project_root: Path) -> List[Dict[str, Any]]:
    """Load real IoT readings from 1_month.json file only."""
    fp = project_root / "data/raw/iot_readings_1_month.json"
    if not fp.exists():
        raise FileNotFoundError(f"IoT readings file not found: {fp}")
    
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    LOGGER.info("Loaded IoT data: %s (%d readings)", fp, len(data))
    return data


def ensure_dirs(project_root: Path) -> Path:
    out_dir = project_root / "data/real_measurements"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def compile_and_setup_standard(manager: SNARKManager, project_root: Path) -> str:
    circuit_file = project_root / "circuits/basic/filter_range.zok"
    circuit_name = "filter_range"
    if not circuit_file.exists():
        raise FileNotFoundError(f"Missing circuit: {circuit_file}")

    if not manager.compile_circuit(str(circuit_file), circuit_name):
        raise RuntimeError("ZoKrates compile failed for filter_range")
    if not manager.setup_circuit(circuit_name):
        raise RuntimeError("ZoKrates setup failed for filter_range")
    return circuit_name


def measure_standard(manager: SNARKManager, circuit_name: str, readings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate N real proofs and aggregate metrics.
    Symmetrisch zu Nova: Proving explizit gemessen, Verify separat gestoppt.
    """
    total_proof_time = 0.0
    total_verify_time = 0.0
    total_size = 0
    successes = 0

    individual_times: List[float] = []

    MIN_VAL = "0"
    MAX_VAL = "10000000"  # 1e7

    for idx, r in enumerate(readings):
        val = r.get("value", 0.0)
        try:
            secret_value = str(int(round(float(val) * 100)))
        except Exception:
            secret_value = "0"

        inputs = [MIN_VAL, MAX_VAL, secret_value]

        # --- Proving ---
        t0 = time.time()
        result = manager.generate_proof(circuit_name, inputs)
        t1 = time.time()
        prove_duration = t1 - t0

        # Verification already done in generate_proof, use those results
        verify_duration = result.metrics.verify_time if result.metrics else 0.0
        verify_ok = result.success  # If proof generation succeeded, verification also succeeded

        individual_times.append(prove_duration + verify_duration)
        LOGGER.info(
            f"[measure_standard] Proof #{idx+1}/{len(readings)}: "
            f"prove={prove_duration:.4f}s, verify={verify_duration:.4f}s"
        )

        if result.success:
            successes += 1
            total_proof_time += prove_duration
            total_verify_time += verify_duration
            total_size += result.metrics.proof_size if result.metrics else 0
        else:
            LOGGER.warning("Standard proof %d failed", idx + 1)

    return {
        "success": successes == len(readings),
        "proofs_attempted": len(readings),
        "proofs_successful": successes,
        "prove_time_s": total_proof_time,
        "verify_time_s": total_verify_time,
        "total_time_s": total_proof_time + total_verify_time,
        "total_proof_size_bytes": total_size,
        "avg_proof_size_bytes": (total_size // max(successes, 1)),
        "individual_times": individual_times,
    }





def prepare_nova_inputs(n: int, readings: List[Dict[str, Any]], *, isolated: bool = True) -> Tuple[Path, int]:
    """Prepare Nova workspace and write inputs; returns nova dir and number of steps.

    - If isolated=True (default), create a fresh temp workspace under circuits/nova_runs/
      and copy the static artifacts from circuits/nova. This avoids cross-run caching
      effects which can cause time irregularities.
    - Circuit expects 3 items per step (as in iot_recursive.zok). We pad with zeros.
    """
    project_root = Path.cwd()
    base_dir = project_root / "circuits/nova"
    if not base_dir.exists():
        raise FileNotFoundError("circuits/nova missing. Ensure Nova circuit exists.")

    if isolated:
        runs_root = project_root / "circuits/nova_runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        nova_dir = runs_root / f"run_{n}_{int(time.time()*1000)}"
        nova_dir.mkdir(parents=True, exist_ok=True)
        # Copy required static files
        for fname in ["out", "out.r1cs", "abi.json", "nova.params"]:
            src = base_dir / fname
            if src.exists():
                shutil.copy(src, nova_dir / fname)
    else:
        nova_dir = base_dir

    # Build steps of 3 integers
    steps: List[Dict[str, Any]] = []
    vals: List[int] = []
    for i in range(n):
        try:
            v = int(round(float(readings[i].get("value", 0.0)) * 10))
        except Exception:
            v = 0
        vals.append(max(0, v))

    # pad to multiple of 3
    while len(vals) % 3 != 0:
        vals.append(0)

    batch_id = 1
    for i in range(0, len(vals), 3):
        steps.append({
            "values": [str(vals[i]), str(vals[i + 1]), str(vals[i + 2])],
            "batch_id": str(batch_id),
        })
        batch_id += 1

    with open(nova_dir / "init.json", "w", encoding="utf-8") as f:
        json.dump({"sum": "0", "count": "0"}, f)
    with open(nova_dir / "steps.json", "w", encoding="utf-8") as f:
        json.dump(steps, f)

    return nova_dir, len(steps)


def _prewarm_nova_artifacts(nova_dir: Path) -> None:
    """Read large/static files to warm the OS page cache for stable timings."""
    targets = [
        nova_dir / "out.r1cs",
        nova_dir / "nova.params",
        nova_dir / "out",
        nova_dir / "abi.json",
    ]
    for path in targets:
        try:
            if path.exists():
                with open(path, "rb") as fh:
                    while fh.read(8 * 1024 * 1024):
                        pass
        except Exception:
            # Best-effort warmup
            pass


def run_nova(nova_dir: Path, *, cleanup_artifacts: bool = True, prewarm: bool = False) -> Dict[str, Any]:
    """Run zokrates nova prove/compress/verify and measure durations + proof size.
    Symmetrisch zu Standard: Proving = prove+compress, Verify = separat gestoppt.
    """
    import subprocess

    original_cwd = Path.cwd()
    os.chdir(nova_dir)
    try:
        if cleanup_artifacts:
            for fname in [
                "proof.json",
                "proof_compressed.json",
                "running_instance.json",
                "proving.key",
                "verification.key",
            ]:
                try:
                    p = Path(fname)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        if prewarm:
            _prewarm_nova_artifacts(nova_dir)

        # --- Prove + Compress ---
        t0 = time.time()
        r_prove = subprocess.run(["zokrates", "nova", "prove"], capture_output=True, text=True)
        if r_prove.returncode != 0:
            raise RuntimeError(f"nova prove failed: {r_prove.stderr}")

        r_comp = subprocess.run(["zokrates", "nova", "compress"], capture_output=True, text=True)
        if r_comp.returncode != 0:
            raise RuntimeError(f"nova compress failed: {r_comp.stderr}")
        prove_time = time.time() - t0

        # --- Verify separat ---
        t1 = time.time()
        r_ver = subprocess.run(["zokrates", "nova", "verify"], capture_output=True, text=True)
        if r_ver.returncode != 0:
            raise RuntimeError(f"nova verify failed: {r_ver.stderr}")
        verify_time = time.time() - t1

        proof_size = 0
        proof_path = nova_dir / "proof.json"
        if proof_path.exists():
            proof_size = proof_path.stat().st_size

        return {
            "success": True,
            "prove_time_s": prove_time,
            "verify_time_s": verify_time,
            "total_time_s": prove_time + verify_time,
            "proof_size_bytes": proof_size,
        }
    finally:
        os.chdir(original_cwd)



def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.median(arr))


def measure_for_counts(project_root: Path, counts: List[int], *, warmup_runs: int = 1, repetitions: int = 3, warm_mode: bool = False) -> Dict[str, Any]:
    data = load_iot_data(project_root)
    out_dir = ensure_dirs(project_root)

    manager = SNARKManager(circuits_dir=str(project_root / "circuits"), output_dir=str(project_root / "data/proofs"))
    circuit_name = compile_and_setup_standard(manager, project_root)

    results: List[Dict[str, Any]] = []
    for n in counts:
        LOGGER.info("\n=== Measuring with %d IoT readings ===", n)
        subset = data[:n]

        # Optional warm-up runs (discarded) to stabilize caches/keys
        for _ in range(max(0, warmup_runs)):
            _ = measure_standard(manager, circuit_name, subset)
            nova_dir_tmp, _ = prepare_nova_inputs(n, subset, isolated=not warm_mode)
            _ = run_nova(nova_dir_tmp, cleanup_artifacts=not warm_mode, prewarm=warm_mode)
            try:
                runs_root = project_root / "circuits/nova_runs"
                if str(nova_dir_tmp).startswith(str(runs_root)) and nova_dir_tmp.exists():
                    shutil.rmtree(nova_dir_tmp, ignore_errors=True)
            except Exception:
                pass

        # Replicated measurements for robustness
        std_totals: List[float] = []
        std_proves: List[float] = []
        std_verifs: List[float] = []
        std_sizes: List[float] = []

        std_all_individual_times: List[List[float]] = []

        nova_totals: List[float] = []
        nova_proves: List[float] = []
        nova_compress: List[float] = []
        nova_verifs: List[float] = []
        nova_sizes: List[float] = []

        steps = 0
        for _ in range(max(1, repetitions)):
            # Standard replicate
            std_m = measure_standard(manager, circuit_name, subset)
            std_totals.append(std_m["total_time_s"])
            std_proves.append(std_m["prove_time_s"])
            std_verifs.append(std_m["verify_time_s"])
            std_sizes.append(float(std_m["total_proof_size_bytes"]))
            std_all_individual_times.append(std_m.get("individual_times", []))

            # Nova replicate
            nova_dir, steps = prepare_nova_inputs(n, subset, isolated=not warm_mode)
            nova_m = run_nova(nova_dir, cleanup_artifacts=not warm_mode, prewarm=warm_mode)
            nova_totals.append(nova_m["total_time_s"])
            nova_proves.append(nova_m["prove_time_s"])
            nova_compress.append(nova_m.get("compress_time_s", 0.0))
            nova_verifs.append(nova_m["verify_time_s"])
            nova_sizes.append(float(nova_m["proof_size_bytes"]))

            try:
                runs_root = project_root / "circuits/nova_runs"
                if str(nova_dir).startswith(str(runs_root)) and nova_dir.exists():
                    shutil.rmtree(nova_dir, ignore_errors=True)
            except Exception:
                pass

        # Statt Median: benutze Mean für total_time und andere Felder
        std_metrics = {
            "success": True,
            "proofs_attempted": len(subset),
            "proofs_successful": len(subset),
            "witness_time_s": 0.0,  # falls du keine witness_times über Replikationen trackst
            "prove_time_s": mean(std_proves),
            "verify_time_s": mean(std_verifs),
            "total_time_s": mean(std_totals),
            "total_proof_size_bytes": int(mean(std_sizes)),
            "avg_proof_size_bytes": int(max(1, mean(std_sizes) / max(len(subset), 1))),
            "individual_times": std_all_individual_times,
        }

        nova_metrics = {
            "success": True,
            "prove_time_s": mean(nova_proves),
            "compress_time_s": mean(nova_compress),
            "verify_time_s": mean(nova_verifs),
            "total_time_s": mean(nova_totals),
            "proof_size_bytes": int(mean(nova_sizes)),
        }

        LOGGER.info("Standard (mean of %d): total=%.2fs, prove=%.2fs, verify=%.2fs",
                    max(1, repetitions), std_metrics["total_time_s"], std_metrics["prove_time_s"], std_metrics["verify_time_s"])
        LOGGER.info("Nova (mean of %d): total=%.2fs, prove=%.2fs, compress=%.2fs, verify=%.2fs",
                    max(1, repetitions), nova_metrics["total_time_s"], nova_metrics["prove_time_s"], nova_metrics["compress_time_s"], nova_metrics["verify_time_s"])

        time_advantage = (std_metrics["total_time_s"] / max(nova_metrics["total_time_s"], 1e-9))
        size_advantage = (std_metrics["total_proof_size_bytes"] / max(nova_metrics["proof_size_bytes"], 1))

        results.append({
            "iot_readings": n,
            "standard": std_metrics,
            "nova": nova_metrics,
            "nova_steps": steps,
            "time_advantage": time_advantage,
            "size_advantage": size_advantage,
            "winner": "Nova" if time_advantage > 1.0 else "Standard",
            "replications": max(1, repetitions),
            "warmup_runs": max(0, warmup_runs),
            "std_replicates_total_s": std_totals,
            "nova_replicates_total_s": nova_totals,
            "std_replicates_individual_times": std_all_individual_times,
        })

        with open(out_dir / "crossover_results.json", "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)

        try:
            runs_root = project_root / "circuits/nova_runs"
            if str(nova_dir).startswith(str(runs_root)) and nova_dir.exists():
                shutil.rmtree(nova_dir, ignore_errors=True)
        except Exception:
            pass

    # Determine crossover
    crossover_point = None
    for row in sorted(results, key=lambda r: r["iot_readings"]):
        if row["time_advantage"] > 1.0:
            crossover_point = row["iot_readings"]
            break

    csv_path = out_dir / "crossover_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("IoT Readings,Standard Zeit (s),Standard Größe (KB),Nova Zeit (s),Nova Größe (KB),Zeit Vorteil,Größe Vorteil,Gewinner\n")
        for row in sorted(results, key=lambda r: r["iot_readings"]):
            f.write(
                f"{row['iot_readings']},{row['standard']['total_time_s']:.2f},{row['standard']['total_proof_size_bytes']/1024:.1f},"
                f"{row['nova']['total_time_s']:.2f},{row['nova']['proof_size_bytes']/1024:.1f},"
                f"{row['time_advantage']:.2f}x,{row['size_advantage']:.2f}x,{row['winner']}\n"
            )

    return {
        "results": results,
        "crossover_point": crossover_point,
        "csv": str(csv_path),
        "json": str(out_dir / "crossover_results.json"),
    }



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure real crossover under resource limits")
    p.add_argument(
        "--reading-counts",
        type=str,
        default="36,43,88",  #hier reading anzahl verändern
        help="Comma-separated IoT reading counts to test",
    )
    p.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs discarded before each measurement")
    p.add_argument("--repetitions", type=int, default=3, help="Number of measured repetitions per count; medians are reported")
    p.add_argument("--mode", type=str, choices=["warm", "cold"], default="cold", help="Warm keeps artifacts (prewarmed), Cold cleans artifacts per run")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    counts = [int(x.strip()) for x in args.reading_counts.split(",") if x.strip()]

    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    summary = measure_for_counts(
        project_root,
        counts,
        warmup_runs=max(0, args.warmup_runs),
        repetitions=max(1, args.repetitions),
        warm_mode=(args.mode == "warm"),
    )

    print("\n🎯 REAL CROSSOVER MEASUREMENT COMPLETED")
    print(f"   Crossover point (time): {summary['crossover_point']}")
    print(f"   CSV: {summary['csv']}")
    print(f"   JSON: {summary['json']}")


if __name__ == "__main__":
    main()


