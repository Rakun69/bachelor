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
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography import x509


import numpy as np

from src.proof_systems.snark_manager import SNARKManager


LOGGER = logging.getLogger("measure_crossover_real")


# Lädt CA- und Device-Zertifikate, prüft Vertrauenskette und gibt Device-Public-Key zurück
def load_and_verify_device_cert():
    ca_cert_path = Path("data/device_keys/ca_cert.pem")
    dev_cert_path = Path("data/device_keys/device1_cert.pem")
    if not ca_cert_path.exists() or not dev_cert_path.exists():
        raise FileNotFoundError("CA or device certificate missing. Run setup script first.")

    with open(ca_cert_path, "rb") as f:
        ca_cert = x509.load_pem_x509_certificate(f.read())
    with open(dev_cert_path, "rb") as f:
        dev_cert = x509.load_pem_x509_certificate(f.read())

    if dev_cert.issuer != ca_cert.subject:
        raise ValueError("Device cert issuer does not match CA subject")

    ca_pub = ca_cert.public_key()
    try:
        ca_pub.verify(dev_cert.signature, dev_cert.tbs_certificate_bytes)
    except Exception as e:
        raise ValueError(f"Device certificate signature invalid: {e}")

    now = time.time()
    if dev_cert.not_valid_before.timestamp() - 60 > now or dev_cert.not_valid_after.timestamp() + 60 < now:
        raise ValueError("Device certificate not currently valid")

    return dev_cert.public_key()


# Verifiziert Signatur eines einzelnen IoT-Readings
def verify_entry(public_key, entry):
    data = entry["data"]
    sig = bytes.fromhex(entry["signature"])
    msg = json.dumps(data, sort_keys=True).encode("utf-8")
    try:
        public_key.verify(sig, msg)
        return True
    except InvalidSignature:
        return False


# Lädt IoT-Readings, validiert Zertifikate, gibt nur gültige Messdaten zurück
def load_iot_data(project_root: Path) -> List[Dict[str, Any]]:
    fp = project_root / "data/raw/iot_readings_1_month.json"
    if not fp.exists():
        raise FileNotFoundError(f"IoT readings file not found: {fp}")

    with open(fp, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    try:
        public_key = load_and_verify_device_cert()
        LOGGER.info("Using device public key from verified certificate (CA)")
    except Exception as e:
        LOGGER.warning(f"Falling back to pinned public key: {e}")
        public_key = load_public_key()
    verified = []
    for entry in raw_data:
        if verify_entry(public_key, entry):
            verified.append(entry["data"])
        else:
            LOGGER.warning("⚠️ Invalid signature, skipping entry")

    LOGGER.info("Loaded %d valid IoT readings", len(verified))
    return verified


# Checkt Ausgabeordner
def ensure_dirs(project_root: Path) -> Path:
    out_dir = project_root / "data/real_measurements"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# Bereitet Standard-Messpfad vor
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


# Standard-ZoKrates-Pfad
def measure_standard(manager: SNARKManager, circuit_name: str, readings: List[Dict[str, Any]]) -> Dict[str, Any]:
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

        t0 = time.time()
        result = manager.generate_proof(circuit_name, inputs)
        t1 = time.time()
        prove_duration = t1 - t0

        verify_duration = result.metrics.verify_time if result.metrics else 0.0
        verify_ok = result.success
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


# Nova-Input-Dateien
def prepare_nova_inputs(n: int, readings: List[Dict[str, Any]], *, isolated: bool = True, batch_size: int = 3) -> Tuple[Path, int]:
    project_root = Path.cwd()
    base_dir = project_root / "circuits/nova"
    if not base_dir.exists():
        raise FileNotFoundError("circuits/nova missing. Ensure Nova circuit exists.")

    if isolated:
        runs_root = project_root / "circuits/nova_runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        nova_dir = runs_root / f"run_{n}_{int(time.time()*1000)}"
        nova_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["iot_recursive.zok"]:
            src = base_dir / fname
            if src.exists():
                shutil.copy(src, nova_dir / fname)
    else:
        nova_dir = base_dir

    steps: List[Dict[str, Any]] = []
    vals: List[int] = []
    for i in range(n):
        try:
            v = int(round(float(readings[i].get("value", 0.0)) * 10))
        except Exception:
            v = 0
        vals.append(max(0, v))

    batch_size = max(1, int(batch_size))

    while len(vals) % batch_size != 0:
        vals.append(0)

    batch_id = 1
    for i in range(0, len(vals), batch_size):
        batch_values = [str(vals[i + j]) for j in range(batch_size)]
        steps.append({
            "values": batch_values,
            "batch_id": str(batch_id),
        })
        batch_id += 1

    with open(nova_dir / "init.json", "w", encoding="utf-8") as f:
        json.dump({"sum": "0", "count": "0"}, f)
    with open(nova_dir / "steps.json", "w", encoding="utf-8") as f:
        json.dump(steps, f)

    try:
        zok_path = nova_dir / "iot_recursive.zok"
        if zok_path.exists():
            import re
            contents = zok_path.read_text(encoding="utf-8")
            contents = re.sub(r"field\[\d+\]", f"field[{batch_size}]", contents)
            contents = re.sub(r"0\.\.\d+", f"0..{batch_size}", contents)
            zok_path.write_text(contents, encoding="utf-8")
        _compile_nova_circuit(nova_dir)
    except Exception:
        pass

    return nova_dir, len(steps)


# Nova-Circuit mit Setup
def _compile_nova_circuit(nova_dir: Path) -> None:
    import subprocess
    import os
    
    original_cwd = os.getcwd()
    os.chdir(nova_dir)
    
    try:
        LOGGER.info("Recompiling Nova circuit to ensure consistency with batch size.")

        if not Path("iot_recursive.zok").exists():
            raise FileNotFoundError(f"iot_recursive.zok missing in {nova_dir}")

        result = subprocess.run(
            ["zokrates", "compile", "-i", "iot_recursive.zok", "--curve", "pallas"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            stdlib_paths = [
                "/usr/local/bin/stdlib",
                "/usr/local/lib/stdlib", 
                "/opt/zokrates/stdlib",
                "/root/.zokrates/stdlib"
            ]
            
            for stdlib_path in stdlib_paths:
                if os.path.exists(stdlib_path):
                    result = subprocess.run(
                        ["zokrates", "compile", "-i", "iot_recursive.zok", "--curve", "pallas", "--stdlib-path", stdlib_path],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        break
        
        if result.returncode != 0:
            raise RuntimeError(f"ZoKrates compile failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        
        result = subprocess.run(
            ["zokrates", "nova", "setup"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ZoKrates setup failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            
        LOGGER.info("Nova Circuit erfolgreich kompiliert und eingerichtet")
        
    except subprocess.TimeoutExpired:
        LOGGER.error("Nova Circuit Kompilierung timeout")
    except Exception as e:
        LOGGER.error(f"Fehler beim Kompilieren des Nova Circuits: {e}")
        raise
    finally:
        os.chdir(original_cwd)


# Vorwärmen, um Messungen mit weniger Dateisystem-Einflüssen zu stabilisieren
def _prewarm_nova_artifacts(nova_dir: Path) -> None:
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
            pass


# Nova-Proof-Workflow
def run_nova(
    nova_dir: Path,
    *,
    cleanup_artifacts: bool = True,
    prewarm: bool = False,
    compress: bool = True,
) -> Dict[str, Any]:
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

        for req in ["out.r1cs", "out", "abi.json", "nova.params", "init.json", "steps.json"]:
            if not (nova_dir / req).exists():
                raise RuntimeError(f"Missing required artifact: {req}")

        t0 = time.time()
        r_prove = subprocess.run(["zokrates", "nova", "prove"], capture_output=True, text=True)
        if r_prove.returncode != 0:
            raise RuntimeError(f"nova prove failed:\nSTDOUT:\n{r_prove.stdout}\nSTDERR:\n{r_prove.stderr}")

        verify_time = 0.0
        if compress:
            r_comp = subprocess.run(["zokrates", "nova", "compress"], capture_output=True, text=True)
            if r_comp.returncode != 0:
                raise RuntimeError(f"nova compress failed:\nSTDOUT:\n{r_comp.stdout}\nSTDERR:\n{r_comp.stderr}")
            prove_time = time.time() - t0

            t1 = time.time()
            r_ver = subprocess.run(["zokrates", "nova", "verify"], capture_output=True, text=True)
            if r_ver.returncode != 0:
                raise RuntimeError(f"nova verify failed:\nSTDOUT:\n{r_ver.stdout}\nSTDERR:\n{r_ver.stderr}")
            verify_time = time.time() - t1

            proof_path = nova_dir / "proof.json"
            proof_size = proof_path.stat().st_size if proof_path.exists() else 0
        else:
            prove_time = time.time() - t0
            acc = nova_dir / "running_instance.json"
            steps = nova_dir / "steps.json"
            proof_size = 0
            if acc.exists():
                proof_size += acc.stat().st_size
            if steps.exists():
                proof_size += steps.stat().st_size

        return {
            "success": True,
            "prove_time_s": prove_time,
            "verify_time_s": verify_time,
            "total_time_s": prove_time + verify_time,
            "proof_size_bytes": proof_size,
            "compressed": compress,
        }
    finally:
        os.chdir(original_cwd)


# Non-ZK-Range-Check
def _compute_nonzk_range_pass(values_scaled_100: List[int], lower: int, upper: int) -> Dict[str, Any]:
    total = len(values_scaled_100)
    passes = 0
    for v in values_scaled_100:
        if lower <= v <= upper:
            passes += 1
    return {"total": total, "passes": passes, "fails": max(0, total - passes)}


# Non-ZK-Baseline
def measure_nonzk(readings: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.time()
    vals: List[int] = []
    for r in readings:
        try:
            v = int(round(float(r.get("value", 0.0)) * 100))
        except Exception:
            v = 0
        vals.append(v)
    lower = 0
    upper = 10_000_000
    stats = _compute_nonzk_range_pass(vals, lower, upper)
    total_time = time.time() - t0
    artifact = {"kind": "nonzk_baseline_range", "stats": stats}
    artifact_size = len(json.dumps(artifact, separators=(",", ":")).encode("utf-8"))
    return {
        "success": True,
        "total_time_s": total_time,
        "artifact_size_bytes": artifact_size,
    }


# Messreihe über verschiedene Reading-Anzahlen und vergleicht Standard und Nova
def measure_for_counts(
    project_root: Path,
    counts: List[int],
    *,
    warmup_runs: int = 1,
    repetitions: int = 3,
    warm_mode: bool = False,
    batch_size: int = 3,
    nova_compress: bool = True,
) -> Dict[str, Any]:
    data = load_iot_data(project_root)
    out_dir = ensure_dirs(project_root)

    manager = SNARKManager(circuits_dir=str(project_root / "circuits"), output_dir=str(project_root / "data/proofs"))
    circuit_name = compile_and_setup_standard(manager, project_root)

    results: List[Dict[str, Any]] = []
    for n in counts:
        LOGGER.info("\n=== Measuring with %d IoT readings ===", n)
        subset = data[:n]

        for _ in range(max(0, warmup_runs)):
            _ = measure_standard(manager, circuit_name, subset)
            nova_dir_tmp, _ = prepare_nova_inputs(n, subset, isolated=not warm_mode, batch_size=batch_size)
            _ = run_nova(nova_dir_tmp, cleanup_artifacts=not warm_mode, prewarm=warm_mode, compress=nova_compress)
            try:
                runs_root = project_root / "circuits/nova_runs"
                if str(nova_dir_tmp).startswith(str(runs_root)) and nova_dir_tmp.exists():
                    shutil.rmtree(nova_dir_tmp, ignore_errors=True)
            except Exception:
                pass

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
        baseline_totals: List[float] = []
        baseline_sizes: List[float] = []
        for _ in range(max(1, repetitions)):
            std_m = measure_standard(manager, circuit_name, subset)
            std_totals.append(std_m["total_time_s"])
            std_proves.append(std_m["prove_time_s"])
            std_verifs.append(std_m["verify_time_s"])
            std_sizes.append(float(std_m["total_proof_size_bytes"]))
            std_all_individual_times.append(std_m.get("individual_times", []))

            nova_dir, steps = prepare_nova_inputs(n, subset, isolated=not warm_mode, batch_size=batch_size)
            nova_m = run_nova(nova_dir, cleanup_artifacts=not warm_mode, prewarm=warm_mode, compress=nova_compress)
            nova_totals.append(nova_m["total_time_s"])
            nova_proves.append(nova_m["prove_time_s"])
            nova_compress.append(0.0)
            nova_verifs.append(nova_m["verify_time_s"])
            nova_sizes.append(float(nova_m["proof_size_bytes"]))

            bz = measure_nonzk(subset)
            baseline_totals.append(bz.get("total_time_s", 0.0))
            baseline_sizes.append(float(bz.get("artifact_size_bytes", 0)))

            try:
                runs_root = project_root / "circuits/nova_runs"
                if str(nova_dir).startswith(str(runs_root)) and nova_dir.exists():
                    shutil.rmtree(nova_dir, ignore_errors=True)
            except Exception:
                pass

        std_metrics = {
            "success": True,
            "proofs_attempted": len(subset),
            "proofs_successful": len(subset),
            "witness_time_s": 0.0,  
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
            "baseline": {
                "total_time_s": mean(baseline_totals),
                "artifact_size_bytes": int(mean(baseline_sizes)),
            },
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



# Standard Argumente für Config
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure real crossover under resource limits")
    p.add_argument(
        "--reading-counts",
        type=str,
        default="36,43,88", 
        help="Comma-separated IoT reading counts to test",
    )
    p.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs discarded before each measurement")
    p.add_argument("--repetitions", type=int, default=3, help="Number of measured repetitions per count; means are reported")
    p.add_argument("--mode", type=str, choices=["warm", "cold"], default="cold", help="Warm keeps artifacts (prewarmed), Cold cleans artifacts per run")
    p.add_argument("--batch-size", type=int, default=3, help="Nova step size (values per step) to compile and use")
    p.add_argument("--nova-compress", action="store_true", help="Use Nova compression (prove+compress+verify). If omitted, run uncompressed and skip verify.")
    return p.parse_args()


# Main
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
        batch_size=max(1, args.batch_size),
        nova_compress=bool(args.nova_compress),
    )

    print("\n🎯 REAL CROSSOVER MEASUREMENT COMPLETED")
    print(f"   Crossover point (time): {summary['crossover_point']}")
    print(f"   CSV: {summary['csv']}")
    print(f"   JSON: {summary['json']}")


if __name__ == "__main__":
    main()


