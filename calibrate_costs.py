from pathlib import Path
import argparse
from src.evaluation.benchmark_framework import BenchmarkFramework, BenchmarkConfig
from src.proof_systems.snark_manager import SNARKManager
from src.iot_simulation.smart_home import SmartHomeSensors

"""Calibration runner for empirical cost constants.

Uses the real SNARKManager (ZoKrates CLI) and SmartHomeSensors simulator
to measure proof times, verify times, proof sizes etc. Writes calibrated
constants to data/benchmarks/calibrated_costs.json.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    out_dir = str(Path(__file__).resolve().parent / "data" / "benchmarks")
    cfg = BenchmarkConfig(
        circuit_types=["median"],
        data_sizes=[args.data_size],
        batch_sizes=[args.batch_size],
        privacy_levels=[2],
        iterations=1,
        output_dir=out_dir,
    )
    framework = BenchmarkFramework(cfg)
    # Real manager + simulator
    snark = SNARKManager()
    sim = SmartHomeSensors()

    # Ensure the reference circuit is compiled + setup once (median example)
    circuits_root = Path("/home/ramon/bachelor/circuits/basic")
    median_zok = circuits_root / "median.zok"
    if median_zok.exists():
        snark.compile_circuit(str(median_zok), "median")
        snark.setup_circuit("median")

    constants = framework.calibrate_cost_constants(
        snark, sim, data_size=args.data_size, batch_size=args.batch_size
    )
    print(constants)
    print(f"Saved to {Path(out_dir) / 'calibrated_costs.json'}")

if __name__ == "__main__":
    main()