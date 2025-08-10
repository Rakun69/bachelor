import json
import math
from pathlib import Path

from src.evaluation.benchmark_framework import BenchmarkFramework, BenchmarkConfig


class DummySnarkManager:
    def __init__(self):
        self.last_setup_seconds = 0.0

    class Result:
        def __init__(self, success, proof, proof_time, proof_size):
            self.success = success
            self.proof = proof

            class Metrics:
                def __init__(self, proof_time, proof_size):
                    self.proof_time = proof_time
                    self.proof_size = proof_size
            self.metrics = Metrics(proof_time, proof_size)

    def generate_proof(self, circuit_type, inputs):
        # Simulate 783-byte standard proof and ~10ms per proof
        return self.Result(True, b"proof", proof_time=0.01, proof_size=783)

    class RecResult:
        def __init__(self, success, proof_size):
            self.success = success

            class Metrics:
                def __init__(self, proof_size):
                    self.proof_size = proof_size
            self.metrics = Metrics(proof_size)

    def create_recursive_proof(self, individual_proofs):
        # Simulate constant 2048-byte final proof
        return self.RecResult(True, proof_size=2048)


class DummyIoTSimulator:
    class Reading:
        def __init__(self, value, timestamp, sensor_type):
            self.value = value
            self.timestamp = timestamp
            self.sensor_type = sensor_type

    def generate_readings(self, duration_hours, time_step_seconds):
        # 60 readings per hour
        count = int(duration_hours * 3600 / time_step_seconds)
        return [self.Reading(20 + (i % 10), f"2025-01-01T{i:02d}:00:00", "temperature") for i in range(count)]


def test_calibrate_cost_constants(tmp_path: Path):
    cfg = BenchmarkConfig(
        circuit_types=["median"],
        data_sizes=[100],
        batch_sizes=[10],
        privacy_levels=[2],
        iterations=1,
        output_dir=str(tmp_path),
    )
    framework = BenchmarkFramework(cfg)
    snark = DummySnarkManager()
    sim = DummyIoTSimulator()

    constants = framework.calibrate_cost_constants(snark, sim, data_size=100, batch_size=10)

    # Basic sanity checks
    assert constants["C_proof_time_per_item"] > 0
    assert constants["C_storage_proof_bytes"] == 783
    assert constants["final_recursive_proof_bytes"] == 2048

    # Verify n_proofs = ceil(100/10) = 10
    assert constants["n_proofs"] == 10

    # File saved
    out = Path(tmp_path) / "calibrated_costs.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["C_storage_proof_bytes"] == 783

