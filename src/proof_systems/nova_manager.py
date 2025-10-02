"""
Unified ZoKrates Nova Manager (based on fixed implementation)
"""

from __future__ import annotations

import json
import subprocess
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class NovaProofResult:
    success: bool
    proof_data: Optional[str] = None
    compressed_proof: Optional[str] = None
    step_count: int = 0
    total_time: float = 0.0
    verify_time: float = 0.0
    proof_size: int = 0
    error_message: Optional[str] = None


class NovaManager:
    """
    Unified Nova manager using ZoKrates experimental Nova CLI.
    Works directly in `circuits/nova` with precompiled artifacts.
    """

    def __init__(self, circuit_dir: str = "circuits/nova", batch_size: int = 3):
        self.working_dir = Path(circuit_dir)
        self.batch_size = batch_size
        self.setup_done = False

    def check_zokrates_nova_support(self) -> bool:
        try:
            result = subprocess.run(["zokrates", "nova", "--help"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def setup(self) -> bool:
        """Validate presence of required Nova artifacts in working dir."""
        try:
            if not self.working_dir.exists():
                logger.error(f"Nova working directory not found: {self.working_dir}")
                return False

            for req in ("out", "nova.params", "abi.json"):
                if not (self.working_dir / req).exists():
                    logger.error(f"Required Nova file missing: {req}")
                    return False

            self.setup_done = True
            return True
        except Exception as e:
            logger.error(f"Nova setup check failed: {e}")
            return False

    def prepare_initial_state(self) -> Dict[str, str]:
        return {"sum": "0", "count": "0"}

    def prepare_step_input(self, iot_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        values: List[str] = []
        for i in range(3):
            if i < len(iot_data) and iot_data[i]:
                try:
                    value = int(float(iot_data[i].get("value", 0)) * 10)
                except Exception:
                    value = 0
                values.append(str(max(0, value)))
            else:
                values.append("0")
        return {"values": values, "batch_id": str(int(time.time()) % 1000)}

    def _execute_nova_proof(self, initial_state: Dict[str, str], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            original_cwd = os.getcwd()
            os.chdir(self.working_dir)

            with open("init.json", "w", encoding="utf-8") as f:
                json.dump(initial_state, f)
            with open("steps.json", "w", encoding="utf-8") as f:
                json.dump(steps, f)

            prove_start = time.time()
            r_prove = subprocess.run(["zokrates", "nova", "prove"], capture_output=True, text=True, timeout=300)
            prove_time = time.time() - prove_start
            if r_prove.returncode != 0:
                return {"success": False, "error": f"Nova prove failed: {r_prove.stderr}"}

            r_comp = subprocess.run(["zokrates", "nova", "compress"], capture_output=True, text=True, timeout=60)
            if r_comp.returncode != 0:
                return {"success": False, "error": f"Nova compress failed: {r_comp.stderr}"}

            verify_start = time.time()
            r_ver = subprocess.run(["zokrates", "nova", "verify"], capture_output=True, text=True, timeout=60)
            verify_time = time.time() - verify_start
            if r_ver.returncode != 0:
                return {"success": False, "error": f"Nova verify failed: {r_ver.stderr}"}

            proof_size = 0
            p = self.working_dir / "proof.json"
            if p.exists():
                proof_size = p.stat().st_size
                with p.open("r", encoding="utf-8") as f:
                    compressed_proof = f.read()
            else:
                compressed_proof = None

            return {
                "success": True,
                "proof": compressed_proof,
                "compressed_proof": compressed_proof,
                "verify_time": verify_time,
                "proof_size": proof_size,
                "prove_time": prove_time,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Nova proof generation timeout"}
        except Exception as e:
            return {"success": False, "error": f"Nova execution error: {str(e)}"}
        finally:
            os.chdir(original_cwd)

    def prove_recursive_batch(self, iot_batches: List[List[Dict[str, Any]]]) -> NovaProofResult:
        if not self.setup_done:
            if not self.setup():
                return NovaProofResult(success=False, error_message="Nova setup failed")

        try:
            start_time = time.time()
            initial_state = self.prepare_initial_state()
            steps = [self.prepare_step_input(batch) for batch in iot_batches]
            result = self._execute_nova_proof(initial_state, steps)
            total_time = time.time() - start_time
            if result.get("success"):
                return NovaProofResult(
                    success=True,
                    proof_data=result.get("proof"),
                    compressed_proof=result.get("compressed_proof"),
                    step_count=len(steps),
                    total_time=total_time,
                    verify_time=result.get("verify_time", 0.0),
                    proof_size=result.get("proof_size", 0),
                )
            return NovaProofResult(success=False, error_message=result.get("error", "unknown_error"))
        except Exception as e:
            logger.error(f"Nova proof generation failed: {e}")
            return NovaProofResult(success=False, error_message=str(e))


