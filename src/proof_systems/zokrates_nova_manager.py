"""
ZoKrates Nova Recursive SNARK Manager
Implements recursive zkSNARKs using ZoKrates experimental Nova support
"""

import json
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class NovaProofResult:
    """Result of Nova recursive proof generation"""
    success: bool
    proof_data: Optional[str] = None
    compressed_proof: Optional[str] = None
    step_count: int = 0
    total_time: float = 0.0
    verify_time: float = 0.0
    proof_size: int = 0
    error_message: Optional[str] = None
    
    # Metrics compatibility removed - demo.py deleted

@dataclass
class NovaMetrics:
    """Performance metrics for Nova proofs"""
    steps_proved: int
    total_readings: int
    prove_time: float
    compress_time: float
    verify_time: float
    proof_size: int
    compressed_size: int
    throughput: float  # readings per second

class ZoKratesNovaManager:
    """
    Manager for ZoKrates Nova recursive SNARKs
    Implements the experimental Nova proof system in ZoKrates
    """
    
    def __init__(self, circuit_path: str = "circuits/nova/iot_recursive.zok", 
                 batch_size: int = 10):
        self.circuit_path = Path(circuit_path)
        self.batch_size = batch_size
        self.working_dir = Path("nova_workspace")
        self.working_dir.mkdir(exist_ok=True)
        
        # Compiled circuit files
        self.compiled_circuit = self.working_dir / "iot_recursive"
        self.proving_key = self.working_dir / "proving.key"
        self.verification_key = self.working_dir / "verification.key"
        
        # State and proof files
        self.init_state_file = self.working_dir / "init_state.json"
        self.steps_file = self.working_dir / "steps.json"
        self.proof_file = self.working_dir / "proof.json"
        self.compressed_proof_file = self.working_dir / "compressed_proof.json"
        
        self.setup_done = False
        
    def check_zokrates_nova_support(self) -> bool:
        """Check if ZoKrates installation supports Nova"""
        try:
            result = subprocess.run(
                ["zokrates", "nova", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def setup(self) -> bool:
        """
        Setup Nova circuit - compile with Pallas curve
        """
        try:
            if not self.check_zokrates_nova_support():
                logger.error("ZoKrates Nova support not found. Update ZoKrates to latest version.")
                return False
            
            # Change to working directory for ZoKrates operations
            original_cwd = os.getcwd()
            os.chdir(self.working_dir)
            
            try:
                # Copy circuit to working directory
                if self.circuit_path.is_absolute():
                    circuit_source = self.circuit_path
                else:
                    circuit_source = Path(original_cwd) / self.circuit_path
                    
                circuit_dest = Path("iot_recursive.zok")  # Relative to working directory
                
                if not circuit_source.exists():
                    logger.error(f"Circuit file not found: {circuit_source}")
                    return False
                
                # Copy circuit file
                import shutil
                shutil.copy2(str(circuit_source), str(circuit_dest))
                logger.info(f"Copied circuit from {circuit_source} to {circuit_dest}")
                
                logger.info("Compiling Nova circuit with Pallas curve...")
                
                # Compile with Pallas curve (required for Nova)
                compile_result = subprocess.run(
                    ["zokrates", "compile", "-i", "iot_recursive.zok", "--curve", "pallas"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if compile_result.returncode != 0:
                    logger.error(f"Circuit compilation failed: {compile_result.stderr}")
                    return False
                
                logger.info("Nova circuit compiled successfully")
                self.setup_done = True
                return True
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Nova setup failed: {e}")
            return False
    
    def prepare_initial_state(self) -> str:
        """
        Create initial state for Nova recursion (simplified)
        """
        # Nova expects a JSON string with the initial state
        return json.dumps({"sum": "0", "count": "0"})
    
    def prepare_step_input(self, iot_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert IoT data to Nova step input format (simplified for 3 values)
        """
        values = []
        
        # Take up to 3 values, pad with zeros if needed
        for i in range(3):
            if i < len(iot_data) and iot_data[i]:
                # Convert value to field element (scale and round)
                value = int(float(iot_data[i].get('value', 0)) * 100)  # 2 decimal precision
                values.append(str(value))
            else:
                # Padding with zeros
                values.append("0")
        
        return {
            "values": values,
            "batch_id": str(int(time.time()))
        }
    
    def _execute_nova_proof(self, initial_state: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute real Nova proof using ZoKrates CLI commands
        """
        try:
            original_cwd = os.getcwd()
            os.chdir(self.working_dir)
            
            # Write initial state
            with open("init.json", "w") as f:
                f.write(initial_state)
            
            # Write steps
            with open("steps.json", "w") as f:
                json.dump(steps, f)
            
            # Run Nova prove
            prove_start = time.time()
            prove_result = subprocess.run(
                ["zokrates", "nova", "prove"],
                capture_output=True,
                text=True,
                timeout=300
            )
            prove_time = time.time() - prove_start
            
            if prove_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Nova prove failed: {prove_result.stderr}"
                }
            
            # Compress to SNARK
            compress_result = subprocess.run(
                ["zokrates", "nova", "compress"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if compress_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Nova compress failed: {compress_result.stderr}"
                }
            
            # Verify
            verify_start = time.time()
            verify_result = subprocess.run(
                ["zokrates", "nova", "verify"],
                capture_output=True,
                text=True,
                timeout=60
            )
            verify_time = time.time() - verify_start
            
            if verify_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Nova verify failed: {verify_result.stderr}"
                }
            
            # Get proof size
            proof_size = 0
            if Path("proof.json").exists():
                proof_size = Path("proof.json").stat().st_size
            
            # Read compressed proof
            compressed_proof = None
            if Path("proof.json").exists():
                with open("proof.json", "r") as f:
                    compressed_proof = f.read()
            
            return {
                "success": True,
                "proof": compressed_proof,
                "compressed_proof": compressed_proof,
                "verify_time": verify_time,
                "proof_size": proof_size,
                "prove_time": prove_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Nova proof generation timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Nova execution error: {str(e)}"
            }
        finally:
            os.chdir(original_cwd)
    
    def prove_recursive_batch(self, iot_batches: List[List[Dict[str, Any]]]) -> NovaProofResult:
        """
        Generate Nova recursive proof for IoT batches using real ZoKrates Nova
        """
        if not self.setup_done:
            if not self.setup_nova():
                return NovaProofResult(
                    success=False,
                    error_message="recursive_proof_failed: Nova setup failed"
                )
        
        try:
            start_time = time.time()
            
            # Prepare initial state and steps
            initial_state = self.prepare_initial_state()
            steps = []
            
            for batch in iot_batches:
                step_input = self.prepare_step_input(batch)
                steps.append(step_input)
            
            # Execute Nova proof generation
            result = self._execute_nova_proof(initial_state, steps)
            total_time = time.time() - start_time
            
            if result["success"]:
                return NovaProofResult(
                    success=True,
                    proof_data=result["proof"],
                    compressed_proof=result["compressed_proof"],
                    step_count=len(steps),
                    total_time=total_time,
                    verify_time=result["verify_time"],
                    proof_size=result["proof_size"]
                )
            else:
                return NovaProofResult(
                    success=False,
                    error_message=f"recursive_proof_failed: {result['error']}"
                )
                
        except Exception as e:
            logger.error(f"Nova proof generation failed: {e}")
            return NovaProofResult(
                success=False,
                error_message=f"recursive_proof_failed: {str(e)}"
            )
    
    def benchmark_vs_traditional(self, iot_data: List[Dict[str, Any]], 
                                traditional_proof_time: float) -> Dict[str, Any]:
        """
        Real Nova vs Traditional ZoKrates benchmarking
        """
        try:
            # Split data into batches for Nova
            batch_size = self.batch_size
            batches = [iot_data[i:i+batch_size] for i in range(0, len(iot_data), batch_size)]
            
            # Run Nova proof
            nova_result = self.prove_recursive_batch(batches)
            
            if nova_result.success:
                return {
                    "nova_available": True,
                    "nova_metrics": {
                        "proof_time": nova_result.total_time,
                        "verify_time": nova_result.verify_time,
                        "proof_size": nova_result.proof_size,
                        "step_count": nova_result.step_count,
                        "data_size": len(iot_data)
                    },
                    "traditional_metrics": {
                        "proof_time": traditional_proof_time,
                        "data_size": len(iot_data)
                    },
                    "improvements": {
                        "time_speedup": traditional_proof_time / max(nova_result.total_time, 0.001),
                        "compression_factor": len(batches)  # Multiple proofs compressed to one
                    }
                }
            else:
                return {
                    "nova_available": False,
                    "error": nova_result.error_message
                }
        except Exception as e:
            return {
                "nova_available": False,
                "error": f"Nova benchmarking failed: {str(e)}"
            }
    
    def get_nova_advantages_analysis(self) -> Dict[str, Any]:
        """DISABLED: Nova analysis disabled - focusing on standard ZoKrates SNARKs"""
        return {
            "disabled": "Nova recursive SNARKs disabled for thesis",
            "reason": "Focusing on proven standard ZoKrates SNARKs for reliable results"
        }
    
    def cleanup(self):
        """Clean up Nova workspace"""
        import shutil
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)

class SNARKManager:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def compile_circuit(self, circuit_file: str, circuit_name: str) -> Path:
        build_dir = self.project_root / "circuits" / "build_standard"
        build_dir.mkdir(parents=True, exist_ok=True)

        out_path = build_dir / f"{circuit_name}.out"

        # Vorher evtl. Altartefakte entfernen (Datei/Symlink/Ordner)
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink(missing_ok=True)

        # WICHTIG: absoluter -o Pfad + check=True + definiertes cwd
        subprocess.run(
            ["zokrates", "compile", "-i", str(circuit_file), "-o", str(out_path)],
            cwd=str(self.project_root),
            check=True,
        )
        return out_path

    def setup_circuit(self, circuit_name: str) -> Path:
        build_dir = self.project_root / "circuits" / "build_standard"
        out_path = build_dir / f"{circuit_name}.out"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing compiled circuit: {out_path}")

        # keys landen im Build-Dir (kein clutter im Source-Tree)
        subprocess.run(
            ["zokrates", "setup", "-i", str(out_path)],
            cwd=str(build_dir),
            check=True,
        )
        return out_path
