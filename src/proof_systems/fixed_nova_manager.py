"""
Fixed ZoKrates Nova Manager - Korrigierte Version
"""

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
    """Result of Nova recursive proof generation"""
    success: bool
    proof_data: Optional[str] = None
    compressed_proof: Optional[str] = None
    step_count: int = 0
    total_time: float = 0.0
    verify_time: float = 0.0
    proof_size: int = 0
    error_message: Optional[str] = None

class FixedZoKratesNovaManager:
    """
    Korrigierte Version des ZoKrates Nova Managers
    Verwendet das korrekte Input-Format für Nova
    """
    
    def __init__(self, circuit_path: str = "circuits/nova/iot_recursive.zok", 
                 batch_size: int = 3):
        self.circuit_path = Path(circuit_path)
        self.batch_size = batch_size
        self.working_dir = Path("circuits/nova")  # Direkt das Nova Verzeichnis verwenden
        
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
        """Setup ist bereits gemacht - prüfe nur ob Dateien existieren"""
        try:
            if not self.working_dir.exists():
                logger.error(f"Nova working directory not found: {self.working_dir}")
                return False
            
            # Prüfe ob wichtige Dateien existieren
            required_files = ["out", "nova.params"]
            for file in required_files:
                if not (self.working_dir / file).exists():
                    logger.error(f"Required Nova file missing: {file}")
                    return False
            
            self.setup_done = True
            return True
            
        except Exception as e:
            logger.error(f"Nova setup check failed: {e}")
            return False
    
    def prepare_initial_state(self) -> Dict[str, str]:
        """
        Create initial state for Nova recursion
        State { field sum, field count }
        """
        return {
            "sum": "0",
            "count": "0"
        }
    
    def prepare_step_input(self, iot_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert IoT data to Nova step input format
        StepInput { field[3] values, field batch_id }
        """
        values = []
        
        # Take up to 3 values, pad with zeros if needed
        for i in range(3):
            if i < len(iot_data) and iot_data[i]:
                # Convert value to field element (scale and round)
                value = int(float(iot_data[i].get('value', 0)) * 10)  # 1 decimal precision
                values.append(str(value))
            else:
                # Padding with zeros
                values.append("0")
        
        return {
            "values": values,
            "batch_id": str(int(time.time()) % 1000)  # Kleinere batch_id
        }
    
    def _execute_nova_proof(self, initial_state: Dict[str, str], 
                           steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute Nova proof using correct format
        """
        try:
            original_cwd = os.getcwd()
            os.chdir(self.working_dir)
            
            # Write initial state (State struct)
            with open("init.json", "w") as f:
                json.dump(initial_state, f)
            
            # Write steps (Array of StepInput structs)
            with open("steps.json", "w") as f:
                json.dump(steps, f)
            
            logger.info(f"Nova inputs prepared: init={initial_state}, steps={len(steps)}")
            
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
        Generate Nova recursive proof for IoT batches
        """
        if not self.setup_done:
            if not self.setup():
                return NovaProofResult(
                    success=False,
                    error_message="Nova setup failed"
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
                    error_message=result["error"]
                )
                
        except Exception as e:
            logger.error(f"Nova proof generation failed: {e}")
            return NovaProofResult(
                success=False,
                error_message=str(e)
            )
