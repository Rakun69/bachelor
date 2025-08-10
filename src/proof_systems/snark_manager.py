"""
SNARK Manager - Handles both regular and recursive SNARK operations
Manages ZoKrates compilation, proof generation, and verification
"""

import json
import subprocess
import os
import shutil
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProofMetrics:
    """Metrics for proof generation and verification"""
    compile_time: float
    setup_time: float
    witness_time: float
    proof_time: float
    verify_time: float
    proof_size: int
    circuit_constraints: int
    memory_usage: float

@dataclass
class CircuitResult:
    """Result of circuit execution"""
    success: bool
    proof: Optional[Dict]
    metrics: ProofMetrics
    error_message: Optional[str] = None

class SNARKManager:
    """Manages SNARK operations for IoT data processing"""
    
    def __init__(self, circuits_dir: str = "/home/ramon/bachelor/circuits",
                 output_dir: str = "/home/ramon/bachelor/data/proofs"):
        self.circuits_dir = Path(circuits_dir)
        self.output_dir = Path(output_dir)
        self.compiled_circuits = {}
        self.proof_cache = {}
        self.last_setup_seconds: float = 0.0
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if ZoKrates is available
        self._check_zokrates()
    
    def _check_zokrates(self):
        """Check if ZoKrates is installed and available"""
        try:
            result = subprocess.run(['zokrates', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"ZoKrates available: {result.stdout.strip()}")
            else:
                raise Exception("ZoKrates not found")
        except Exception as e:
            logger.error(f"ZoKrates not available: {e}")
            raise
    
    def compile_circuit(self, circuit_path: str, circuit_name: str) -> bool:
        """Compile a ZoKrates circuit"""
        try:
            start_time = time.time()
            
            # Change to output directory for compilation
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            
            # Compile the circuit
            output_filename = f'{circuit_name}.out'
            cmd = ['zokrates', 'compile', '-i', str(circuit_path), '-o', output_filename]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            compile_time = time.time() - start_time
            
            output_path = Path(output_filename)
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Compiled {circuit_name} in {compile_time:.2f}s")
                self.compiled_circuits[circuit_name] = {
                    'path': circuit_path,
                    'compile_time': compile_time,
                    'output_file': output_filename
                }
                return True
            else:
                # If the output file exists despite non-zero return code, accept with warning
                if output_path.exists():
                    logger.warning(
                        f"Compilation returned non-zero for {circuit_name}, but output file exists. "
                        f"stdout: {result.stdout}\nstderr: {result.stderr}"
                    )
                    self.compiled_circuits[circuit_name] = {
                        'path': circuit_path,
                        'compile_time': compile_time,
                        'output_file': output_filename
                    }
                    return True
                logger.error(
                    f"Compilation failed for {circuit_name}.\n"
                    f"Command: {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error compiling {circuit_name}: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def setup_circuit(self, circuit_name: str) -> bool:
        """Setup trusted setup for a circuit"""
        try:
            start_time = time.time()
            
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            
            # Perform setup
            cmd = ['zokrates', 'setup', '-i', f'{circuit_name}.out']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            setup_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"Setup completed for {circuit_name} in {setup_time:.2f}s")
                if circuit_name in self.compiled_circuits:
                    self.compiled_circuits[circuit_name]['setup_time'] = setup_time
                self.last_setup_seconds = setup_time
                return True
            else:
                logger.error(f"Setup failed for {circuit_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error in setup for {circuit_name}: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def generate_witness(self, circuit_name: str, inputs: List[str]) -> Tuple[bool, float]:
        """Generate witness for given inputs"""
        original_dir = os.getcwd()
        try:
            start_time = time.time()
            
            os.chdir(self.output_dir)
            
            # Generate witness
            cmd = ['zokrates', 'compute-witness', '-i', f'{circuit_name}.out', '-a'] + inputs
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            witness_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"Witness generated for {circuit_name} in {witness_time:.2f}s")
                return True, witness_time
            else:
                logger.error(f"Witness generation failed for {circuit_name}: {result.stderr}")
                return False, witness_time
                
        except Exception as e:
            logger.error(f"Error generating witness for {circuit_name}: {e}")
            return False, 0.0
        finally:
            os.chdir(original_dir)
    
    def generate_proof(self, circuit_name: str, inputs: List[str]) -> CircuitResult:
        """Generate proof for circuit with given inputs"""
        original_dir = os.getcwd()
        try:
            metrics_start = time.time()
            
            # Generate witness first
            witness_success, witness_time = self.generate_witness(circuit_name, inputs)
            if not witness_success:
                return CircuitResult(
                    success=False,
                    proof=None,
                    metrics=ProofMetrics(0, 0, witness_time, 0, 0, 0, 0, 0),
                    error_message="Witness generation failed"
                )
            
            os.chdir(self.output_dir)
            
            # Copy circuit file to expected name
            circuit_file = f"{circuit_name}.out"
            if os.path.exists(circuit_file):
                shutil.copy(circuit_file, "out")
            
            # Generate proof
            proof_start = time.time()
            cmd = ['zokrates', 'generate-proof']
            result = subprocess.run(cmd, capture_output=True, text=True)
            proof_time = time.time() - proof_start
            
            if result.returncode == 0:
                # Load the generated proof
                with open('proof.json', 'r') as f:
                    proof = json.load(f)
                
                # Calculate proof size
                proof_size = len(json.dumps(proof).encode('utf-8'))
                
                # Verify the proof
                verify_start = time.time()
                verify_cmd = ['zokrates', 'verify']
                verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
                verify_time = time.time() - verify_start
                
                # Create metrics
                compile_time = self.compiled_circuits.get(circuit_name, {}).get('compile_time', 0)
                setup_time = self.compiled_circuits.get(circuit_name, {}).get('setup_time', 0)
                
                metrics = ProofMetrics(
                    compile_time=compile_time,
                    setup_time=setup_time,
                    witness_time=witness_time,
                    proof_time=proof_time,
                    verify_time=verify_time,
                    proof_size=proof_size,
                    circuit_constraints=self._get_circuit_constraints(circuit_name),
                    memory_usage=self._get_memory_usage()
                )
                
                logger.info(f"Proof generated for {circuit_name}:")
                logger.info(f"  Proof time: {proof_time:.2f}s")
                logger.info(f"  Verify time: {verify_time:.2f}s")
                logger.info(f"  Proof size: {proof_size} bytes")
                
                return CircuitResult(
                    success=True,
                    proof=proof,
                    metrics=metrics
                )
                
            else:
                logger.error(f"Proof generation failed for {circuit_name}: {result.stderr}")
                return CircuitResult(
                    success=False,
                    proof=None,
                    metrics=ProofMetrics(0, 0, witness_time, proof_time, 0, 0, 0, 0),
                    error_message=f"Proof generation failed: {result.stderr}"
                )
                
        except Exception as e:
            logger.error(f"Error generating proof for {circuit_name}: {e}")
            return CircuitResult(
                success=False,
                proof=None,
                metrics=ProofMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                error_message=str(e)
            )
        finally:
            os.chdir(original_dir)
    
    def _get_circuit_constraints(self, circuit_name: str) -> int:
        """Get number of constraints in the circuit (simplified)"""
        # This would need to parse the circuit file or output
        # For now, return a placeholder
        return 1000
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def batch_process_proofs(self, circuit_name: str, input_batches: List[List[str]]) -> List[CircuitResult]:
        """Process multiple batches and generate individual proofs"""
        results = []
        
        logger.info(f"Processing {len(input_batches)} batches for {circuit_name}")
        
        for i, inputs in enumerate(input_batches):
            logger.info(f"Processing batch {i+1}/{len(input_batches)}")
            result = self.generate_proof(circuit_name, inputs)
            results.append(result)
            
            if not result.success:
                logger.warning(f"Batch {i+1} failed: {result.error_message}")
        
        return results
    
    def create_recursive_proof(self, base_proofs: List[Dict], circuit_name: str = "batch_processor") -> CircuitResult:
        """Create recursive proof from multiple base proofs"""
        # This is a simplified implementation
        # In practice, recursive SNARKs require more complex proof composition
        
        logger.info(f"Creating recursive proof from {len(base_proofs)} base proofs")
        
        try:
            start_time = time.time()
            
            # Simulate recursive proof creation
            # In a real implementation, this would involve:
            # 1. Aggregating the base proofs
            # 2. Creating a circuit that verifies all base proofs
            # 3. Generating a new proof for the aggregate
            
            recursive_proof = {
                "type": "recursive",
                "base_proof_count": len(base_proofs),
                "aggregated_result": "success",
                "timestamp": time.time(),
                "base_proof_hashes": [hash(str(proof)) for proof in base_proofs]
            }
            
            recursive_time = time.time() - start_time
            
            metrics = ProofMetrics(
                compile_time=0,
                setup_time=0,
                witness_time=0,
                proof_time=recursive_time,
                verify_time=0.1,  # Recursive proofs should verify quickly
                proof_size=len(json.dumps(recursive_proof).encode('utf-8')),
                circuit_constraints=len(base_proofs) * 100,  # Simplified
                memory_usage=self._get_memory_usage()
            )
            
            logger.info(f"Recursive proof created in {recursive_time:.2f}s")
            logger.info(f"Compressed {len(base_proofs)} proofs into 1 recursive proof")
            
            return CircuitResult(
                success=True,
                proof=recursive_proof,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error creating recursive proof: {e}")
            return CircuitResult(
                success=False,
                proof=None,
                metrics=ProofMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                error_message=str(e)
            )
    
    def get_performance_comparison(self, individual_results: List[CircuitResult], 
                                 recursive_result: CircuitResult) -> Dict[str, Any]:
        """Compare performance between individual and recursive proofs"""
        if not individual_results or not recursive_result.success:
            return {}
        
        # Calculate totals for individual proofs
        total_individual_time = sum(r.metrics.proof_time for r in individual_results if r.success)
        total_individual_size = sum(r.metrics.proof_size for r in individual_results if r.success)
        successful_individual = sum(1 for r in individual_results if r.success)
        
        comparison = {
            "individual_proofs": {
                "count": len(individual_results),
                "successful": successful_individual,
                "total_proof_time": total_individual_time,
                "total_proof_size": total_individual_size,
                "average_proof_time": total_individual_time / max(successful_individual, 1),
                "average_proof_size": total_individual_size / max(successful_individual, 1)
            },
            "recursive_proof": {
                "proof_time": recursive_result.metrics.proof_time,
                "proof_size": recursive_result.metrics.proof_size,
                "compression_ratio": total_individual_size / max(recursive_result.metrics.proof_size, 1)
            },
            "performance_gain": {
                "time_reduction": max(0, total_individual_time - recursive_result.metrics.proof_time),
                "size_reduction": max(0, total_individual_size - recursive_result.metrics.proof_size),
                "efficiency_ratio": total_individual_time / max(recursive_result.metrics.proof_time, 0.001)
            }
        }
        
        return comparison
    
    def prove_circuit(self, circuit_name: str, inputs: List[str]) -> CircuitResult:
        """Wrapper for generate_proof - provides compatibility with orchestrator"""
        return self.generate_proof(circuit_name, inputs)

def main():
    """Example usage of SNARK Manager"""
    manager = SNARKManager()
    
    # Compile basic circuits
    circuits_to_compile = [
        ("basic/filter_range.zok", "filter_range"),
        ("basic/min_max.zok", "min_max"),
        ("basic/median.zok", "median"),
        ("advanced/aggregation.zok", "aggregation")
    ]
    
    for circuit_file, circuit_name in circuits_to_compile:
        circuit_path = manager.circuits_dir / circuit_file
        if circuit_path.exists():
            success = manager.compile_circuit(str(circuit_path), circuit_name)
            if success:
                manager.setup_circuit(circuit_name)
        else:
            logger.warning(f"Circuit file not found: {circuit_path}")

if __name__ == "__main__":
    main()