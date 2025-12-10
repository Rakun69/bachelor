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
    success: bool
    proof: Optional[Dict]
    metrics: ProofMetrics
    error_message: Optional[str] = None

class SNARKManager:   
    # Initialisiert ZoKrates-Manager
    def __init__(self, circuits_dir: str = "circuits",
                 output_dir: str = "data/proofs"):
        self.circuits_dir = Path(circuits_dir)
        self.output_dir = Path(output_dir)
        self.compiled_circuits = {}
        self.proof_cache = {}
        self.last_setup_seconds: float = 0.0
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._check_zokrates()
    
    
    # Prüft ob ZoKrates im System verfügbar
    def _check_zokrates(self):
        try:
            result = subprocess.run(['zokrates', '--version'], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"ZoKrates available: {result.stdout.strip()}")
            else:
                raise Exception("ZoKrates not found")
        except Exception as e:
            logger.error(f"ZoKrates not available: {e}")
            raise
    

    # Kompiliert ZoKrates-Circuit in .out-Datei
    def compile_circuit(self, circuit_path: str, circuit_name: str) -> bool:
        try:
            start_time = time.time()
            
            abs_circuit_path = os.path.abspath(circuit_path)
            
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            
            output_filename = f'{circuit_name}.out'
            cmd = ['zokrates', 'compile', '-i', abs_circuit_path, '-o', output_filename]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                logger.warning(f"Circuit compilation timed out for {circuit_name} after 120 seconds")
                result = type('MockResult', (), {'returncode': 1, 'stderr': 'Compilation timed out'})() 
            
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
    

    # Führt Setup aus und macht rdy für Witness- und Proof-Generierung
    def setup_circuit(self, circuit_name: str) -> bool:
        try:
            start_time = time.time()
            
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            
            # setup
            cmd = ['zokrates', 'setup', '-i', f'{circuit_name}.out']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning(f"Circuit setup timed out for {circuit_name} after 60 seconds")
                result = type('MockResult', (), {'returncode': 1, 'stderr': 'Setup timed out'})()
            
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
    

    # Berechnet Witness für Circuit
    def generate_witness(self, circuit_name: str, inputs: List[str]) -> Tuple[bool, float]:
        original_dir = os.getcwd()
        try:
            start_time = time.time()
            
            os.chdir(self.output_dir)
            
            # witness
            cmd = ['zokrates', 'compute-witness', '-i', f'{circuit_name}.out', '-a'] + inputs
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning(f"Witness generation timed out for {circuit_name} after 30 seconds")
                result = type('MockResult', (), {'returncode': 1, 'stderr': 'Witness generation timed out'})()
            
            witness_time = max(0.0, time.time() - start_time)
            
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


    # Standard-Proof-Workflow (Witness, Proof-Generierung und Verifikation)
    def generate_proof(self, circuit_name: str, inputs: List[str]) -> CircuitResult:
        original_dir = os.getcwd()
        proof_time = 0.0
        verify_time = 0.0
        try:
            # Witness generieren und messen
            witness_start = time.time()
            witness_success, witness_time = self.generate_witness(circuit_name, inputs)
            witness_end = time.time()
            logger.info(f"[snark_manager] Witness generation took {witness_end - witness_start:.4f}s for circuit {circuit_name}")

            if not witness_success:
                return CircuitResult(
                    success=False,
                    proof=None,
                    metrics=ProofMetrics(0, 0, witness_time, 0, 0, 0, 0, 0),
                    error_message="Witness generation failed"
                )

            os.chdir(self.output_dir)

            circuit_file = f"{circuit_name}.out"
            if os.path.exists(circuit_file):
                shutil.copy(circuit_file, "out")

            proof_start = time.time()
            cmd = ['zokrates', 'generate-proof']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                proof_time = max(0.0, time.time() - proof_start)
            except subprocess.TimeoutExpired:
                logger.warning(f"Proof generation timed out for {circuit_name} after 60 seconds")
                proof_time = 60.0
                result = type('MockResult', (), {'returncode': 1, 'stderr': 'Proof generation timed out'})()

            logger.info(f"[snark_manager] Single proof generation for circuit {circuit_name} with inputs {inputs[:3]}... took {proof_time:.4f}s")

            if result.returncode == 0:
                with open('proof.json', 'r') as f:
                    proof = json.load(f)

                proof_size = len(json.dumps(proof).encode('utf-8'))

                verify_start = time.time()
                verify_cmd = ['zokrates', 'verify']
                try:
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
                    verify_time = max(0.0, time.time() - verify_start)
                    verify_success = verify_result.returncode == 0
                except subprocess.TimeoutExpired:
                    logger.warning(f"Verification timed out for {circuit_name} after 30 seconds")
                    verify_time = 30.0
                    verify_result = type('MockResult', (), {'returncode': 1, 'stderr': 'Verification timed out'})()
                    verify_success = False

                logger.info(f"[snark_manager] Single verify phase for circuit {circuit_name} took {verify_time:.4f}s")
                
                if not verify_success:
                    logger.error(f"Verification failed for {circuit_name}: {verify_result.stderr}")
                    return CircuitResult(
                        success=False,
                        proof=proof,
                        metrics=ProofMetrics(0, 0, witness_time, proof_time, verify_time, proof_size, 0, 0),
                        error_message=f"Verification failed: {verify_result.stderr}"
                    )

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

                logger.info(f"[snark_manager] Proof generated for {circuit_name}: proof_time={proof_time:.4f}s, verify_time={verify_time:.4f}s, size={proof_size} bytes")

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
        return 1000
    

    # Ermittelt aktuellen Speicherverbrauch
    def _get_memory_usage(self) -> float:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    

    # Standard-Proof erzeugen
    def prove_circuit(self, circuit_name: str, inputs: List[str]) -> CircuitResult:
        return self.generate_proof(circuit_name, inputs)


# Main
def main():
    manager = SNARKManager()
    
    circuits_to_compile = [
        ("basic/filter_range.zok", "filter_range"),
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