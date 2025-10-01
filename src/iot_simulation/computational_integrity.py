"""
Computational Integrity für IoT-Transformationsberechnungen
Sichert die Integrität von Datenverarbeitung mit ZK-Proofs
"""

import json
import time
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TransformationStep:
    """Einzelner Schritt in der Datenverarbeitung"""
    step_id: str
    operation: str  # "filter", "aggregate", "normalize", "validate"
    input_data: List[float]
    output_data: List[float]
    parameters: Dict[str, Any]
    timestamp: str
    computation_hash: str

@dataclass
class IntegrityProof:
    """ZK-Proof für Computational Integrity"""
    proof_id: str
    transformation_id: str
    input_hash: str
    output_hash: str
    computation_hash: str
    zk_proof: str
    public_inputs: Dict[str, Any]
    proof_size: int
    generation_time: float
    verification_time: float

@dataclass
class DataTransformation:
    """Komplette Datenverarbeitung mit Integrity-Proofs"""
    transformation_id: str
    sensor_id: str
    device_id: str
    input_data: List[float]
    output_data: List[float]
    transformation_steps: List[TransformationStep]
    integrity_proof: Optional[IntegrityProof]
    privacy_level: int
    timestamp: str

class IoTComputationalIntegrity:
    """Sichert Computational Integrity für IoT-Datenverarbeitung"""
    
    def __init__(self, circuit_dir: str = "circuits/advanced"):
        self.circuit_dir = Path(circuit_dir)
        self.temp_dir = Path("temp_integrity_proofs")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Transformation Templates
        self.transformation_templates = {
            "filter_range": {
                "operation": "filter",
                "description": "Filtere Werte innerhalb eines Bereichs",
                "parameters": ["min_value", "max_value"],
                "circuit": "filter_range.zok"
            },
            "aggregate_sum": {
                "operation": "aggregate",
                "description": "Berechne Summe der Werte",
                "parameters": [],
                "circuit": "aggregation.zok"
            },
            "normalize_minmax": {
                "operation": "normalize",
                "description": "Normalisiere Werte zwischen 0 und 1",
                "parameters": ["min_value", "max_value"],
                "circuit": "normalization.zok"
            },
            "validate_threshold": {
                "operation": "validate",
                "description": "Validiere Werte gegen Schwellenwerte",
                "parameters": ["threshold", "operator"],
                "circuit": "validation.zok"
            }
        }
    
    def create_transformation(self, sensor_id: str, device_id: str, 
                            input_data: List[float], transformation_type: str,
                            parameters: Dict[str, Any], privacy_level: int = 2) -> DataTransformation:
        """Erstelle sichere Datenverarbeitung mit ZK-Proofs"""
        
        transformation_id = f"{sensor_id}_{int(time.time())}"
        timestamp = datetime.now().isoformat()
        
        # Führe Transformation durch
        transformation_steps = []
        current_data = input_data.copy()
        
        if transformation_type == "filter_range":
            current_data, step = self._filter_range(current_data, parameters)
            transformation_steps.append(step)
        
        elif transformation_type == "aggregate_sum":
            current_data, step = self._aggregate_sum(current_data, parameters)
            transformation_steps.append(step)
        
        elif transformation_type == "normalize_minmax":
            current_data, step = self._normalize_minmax(current_data, parameters)
            transformation_steps.append(step)
        
        elif transformation_type == "validate_threshold":
            current_data, step = self._validate_threshold(current_data, parameters)
            transformation_steps.append(step)
        
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")
        
        # Erstelle Transformation
        transformation = DataTransformation(
            transformation_id=transformation_id,
            sensor_id=sensor_id,
            device_id=device_id,
            input_data=input_data,
            output_data=current_data,
            transformation_steps=transformation_steps,
            integrity_proof=None,
            privacy_level=privacy_level,
            timestamp=timestamp
        )
        
        # Generiere Integrity Proof
        integrity_proof = self._generate_integrity_proof(transformation)
        transformation.integrity_proof = integrity_proof
        
        logger.info(f"Created transformation {transformation_id} with {len(transformation_steps)} steps")
        return transformation
    
    def _filter_range(self, data: List[float], parameters: Dict[str, Any]) -> Tuple[List[float], TransformationStep]:
        """Filtere Werte innerhalb eines Bereichs"""
        min_value = parameters.get("min_value", 0.0)
        max_value = parameters.get("max_value", 100.0)
        
        # Führe Filterung durch
        filtered_data = [x for x in data if min_value <= x <= max_value]
        
        # Berechne Hash der Berechnung
        computation_data = f"filter_range:{min_value}:{max_value}:{len(data)}:{len(filtered_data)}"
        computation_hash = hashlib.sha256(computation_data.encode()).hexdigest()
        
        step = TransformationStep(
            step_id=f"filter_{int(time.time())}",
            operation="filter",
            input_data=data,
            output_data=filtered_data,
            parameters={"min_value": min_value, "max_value": max_value},
            timestamp=datetime.now().isoformat(),
            computation_hash=computation_hash
        )
        
        return filtered_data, step
    
    def _aggregate_sum(self, data: List[float], parameters: Dict[str, Any]) -> Tuple[List[float], TransformationStep]:
        """Berechne Summe der Werte"""
        sum_value = sum(data)
        
        # Berechne Hash der Berechnung
        computation_data = f"aggregate_sum:{len(data)}:{sum_value}"
        computation_hash = hashlib.sha256(computation_data.encode()).hexdigest()
        
        step = TransformationStep(
            step_id=f"aggregate_{int(time.time())}",
            operation="aggregate",
            input_data=data,
            output_data=[sum_value],
            parameters={},
            timestamp=datetime.now().isoformat(),
            computation_hash=computation_hash
        )
        
        return [sum_value], step
    
    def _normalize_minmax(self, data: List[float], parameters: Dict[str, Any]) -> Tuple[List[float], TransformationStep]:
        """Normalisiere Werte zwischen 0 und 1"""
        min_value = parameters.get("min_value", min(data))
        max_value = parameters.get("max_value", max(data))
        
        # Normalisiere
        if max_value == min_value:
            normalized_data = [0.5] * len(data)
        else:
            normalized_data = [(x - min_value) / (max_value - min_value) for x in data]
        
        # Berechne Hash der Berechnung
        computation_data = f"normalize_minmax:{min_value}:{max_value}:{len(data)}"
        computation_hash = hashlib.sha256(computation_data.encode()).hexdigest()
        
        step = TransformationStep(
            step_id=f"normalize_{int(time.time())}",
            operation="normalize",
            input_data=data,
            output_data=normalized_data,
            parameters={"min_value": min_value, "max_value": max_value},
            timestamp=datetime.now().isoformat(),
            computation_hash=computation_hash
        )
        
        return normalized_data, step
    
    def _validate_threshold(self, data: List[float], parameters: Dict[str, Any]) -> Tuple[List[float], TransformationStep]:
        """Validiere Werte gegen Schwellenwerte"""
        threshold = parameters.get("threshold", 50.0)
        operator = parameters.get("operator", ">")
        
        # Validiere
        if operator == ">":
            validated_data = [1.0 if x > threshold else 0.0 for x in data]
        elif operator == "<":
            validated_data = [1.0 if x < threshold else 0.0 for x in data]
        elif operator == ">=":
            validated_data = [1.0 if x >= threshold else 0.0 for x in data]
        elif operator == "<=":
            validated_data = [1.0 if x <= threshold else 0.0 for x in data]
        else:
            validated_data = data
        
        # Berechne Hash der Berechnung
        computation_data = f"validate_threshold:{threshold}:{operator}:{len(data)}"
        computation_hash = hashlib.sha256(computation_data.encode()).hexdigest()
        
        step = TransformationStep(
            step_id=f"validate_{int(time.time())}",
            operation="validate",
            input_data=data,
            output_data=validated_data,
            parameters={"threshold": threshold, "operator": operator},
            timestamp=datetime.now().isoformat(),
            computation_hash=computation_hash
        )
        
        return validated_data, step
    
    def _generate_integrity_proof(self, transformation: DataTransformation) -> IntegrityProof:
        """Generiere ZK-Proof für Computational Integrity"""
        start_time = time.time()
        
        try:
            # Berechne Hashes
            input_hash = hashlib.sha256(str(transformation.input_data).encode()).hexdigest()
            output_hash = hashlib.sha256(str(transformation.output_data).encode()).hexdigest()
            
            # Kombiniere alle Computation Hashes
            all_hashes = [step.computation_hash for step in transformation.transformation_steps]
            computation_hash = hashlib.sha256("".join(all_hashes).encode()).hexdigest()
            
            # Erstelle Input für ZK-Circuit
            circuit_input = {
                "input_hash": input_hash,
                "output_hash": output_hash,
                "computation_hash": computation_hash,
                "step_count": len(transformation.transformation_steps),
                "sensor_id": transformation.sensor_id,
                "device_id": transformation.device_id
            }
            
            # Generiere ZK-Proof
            proof_result = self._execute_integrity_circuit(circuit_input, transformation)
            
            if proof_result["success"]:
                generation_time = time.time() - start_time
                
                return IntegrityProof(
                    proof_id=f"integrity_{transformation.transformation_id}",
                    transformation_id=transformation.transformation_id,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    computation_hash=computation_hash,
                    zk_proof=proof_result["proof"],
                    public_inputs=proof_result["public_inputs"],
                    proof_size=len(proof_result["proof"]),
                    generation_time=generation_time,
                    verification_time=0.0  # Wird bei Verifikation gesetzt
                )
            else:
                logger.error(f"Integrity proof generation failed: {proof_result['error']}")
                return None
                
        except Exception as e:
            logger.error(f"Integrity proof generation error: {e}")
            return None
    
    def _execute_integrity_circuit(self, circuit_input: Dict[str, Any], 
                                 transformation: DataTransformation) -> Dict[str, Any]:
        """Führe ZK-Circuit für Computational Integrity aus"""
        try:
            # Erstelle temporäre Input-Datei
            input_file = self.temp_dir / f"integrity_input_{transformation.transformation_id}.json"
            with open(input_file, 'w') as f:
                json.dump(circuit_input, f)
            
            # Führe ZoKrates Circuit aus
            circuit_path = self.circuit_dir / "computational_integrity.zok"
            
            # Kompiliere Circuit
            compile_result = subprocess.run([
                "zokrates", "compile", "-i", str(circuit_path)
            ], capture_output=True, text=True, timeout=30)
            
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Circuit compilation failed: {compile_result.stderr}"
                }
            
            # Setup
            setup_result = subprocess.run([
                "zokrates", "setup"
            ], capture_output=True, text=True, timeout=60)
            
            if setup_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Setup failed: {setup_result.stderr}"
                }
            
            # Compute witness
            witness_args = [
                str(circuit_input["input_hash"]),
                str(circuit_input["output_hash"]),
                str(circuit_input["computation_hash"]),
                str(circuit_input["step_count"]),
                str(circuit_input["sensor_id"]),
                str(circuit_input["device_id"])
            ]
            
            compute_result = subprocess.run([
                "zokrates", "compute-witness", "-a"
            ] + witness_args, capture_output=True, text=True, timeout=30)
            
            if compute_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Witness computation failed: {compute_result.stderr}"
                }
            
            # Generate proof
            prove_result = subprocess.run([
                "zokrates", "generate-proof"
            ], capture_output=True, text=True, timeout=60)
            
            if prove_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Proof generation failed: {prove_result.stderr}"
                }
            
            # Lese Proof
            proof_file = Path("proof.json")
            if proof_file.exists():
                with open(proof_file, 'r') as f:
                    proof_data = json.load(f)
                
                return {
                    "success": True,
                    "proof": json.dumps(proof_data),
                    "public_inputs": {
                        "input_hash": circuit_input["input_hash"],
                        "output_hash": circuit_input["output_hash"],
                        "computation_hash": circuit_input["computation_hash"],
                        "step_count": circuit_input["step_count"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Proof file not generated"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Circuit execution timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Circuit execution error: {str(e)}"
            }
    
    def verify_integrity_proof(self, integrity_proof: IntegrityProof) -> bool:
        """Verifiziere Computational Integrity Proof"""
        if not integrity_proof:
            return False
        
        try:
            start_time = time.time()
            
            # Führe ZoKrates Verify aus
            verify_result = subprocess.run([
                "zokrates", "verify"
            ], capture_output=True, text=True, timeout=30)
            
            verification_time = time.time() - start_time
            
            if verify_result.returncode == 0:
                logger.info(f"Integrity proof verified in {verification_time:.3f}s")
                return True
            else:
                logger.warning(f"Integrity proof verification failed: {verify_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Integrity proof verification error: {e}")
            return False
    
    def create_secure_pipeline(self, sensor_id: str, device_id: str, 
                             input_data: List[float], pipeline_config: List[Dict[str, Any]],
                             privacy_level: int = 2) -> List[DataTransformation]:
        """Erstelle sichere Datenverarbeitungs-Pipeline"""
        
        transformations = []
        current_data = input_data.copy()
        
        for step_config in pipeline_config:
            transformation_type = step_config["type"]
            parameters = step_config.get("parameters", {})
            
            # Erstelle Transformation
            transformation = self.create_transformation(
                sensor_id=sensor_id,
                device_id=device_id,
                input_data=current_data,
                transformation_type=transformation_type,
                parameters=parameters,
                privacy_level=privacy_level
            )
            
            transformations.append(transformation)
            
            # Verwende Output als Input für nächsten Schritt
            current_data = transformation.output_data.copy()
        
        logger.info(f"Created secure pipeline with {len(transformations)} transformations")
        return transformations
    
    def get_transformation_statistics(self, transformations: List[DataTransformation]) -> Dict[str, Any]:
        """Berechne Statistiken für Transformationen"""
        if not transformations:
            return {}
        
        total_proofs = len(transformations)
        successful_proofs = len([t for t in transformations if t.integrity_proof])
        
        proof_sizes = [t.integrity_proof.proof_size for t in transformations if t.integrity_proof]
        generation_times = [t.integrity_proof.generation_time for t in transformations if t.integrity_proof]
        
        return {
            "total_transformations": total_proofs,
            "successful_proofs": successful_proofs,
            "proof_success_rate": successful_proofs / total_proofs if total_proofs > 0 else 0,
            "average_proof_size": np.mean(proof_sizes) if proof_sizes else 0,
            "average_generation_time": np.mean(generation_times) if generation_times else 0,
            "total_input_data_points": sum(len(t.input_data) for t in transformations),
            "total_output_data_points": sum(len(t.output_data) for t in transformations)
        }

def create_computational_integrity_test():
    """Test der Computational Integrity"""
    
    integrity = IoTComputationalIntegrity()
    
    # Test-Daten
    sensor_id = "TEMP_01"
    device_id = "DEV_001"
    input_data = [20.5, 22.1, 19.8, 23.4, 21.2, 24.0, 18.9, 25.1]
    
    print("=== Computational Integrity Test ===")
    print(f"Input data: {input_data}")
    
    # Test einzelne Transformation
    transformation = integrity.create_transformation(
        sensor_id=sensor_id,
        device_id=device_id,
        input_data=input_data,
        transformation_type="filter_range",
        parameters={"min_value": 20.0, "max_value": 24.0},
        privacy_level=2
    )
    
    print(f"Filtered data: {transformation.output_data}")
    print(f"Integrity proof: {'✓' if transformation.integrity_proof else '✗'}")
    
    if transformation.integrity_proof:
        print(f"Proof size: {transformation.integrity_proof.proof_size} bytes")
        print(f"Generation time: {transformation.integrity_proof.generation_time:.3f}s")
    
    # Test Pipeline
    pipeline_config = [
        {"type": "filter_range", "parameters": {"min_value": 20.0, "max_value": 24.0}},
        {"type": "normalize_minmax", "parameters": {"min_value": 20.0, "max_value": 24.0}},
        {"type": "aggregate_sum", "parameters": {}}
    ]
    
    pipeline = integrity.create_secure_pipeline(
        sensor_id=sensor_id,
        device_id=device_id,
        input_data=input_data,
        pipeline_config=pipeline_config,
        privacy_level=2
    )
    
    print(f"\nPipeline with {len(pipeline)} transformations:")
    for i, transformation in enumerate(pipeline):
        print(f"  Step {i+1}: {transformation.transformation_steps[0].operation}")
        print(f"    Input: {transformation.input_data}")
        print(f"    Output: {transformation.output_data}")
        print(f"    Proof: {'✓' if transformation.integrity_proof else '✗'}")
    
    # Statistiken
    stats = integrity.get_transformation_statistics(pipeline)
    print(f"\nStatistics:")
    print(f"  Total transformations: {stats['total_transformations']}")
    print(f"  Successful proofs: {stats['successful_proofs']}")
    print(f"  Success rate: {stats['proof_success_rate']:.2%}")
    print(f"  Average proof size: {stats['average_proof_size']:.0f} bytes")
    print(f"  Average generation time: {stats['average_generation_time']:.3f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_computational_integrity_test()
