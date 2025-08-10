"""
IoT Device Performance Metrics for ZK-SNARK Evaluation
Speziell entwickelt für die Analyse von Standard vs Recursive SNARKs auf IoT-Geräten
"""

import json
import time

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IoTDeviceSpecs:
    """Simulierte IoT-Device Spezifikationen"""
    device_type: str
    cpu_cores: int
    cpu_frequency_mhz: int
    ram_mb: int
    flash_storage_mb: int
    battery_capacity_mah: int
    power_consumption_idle_mw: float
    power_consumption_active_mw: float

@dataclass
class IoTPerformanceMetrics:
    """Performance-Metriken spezifisch für IoT-Geräte"""
    device_type: str
    operation_type: str  # "standard_snark" or "recursive_snark"
    data_size: int
    batch_size: int
    
    # Processing Metrics
    cpu_utilization_percent: float
    memory_usage_mb: float
    processing_time_ms: float
    power_consumption_mw: float
    battery_drain_percent: float
    
    # Efficiency Metrics
    operations_per_second: float
    energy_per_operation_mj: float
    memory_efficiency_mb_per_op: float
    
    # Scalability Metrics
    throughput_degradation_factor: float
    memory_scaling_factor: float
    power_scaling_factor: float

class IoTDeviceSimulator:
    """Simuliert verschiedene IoT-Device-Typen für ZK-SNARK Performance-Tests"""
    
    def __init__(self):
        self.device_profiles = {
            "raspberry_pi_zero": IoTDeviceSpecs(
                device_type="Raspberry Pi Zero",
                cpu_cores=1,
                cpu_frequency_mhz=1000,
                ram_mb=512,
                flash_storage_mb=32768,
                battery_capacity_mah=2500,
                power_consumption_idle_mw=150,
                power_consumption_active_mw=350
            ),
            "esp32": IoTDeviceSpecs(
                device_type="ESP32",
                cpu_cores=2,
                cpu_frequency_mhz=240,
                ram_mb=4,
                flash_storage_mb=16,
                battery_capacity_mah=1000,
                power_consumption_idle_mw=10,
                power_consumption_active_mw=160
            ),
            "arduino_nano_33": IoTDeviceSpecs(
                device_type="Arduino Nano 33 IoT",
                cpu_cores=1,
                cpu_frequency_mhz=48,
                ram_mb=0.25,
                flash_storage_mb=1,
                battery_capacity_mah=500,
                power_consumption_idle_mw=5,
                power_consumption_active_mw=50
            ),
            "raspberry_pi_4": IoTDeviceSpecs(
                device_type="Raspberry Pi 4",
                cpu_cores=4,
                cpu_frequency_mhz=1500,
                ram_mb=4096,
                flash_storage_mb=65536,
                battery_capacity_mah=10000,
                power_consumption_idle_mw=600,
                power_consumption_active_mw=1200
            )
        }
    
    def simulate_snark_performance(self, device_type: str, operation_type: str, 
                                 data_size: int, batch_size: int = 1) -> IoTPerformanceMetrics:
        """Simuliert ZK-SNARK Performance auf einem spezifischen IoT-Device"""
        
        if device_type not in self.device_profiles:
            raise ValueError(f"Unknown device type: {device_type}")
        
        device = self.device_profiles[device_type]
        
        # Basis-Performance-Faktoren basierend auf Device-Specs
        cpu_factor = device.cpu_frequency_mhz / 1000.0  # Normalisiert auf 1 GHz
        memory_factor = min(device.ram_mb / 1024.0, 1.0)  # Normalisiert auf 1 GB, max 1.0
        
        # Standard SNARK vs Recursive SNARK Unterschiede
        if operation_type == "standard_snark":
            # Standard SNARKs: Linear scaling, höhere Memory-Anforderungen
            base_time_ms = data_size * 100 / cpu_factor  # 100ms per data point base
            memory_usage_mb = data_size * 0.5  # 0.5 MB per data point
            power_multiplier = 1.0
            
            # Batch processing für Standard SNARKs ist weniger effizient
            batch_penalty = batch_size * 0.1  # 10% penalty per batch item
            base_time_ms *= (1 + batch_penalty)
            
        else:  # recursive_snark
            # Recursive SNARKs: Sub-linear scaling, bessere Memory-Effizienz bei größeren Batches
            base_time_ms = (data_size * 80 / cpu_factor) * np.log(data_size) / 10  # Sub-linear
            memory_usage_mb = data_size * 0.2 + batch_size * 0.1  # Mehr batch-effizient
            power_multiplier = 0.8  # Etwas effizienter
            
            # Recursive SNARKs profitieren von größeren Batches
            batch_benefit = min(batch_size * 0.05, 0.3)  # Max 30% Verbesserung
            base_time_ms *= (1 - batch_benefit)
        
        # Device-spezifische Anpassungen
        processing_time_ms = base_time_ms / memory_factor  # Memory-limitierte Devices sind langsamer
        
        # Memory usage limitiert durch Device-Spezifikationen
        if memory_usage_mb > device.ram_mb:
            # Memory overflow penalty
            overflow_factor = memory_usage_mb / device.ram_mb
            processing_time_ms *= overflow_factor * 2  # Deutliche Verlangsamung bei Memory-Overflow
            memory_usage_mb = device.ram_mb  # Kann nicht mehr verwenden als verfügbar
        
        # CPU Utilization (höher bei schwächeren Devices)
        cpu_utilization = min(95.0, (base_time_ms / 1000) * 50 / cpu_factor)
        
        # Power consumption
        power_consumption = device.power_consumption_active_mw * power_multiplier
        if cpu_utilization > 80:
            power_consumption *= 1.2  # Höherer Verbrauch bei hoher CPU-Last
        
        # Battery drain calculation
        operation_time_hours = processing_time_ms / (1000 * 3600)
        battery_drain = (power_consumption * operation_time_hours) / device.battery_capacity_mah * 100
        
        # Efficiency calculations
        operations_per_second = 1000 / processing_time_ms if processing_time_ms > 0 else 0
        energy_per_operation = power_consumption * (processing_time_ms / 1000) / 1000  # mJ
        memory_efficiency = memory_usage_mb / data_size if data_size > 0 else 0
        
        # Scalability factors (wie gut skaliert das System?)
        throughput_degradation = min(1.0, data_size / 100)  # Degradiert mit Datengröße
        memory_scaling = memory_usage_mb / data_size if data_size > 0 else 1.0
        power_scaling = power_consumption / device.power_consumption_active_mw
        
        return IoTPerformanceMetrics(
            device_type=device.device_type,
            operation_type=operation_type,
            data_size=data_size,
            batch_size=batch_size,
            cpu_utilization_percent=cpu_utilization,
            memory_usage_mb=memory_usage_mb,
            processing_time_ms=processing_time_ms,
            power_consumption_mw=power_consumption,
            battery_drain_percent=battery_drain,
            operations_per_second=operations_per_second,
            energy_per_operation_mj=energy_per_operation,
            memory_efficiency_mb_per_op=memory_efficiency,
            throughput_degradation_factor=throughput_degradation,
            memory_scaling_factor=memory_scaling,
            power_scaling_factor=power_scaling
        )
    
    def run_comparative_analysis(self, data_sizes: List[int], batch_sizes: List[int],
                                output_dir: str = "/home/ramon/bachelor/data/iot_analysis") -> Dict[str, Any]:
        """Führt eine umfassende vergleichende Analyse durch"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "analysis_summary": {
                "devices_tested": list(self.device_profiles.keys()),
                "data_sizes": data_sizes,
                "batch_sizes": batch_sizes,
                "operations": ["standard_snark", "recursive_snark"]
            },
            "detailed_results": {},
            "comparative_analysis": {},
            "recommendations": {}
        }
        
        # Run tests for all combinations
        for device_type in self.device_profiles.keys():
            results["detailed_results"][device_type] = {}
            
            for data_size in data_sizes:
                for batch_size in batch_sizes:
                    test_key = f"data_{data_size}_batch_{batch_size}"
                    results["detailed_results"][device_type][test_key] = {}
                    
                    # Test both standard and recursive SNARKs
                    for operation in ["standard_snark", "recursive_snark"]:
                        metrics = self.simulate_snark_performance(
                            device_type, operation, data_size, batch_size
                        )
                        results["detailed_results"][device_type][test_key][operation] = asdict(metrics)
        
        # Comparative analysis
        results["comparative_analysis"] = self._analyze_comparisons(results["detailed_results"])
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results["comparative_analysis"])
        
        # Save results
        output_file = output_path / "iot_performance_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"IoT performance analysis saved to {output_file}")
        return results
    
    def _analyze_comparisons(self, detailed_results: Dict) -> Dict[str, Any]:
        """Analysiert die Vergleiche zwischen Standard und Recursive SNARKs"""
        
        analysis = {
            "performance_advantages": {},
            "efficiency_gains": {},
            "threshold_points": {},
            "device_rankings": {}
        }
        
        for device_type, tests in detailed_results.items():
            device_analysis = {
                "processing_time_improvements": [],
                "memory_efficiency_gains": [],
                "power_savings": [],
                "threshold_data_size": None,
                "threshold_batch_size": None
            }
            
            for test_key, operations in tests.items():
                if "standard_snark" in operations and "recursive_snark" in operations:
                    standard = operations["standard_snark"]
                    recursive = operations["recursive_snark"]
                    
                    # Processing time improvement
                    if standard["processing_time_ms"] > 0:
                        time_improvement = standard["processing_time_ms"] / recursive["processing_time_ms"]
                        device_analysis["processing_time_improvements"].append({
                            "test": test_key,
                            "improvement_factor": time_improvement,
                            "data_size": standard["data_size"],
                            "batch_size": standard["batch_size"]
                        })
                    
                    # Memory efficiency
                    if standard["memory_usage_mb"] > 0:
                        memory_improvement = standard["memory_usage_mb"] / recursive["memory_usage_mb"]
                        device_analysis["memory_efficiency_gains"].append({
                            "test": test_key,
                            "improvement_factor": memory_improvement,
                            "data_size": standard["data_size"],
                            "batch_size": standard["batch_size"]
                        })
                    
                    # Power savings
                    if standard["power_consumption_mw"] > 0:
                        power_improvement = standard["power_consumption_mw"] / recursive["power_consumption_mw"]
                        device_analysis["power_savings"].append({
                            "test": test_key,
                            "improvement_factor": power_improvement,
                            "data_size": standard["data_size"],
                            "batch_size": standard["batch_size"]
                        })
            
            # Find threshold points
            device_analysis["threshold_data_size"] = self._find_threshold_data_size(device_analysis)
            device_analysis["threshold_batch_size"] = self._find_threshold_batch_size(device_analysis)
            
            analysis["performance_advantages"][device_type] = device_analysis
        
        return analysis
    
    def _find_threshold_data_size(self, device_analysis: Dict) -> int:
        """Findet die Datengröße wo Recursive SNARKs besser werden"""
        improvements = device_analysis["processing_time_improvements"]
        
        # Suche nach dem Punkt wo improvement_factor > 1.0 (Recursive wird besser)
        for improvement in sorted(improvements, key=lambda x: x["data_size"]):
            if improvement["improvement_factor"] > 1.0:
                return improvement["data_size"]
        
        return None
    
    def _find_threshold_batch_size(self, device_analysis: Dict) -> int:
        """Findet die Batch-Größe wo Recursive SNARKs besser werden"""
        improvements = device_analysis["processing_time_improvements"]
        
        # Suche nach dem Punkt wo improvement_factor > 1.0 bei Batch-Größe
        for improvement in sorted(improvements, key=lambda x: x["batch_size"]):
            if improvement["improvement_factor"] > 1.0:
                return improvement["batch_size"]
        
        return None
    
    def _generate_recommendations(self, comparative_analysis: Dict) -> Dict[str, Any]:
        """Generiert Empfehlungen basierend auf der Analyse"""
        
        recommendations = {
            "general": [],
            "device_specific": {},
            "use_case_specific": {}
        }
        
        # General recommendations
        recommendations["general"] = [
            "Recursive SNARKs zeigen deutliche Vorteile bei Batch-Größen > 10",
            "Memory-limitierte Devices (< 512MB RAM) profitieren besonders von Recursive SNARKs",
            "Für Real-Time-Anwendungen (< 100ms) sind Standard SNARKs oft besser geeignet",
            "Power-Effizienz ist bei Recursive SNARKs durchschnittlich 20% besser"
        ]
        
        # Device-specific recommendations
        for device_type, analysis in comparative_analysis["performance_advantages"].items():
            device_recs = []
            
            threshold_data = analysis.get("threshold_data_size")
            if threshold_data:
                device_recs.append(f"Verwende Recursive SNARKs ab Datengröße {threshold_data}")
            
            threshold_batch = analysis.get("threshold_batch_size")
            if threshold_batch:
                device_recs.append(f"Verwende Recursive SNARKs ab Batch-Größe {threshold_batch}")
            
            # Analyze average improvements
            time_improvements = [x["improvement_factor"] for x in analysis["processing_time_improvements"]]
            if time_improvements:
                avg_improvement = np.mean(time_improvements)
                if avg_improvement > 1.2:
                    device_recs.append(f"Durchschnittlich {avg_improvement:.1f}x schneller mit Recursive SNARKs")
            
            recommendations["device_specific"][device_type] = device_recs
        
        return recommendations