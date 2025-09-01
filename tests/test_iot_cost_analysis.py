#!/usr/bin/env python3
"""
💰 IOT DEVICE COST ANALYSIS
Simuliert realistische IoT-Device-Kosten: CPU, Memory, Energy, Network
"""

import sys
import time
import psutil
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

@dataclass
class IoTDeviceProfile:
    """IoT Device Hardware Profile"""
    name: str
    cpu_mhz: int           # CPU Frequenz in MHz
    ram_mb: int            # RAM in MB
    energy_budget_mj: float # Energie-Budget in Millijoule
    network_kbps: int      # Netzwerk-Bandbreite in kbps
    cost_per_mhz_ms: float # Kosten pro MHz*Millisekunde
    cost_per_mb_s: float   # Kosten pro MB*Sekunde RAM
    cost_per_mj: float     # Kosten pro Millijoule Energie
    cost_per_kb: float     # Kosten pro KB Netzwerk

@dataclass
class CostMetrics:
    """Kosten-Metriken für einen Test"""
    cpu_cost: float        # CPU-Kosten
    memory_cost: float     # Memory-Kosten  
    energy_cost: float     # Energie-Kosten
    network_cost: float    # Netzwerk-Kosten
    total_cost: float      # Gesamtkosten
    execution_time: float  # Ausführungszeit
    peak_memory_mb: float  # Peak Memory Usage
    proof_size_kb: float   # Proof-Größe
    feasible: bool         # Machbar auf diesem Device?

# IoT Device Profiles (basierend auf realen Geräten)
IOT_PROFILES = {
    "raspberry_pi_zero": IoTDeviceProfile(
        name="Raspberry Pi Zero",
        cpu_mhz=1000,
        ram_mb=512,
        energy_budget_mj=5000,  # ~5 Joule Budget
        network_kbps=100,
        cost_per_mhz_ms=0.001,
        cost_per_mb_s=0.002,
        cost_per_mj=0.01,
        cost_per_kb=0.0001
    ),
    "esp32": IoTDeviceProfile(
        name="ESP32 Microcontroller",
        cpu_mhz=240,
        ram_mb=4,  # 4MB PSRAM
        energy_budget_mj=1000,  # ~1 Joule Budget
        network_kbps=50,
        cost_per_mhz_ms=0.005,
        cost_per_mb_s=0.01,
        cost_per_mj=0.02,
        cost_per_kb=0.0002
    ),
    "arduino_nano": IoTDeviceProfile(
        name="Arduino Nano 33 IoT",
        cpu_mhz=48,
        ram_mb=0.032,  # 32KB RAM
        energy_budget_mj=500,   # ~0.5 Joule Budget
        network_kbps=20,
        cost_per_mhz_ms=0.01,
        cost_per_mb_s=0.05,
        cost_per_mj=0.05,
        cost_per_kb=0.0005
    ),
    "cloud_vm": IoTDeviceProfile(
        name="Cloud VM (Referenz)",
        cpu_mhz=3000,
        ram_mb=8192,
        energy_budget_mj=100000,  # Quasi unbegrenzt
        network_kbps=10000,
        cost_per_mhz_ms=0.0001,
        cost_per_mb_s=0.0001,
        cost_per_mj=0.001,
        cost_per_kb=0.00001
    )
}

def measure_system_resources():
    """Misst aktuelle System-Ressourcen"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_mb": psutil.virtual_memory().used / (1024*1024),
        "memory_percent": psutil.virtual_memory().percent
    }

def calculate_iot_costs(
    device_profile: IoTDeviceProfile,
    execution_time: float,
    peak_memory_mb: float,
    proof_size_kb: float
) -> CostMetrics:
    """Berechnet IoT-Device-Kosten"""
    
    # CPU-Kosten (basierend auf Zeit und Frequenz)
    cpu_cost = device_profile.cost_per_mhz_ms * device_profile.cpu_mhz * (execution_time * 1000)
    
    # Memory-Kosten (basierend auf Peak-Usage und Zeit)
    memory_cost = device_profile.cost_per_mb_s * peak_memory_mb * execution_time
    
    # Energie-Kosten (geschätzt basierend auf CPU-Last)
    estimated_energy_mj = (execution_time * 1000) * (device_profile.cpu_mhz / 1000) * 0.1  # Schätzung
    energy_cost = device_profile.cost_per_mj * estimated_energy_mj
    
    # Netzwerk-Kosten (Proof-Übertragung)
    network_cost = device_profile.cost_per_kb * proof_size_kb
    
    # Gesamtkosten
    total_cost = cpu_cost + memory_cost + energy_cost + network_cost
    
    # Machbarkeits-Check
    feasible = (
        peak_memory_mb <= device_profile.ram_mb and
        estimated_energy_mj <= device_profile.energy_budget_mj and
        proof_size_kb <= (device_profile.network_kbps * execution_time * 0.8)  # 80% Netzwerk-Auslastung
    )
    
    return CostMetrics(
        cpu_cost=cpu_cost,
        memory_cost=memory_cost,
        energy_cost=energy_cost,
        network_cost=network_cost,
        total_cost=total_cost,
        execution_time=execution_time,
        peak_memory_mb=peak_memory_mb,
        proof_size_kb=proof_size_kb,
        feasible=feasible
    )

def test_standard_snark_costs(num_items: int) -> Dict:
    """Testet Standard SNARK Kosten"""
    print(f"   📊 Standard SNARK: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        manager = SNARKManager()
        
        # Setup
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Generiere Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Messe Ressourcen
        start_resources = measure_system_resources()
        start_time = time.time()
        peak_memory = start_resources["memory_mb"]
        
        successful_proofs = 0
        total_proof_size = 0
        
        for reading in temp_readings:
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            # Messe Memory während Proof
            current_resources = measure_system_resources()
            peak_memory = max(peak_memory, current_resources["memory_mb"])
            
            result = manager.generate_proof("filter_range", inputs)
            if result.success:
                successful_proofs += 1
                total_proof_size += result.metrics.proof_size
        
        execution_time = time.time() - start_time
        avg_proof_size_kb = (total_proof_size / 1024) if successful_proofs > 0 else 0
        
        return {
            "success": True,
            "type": "standard",
            "num_items": num_items,
            "execution_time": execution_time,
            "peak_memory_mb": peak_memory - start_resources["memory_mb"],
            "proof_size_kb": avg_proof_size_kb * successful_proofs,  # Gesamte Proof-Größe
            "successful_proofs": successful_proofs
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_recursive_snark_costs(num_items: int) -> Dict:
    """Testet Recursive SNARK Kosten"""
    print(f"   🚀 Recursive SNARK: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup
        if not nova_manager.setup():
            return {"success": False, "error": "Setup failed"}
        
        # Generiere Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Bereite Batches vor
        batches = []
        for i in range(0, len(temp_readings), 3):
            batch_readings = temp_readings[i:i+3]
            while len(batch_readings) < 3:
                batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
            
            batch_dicts = [{'value': r.value} for r in batch_readings]
            batches.append(batch_dicts)
        
        # Messe Ressourcen
        start_resources = measure_system_resources()
        start_time = time.time()
        
        result = nova_manager.prove_recursive_batch(batches)
        
        execution_time = time.time() - start_time
        end_resources = measure_system_resources()
        
        if result.success:
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "execution_time": execution_time,
                "peak_memory_mb": max(start_resources["memory_mb"], end_resources["memory_mb"]) - start_resources["memory_mb"],
                "proof_size_kb": result.proof_size / 1024,
                "steps": len(batches)
            }
        else:
            return {"success": False, "error": "Recursive proof failed"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Hauptfunktion"""
    print("💰 IOT DEVICE COST ANALYSIS")
    print("Simuliert realistische IoT-Device-Kosten und Machbarkeit")
    print("=" * 80)
    
    # Test verschiedene Item-Anzahlen
    test_items = [20, 50, 100, 200]
    
    all_results = []
    
    for num_items in test_items:
        print(f"\n🔬 TESTE: {num_items} Items")
        print("-" * 40)
        
        # Standard SNARK Test
        std_result = test_standard_snark_costs(num_items)
        if std_result["success"]:
            all_results.append(std_result)
        
        # Recursive SNARK Test
        rec_result = test_recursive_snark_costs(num_items)
        if rec_result["success"]:
            all_results.append(rec_result)
        
        time.sleep(1)
    
    # Kosten-Analyse für alle IoT-Devices
    print("\n" + "=" * 80)
    print("💰 IOT DEVICE COST ANALYSIS")
    print("=" * 80)
    
    for device_name, profile in IOT_PROFILES.items():
        print(f"\n🔧 DEVICE: {profile.name}")
        print("-" * 60)
        
        print(f"{'Items':<6} {'Type':<10} {'Time(s)':<8} {'Memory(MB)':<12} {'Proof(KB)':<10} {'Cost($)':<8} {'Feasible':<10}")
        print("-" * 70)
        
        for result in all_results:
            if not result["success"]:
                continue
                
            costs = calculate_iot_costs(
                profile,
                result["execution_time"],
                result["peak_memory_mb"],
                result["proof_size_kb"]
            )
            
            feasible_str = "✅ YES" if costs.feasible else "❌ NO"
            
            print(f"{result['num_items']:<6} {result['type']:<10} {costs.execution_time:<8.2f} "
                  f"{costs.peak_memory_mb:<12.2f} {costs.proof_size_kb:<10.1f} "
                  f"{costs.total_cost:<8.4f} {feasible_str:<10}")
    
    # Speichere Ergebnisse
    results_dir = project_root / "data" / "iot_cost_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "iot_cost_analysis.json"
    with open(results_file, 'w') as f:
        json.dump({
            "device_profiles": {k: {
                "name": v.name,
                "cpu_mhz": v.cpu_mhz,
                "ram_mb": v.ram_mb,
                "energy_budget_mj": v.energy_budget_mj
            } for k, v in IOT_PROFILES.items()},
            "test_results": all_results,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n💾 Ergebnisse gespeichert: {results_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 IOT COST ANALYSIS ABGESCHLOSSEN!' if success else '❌ IOT COST ANALYSIS FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
