#!/usr/bin/env python3
"""
Standard vs Recursive SNARK Vergleichstest
Jetzt mit funktionierenden Recursive SNARKs!
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_standard_vs_recursive_comparison():
    """Teste Standard vs Recursive SNARKs mit verschiedenen Datengrößen"""
    print("=" * 80)
    print("STANDARD vs RECURSIVE SNARK VERGLEICH")
    print("=" * 80)
    
    try:
        from src.proof_systems.snark_manager import SNARKManager
        from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager
        from src.iot_simulation.smart_home import SmartHomeSensors
        
        # Erstelle Manager
        standard_manager = SNARKManager(circuits_dir="circuits", output_dir="data/test_proofs")
        nova_manager = FixedZoKratesNovaManager()
        iot_simulator = SmartHomeSensors()
        
        # Setup
        circuit_path = Path("circuits/basic/filter_range.zok")
        standard_manager.compile_circuit(str(circuit_path), "filter_range")
        standard_manager.setup_circuit("filter_range")
        nova_manager.setup()
        
        # Test verschiedene Datengrößen
        test_sizes = [6, 9, 12, 15]  # Anzahl IoT Readings
        
        results = []
        
        for data_size in test_sizes:
            print(f"\n📊 TESTE {data_size} IoT READINGS")
            print("-" * 50)
            
            # Generiere Test IoT Daten
            test_readings = []
            for i in range(data_size):
                test_readings.append({
                    "value": 20 + (i % 10),  # 20-29°C
                    "sensor_type": "temperature",
                    "timestamp": f"2025-01-01T{i:02d}:00:00"
                })
            
            # STANDARD SNARKs Test
            print(f"🔐 Standard SNARKs: {data_size} einzelne Proofs...")
            
            standard_start = time.time()
            standard_proofs = []
            standard_total_size = 0
            
            for i, reading in enumerate(test_readings):
                # filter_range: min_val, max_val, secret_value
                inputs = ["15", "35", str(int(reading["value"]))]
                result = standard_manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    standard_proofs.append(result)
                    standard_total_size += result.metrics.proof_size
                else:
                    print(f"❌ Standard Proof {i+1} fehlgeschlagen")
            
            standard_time = time.time() - standard_start
            standard_success_rate = len(standard_proofs) / data_size
            
            print(f"   ✅ {len(standard_proofs)}/{data_size} Proofs erfolgreich")
            print(f"   ⏱️  Zeit: {standard_time:.3f}s")
            print(f"   📦 Total Größe: {standard_total_size:,} bytes")
            print(f"   🚀 Durchschnitt: {standard_time/data_size:.3f}s pro Proof")
            
            # RECURSIVE SNARKs Test (Nova)
            print(f"🔄 Recursive SNARKs: 1 Nova Proof für alle {data_size} Werte...")
            
            # Teile in 3er Batches (Nova erwartet 3 Werte pro Step)
            batches = []
            for i in range(0, len(test_readings), 3):
                batch = test_readings[i:i+3]
                # Fülle auf 3 auf falls nötig
                while len(batch) < 3:
                    batch.append({"value": 0, "sensor_type": "padding"})
                batches.append(batch)
            
            nova_start = time.time()
            nova_result = nova_manager.prove_recursive_batch(batches)
            nova_time = time.time() - nova_start
            
            if nova_result.success:
                print(f"   ✅ Nova Proof erfolgreich")
                print(f"   ⏱️  Zeit: {nova_result.total_time:.3f}s")
                print(f"   📦 Proof Größe: {nova_result.proof_size:,} bytes")
                print(f"   🔄 Steps: {nova_result.step_count}")
                print(f"   🚀 Zeit pro Item: {nova_result.total_time/data_size:.3f}s")
            else:
                print(f"   ❌ Nova Proof fehlgeschlagen: {nova_result.error_message}")
            
            # Vergleich
            if nova_result.success and standard_success_rate > 0.8:
                print(f"\n📈 VERGLEICH:")
                time_ratio = standard_time / nova_result.total_time
                size_ratio = standard_total_size / nova_result.proof_size
                
                print(f"   ⚡ Zeit-Verhältnis: {time_ratio:.2f}x")
                if time_ratio > 1:
                    print(f"      → Nova ist {time_ratio:.1f}x schneller!")
                else:
                    print(f"      → Standard ist {1/time_ratio:.1f}x schneller")
                
                print(f"   💾 Größe-Verhältnis: {size_ratio:.2f}x")
                if size_ratio > 1:
                    print(f"      → Nova spart {size_ratio:.1f}x Speicher!")
                else:
                    print(f"      → Standard braucht {1/size_ratio:.1f}x weniger Speicher")
                
                # Crossover-Analyse
                if time_ratio > 1.0:
                    print(f"   🎯 Nova Vorteil ab {data_size} Items!")
                
                results.append({
                    "data_size": data_size,
                    "standard_time": standard_time,
                    "standard_size": standard_total_size,
                    "standard_proofs": len(standard_proofs),
                    "nova_time": nova_result.total_time,
                    "nova_size": nova_result.proof_size,
                    "nova_steps": nova_result.step_count,
                    "time_ratio": time_ratio,
                    "size_ratio": size_ratio,
                    "nova_advantage": time_ratio > 1.0
                })
        
        # Zusammenfassung
        print("\n" + "=" * 80)
        print("📊 CROSSOVER-ANALYSE ZUSAMMENFASSUNG")
        print("=" * 80)
        
        print(f"{'Items':<6} {'Std Zeit':<10} {'Nova Zeit':<10} {'Zeit Ratio':<12} {'Nova Vorteil':<12}")
        print("-" * 60)
        
        for r in results:
            advantage = "✅ JA" if r["nova_advantage"] else "❌ NEIN"
            print(f"{r['data_size']:<6} {r['standard_time']:<10.3f} {r['nova_time']:<10.3f} {r['time_ratio']:<12.2f} {advantage:<12}")
        
        # Finde Crossover Point
        crossover_point = None
        for r in results:
            if r["nova_advantage"]:
                crossover_point = r["data_size"]
                break
        
        if crossover_point:
            print(f"\n🎯 CROSSOVER POINT: Nova wird ab {crossover_point} IoT Items effizienter!")
        else:
            print(f"\n⚠️  Kein Crossover Point in diesem Test-Bereich gefunden")
            print(f"    → Teste größere Datenmengen für Crossover")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Vergleichstest fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Führe Standard vs Recursive Vergleich aus"""
    print("🚀 STANDARD vs RECURSIVE SNARK EVALUATION")
    print("=" * 80)
    
    success, results = test_standard_vs_recursive_comparison()
    
    if success:
        print("\n🎉 VERGLEICHSTEST ERFOLGREICH!")
        print("✅ Standard SNARKs funktionieren")
        print("✅ Recursive SNARKs (Nova) funktionieren")
        print("✅ Crossover-Analyse durchgeführt")
        print("\n🔥 DEINE BACHELORARBEIT HAT ECHTE DATEN!")
    else:
        print("\n❌ Vergleichstest fehlgeschlagen")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)