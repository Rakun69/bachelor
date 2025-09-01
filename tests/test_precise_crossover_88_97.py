#!/usr/bin/env python3
"""
🎯 PRECISE CROSSOVER ANALYSIS 88-97 ITEMS
Findet den EXAKTEN Crossover-Point mit korrigierten Proof-Größen
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def precise_standard_test(num_items: int) -> dict:
    """Präziser Standard SNARK Test mit korrekter Proof-Größen-Berechnung"""
    try:
        print(f"   📊 Standard ({num_items} items): ", end="", flush=True)
        
        sensors = SmartHomeSensors()
        manager = SNARKManager()
        
        # Setup
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Generiere Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Messe Zeit und sammle korrekte Metriken
        start_time = time.time()
        successful = 0
        individual_proof_sizes = []  # Sammle individuelle Größen
        total_verify_time = 0
        
        for reading in temp_readings:
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            result = manager.generate_proof("filter_range", inputs)
            if result.success:
                successful += 1
                individual_proof_sizes.append(result.metrics.proof_size)
                total_verify_time += result.metrics.verify_time
        
        total_time = time.time() - start_time
        
        # Korrekte Proof-Größen-Berechnung
        if individual_proof_sizes:
            avg_proof_size_bytes = sum(individual_proof_sizes) / len(individual_proof_sizes)
            total_proof_size_bytes = sum(individual_proof_sizes)  # Gesamtgröße aller Proofs
            total_proof_size_kb = total_proof_size_bytes / 1024
            avg_proof_size_kb = avg_proof_size_bytes / 1024
        else:
            total_proof_size_kb = 0
            avg_proof_size_kb = 0
        
        avg_verify_time = total_verify_time / successful if successful > 0 else 0
        throughput = successful / total_time if total_time > 0 else 0
        
        print(f"{total_time:.3f}s ({successful}/{num_items})")
        
        return {
            "success": True,
            "num_items": num_items,
            "total_time": total_time,
            "successful_proofs": successful,
            "total_proof_size_kb": total_proof_size_kb,
            "avg_proof_size_kb": avg_proof_size_kb,
            "avg_verify_time": avg_verify_time,
            "throughput": throughput,
            "individual_proof_sizes": individual_proof_sizes[:3]  # Erste 3 für Debugging
        }
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return {"success": False, "error": str(e)}

def precise_recursive_test(num_items: int) -> dict:
    """Präziser Recursive SNARK Test"""
    try:
        print(f"   🚀 Recursive ({num_items} items): ", end="", flush=True)
        
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup
        if not nova_manager.setup():
            print("❌ Setup failed")
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
        
        # Messe Zeit
        start_time = time.time()
        result = nova_manager.prove_recursive_batch(batches)
        total_time = time.time() - start_time
        
        if result.success:
            proof_size_kb = result.proof_size / 1024
            throughput = num_items / total_time if total_time > 0 else 0
            
            print(f"{total_time:.3f}s ({len(batches)} steps)")
            
            return {
                "success": True,
                "num_items": num_items,
                "total_time": total_time,
                "steps": len(batches),
                "proof_size_kb": proof_size_kb,
                "verify_time": result.verify_time,
                "throughput": throughput
            }
        else:
            print(f"❌ Failed: {result.error_message}")
            return {"success": False, "error": result.error_message}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def find_exact_crossover_88_97():
    """Findet den exakten Crossover-Point zwischen 88-97 Items"""
    print("🎯 PRECISE CROSSOVER ANALYSIS 88-97 ITEMS")
    print("Findet den EXAKTEN Crossover-Point mit korrigierten Proof-Größen")
    print("=" * 70)
    
    # Teste jeden einzelnen Wert von 88 bis 97
    test_points = list(range(88, 98))  # 88, 89, 90, ..., 97
    results = []
    
    for num_items in test_points:
        print(f"\n🔬 TESTE: {num_items} Items")
        print("-" * 30)
        
        # Standard Test
        std_result = precise_standard_test(num_items)
        
        # Recursive Test
        rec_result = precise_recursive_test(num_items)
        
        # Nur hinzufügen wenn beide erfolgreich
        if std_result["success"] and rec_result["success"]:
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if rec_result["total_time"] < std_result["total_time"] else "Standard"
            time_diff = abs(rec_result["total_time"] - std_result["total_time"])
            percentage_diff = (time_diff / min(std_result["total_time"], rec_result["total_time"])) * 100
            
            result = {
                "num_items": num_items,
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "ratio": ratio,
                "winner": winner,
                "recursive_wins": rec_result["total_time"] < std_result["total_time"],
                "time_difference": time_diff,
                "percentage_difference": percentage_diff,
                "standard_proof_size_kb": std_result["total_proof_size_kb"],
                "recursive_proof_size_kb": rec_result["proof_size_kb"],
                "standard_avg_proof_kb": std_result["avg_proof_size_kb"],
                "standard_throughput": std_result["throughput"],
                "recursive_throughput": rec_result["throughput"],
                "proof_size_ratio": std_result["total_proof_size_kb"] / max(rec_result["proof_size_kb"], 0.1)
            }
            results.append(result)
            
            print(f"   📊 {winner} gewinnt um {percentage_diff:.1f}% (Ratio: {ratio:.4f})")
            print(f"   📦 Proof-Größen: Std {std_result['total_proof_size_kb']:.1f}KB vs Rec {rec_result['proof_size_kb']:.1f}KB")
        else:
            print(f"   ❌ Test fehlgeschlagen - überspringe {num_items} Items")
        
        # Kurze Pause zwischen Tests
        time.sleep(1)
    
    return results

def analyze_precise_crossover(results):
    """Analysiert den präzisen Crossover-Point"""
    print("\n" + "=" * 80)
    print("🎯 PRÄZISE CROSSOVER-POINT ANALYSE")
    print("=" * 80)
    
    if not results:
        print("❌ Keine gültigen Ergebnisse")
        return None
    
    print(f"{'Items':<5} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Diff%':<7} {'Winner':<10} {'Std_KB':<8} {'Rec_KB':<8}")
    print("-" * 80)
    
    exact_crossover = None
    crossover_candidates = []
    
    for result in results:
        items = result["num_items"]
        std_time = result["standard_time"]
        rec_time = result["recursive_time"]
        ratio = result["ratio"]
        diff_pct = result["percentage_difference"]
        winner = result["winner"]
        std_kb = result["standard_proof_size_kb"]
        rec_kb = result["recursive_proof_size_kb"]
        
        # Markiere Crossover-Kandidaten (sehr nah beieinander)
        marker = ""
        if diff_pct < 2.0:  # Weniger als 2% Unterschied
            marker = " ⚡"
            crossover_candidates.append(result)
        
        print(f"{items:<5} {std_time:<8.3f} {rec_time:<8.3f} {ratio:<8.4f} {diff_pct:<7.1f} {winner:<10} {std_kb:<8.1f} {rec_kb:<8.1f}{marker}")
        
        # Finde ersten echten Crossover
        if result["recursive_wins"] and exact_crossover is None:
            exact_crossover = items
    
    # Detaillierte Crossover-Analyse
    print(f"\n🎯 CROSSOVER-ANALYSE:")
    
    if exact_crossover:
        print(f"   ✅ EXAKTER CROSSOVER: {exact_crossover} Items")
        print(f"   → Ab {exact_crossover} Items sind Recursive SNARKs schneller!")
        
        # Finde das Crossover-Ergebnis
        crossover_result = next(r for r in results if r["num_items"] == exact_crossover)
        print(f"   → Vorteil: {crossover_result['percentage_difference']:.1f}% schneller")
        print(f"   → Ratio: {crossover_result['ratio']:.4f}")
    else:
        print(f"   ⚠️  Crossover nicht im getesteten Bereich gefunden")
    
    # Analyse der Crossover-Kandidaten
    if crossover_candidates:
        print(f"\n⚡ CROSSOVER-KANDIDATEN (< 2% Unterschied):")
        for candidate in crossover_candidates:
            print(f"   {candidate['num_items']} Items: {candidate['percentage_difference']:.1f}% Unterschied")
    
    # Proof-Größen-Analyse
    print(f"\n📦 PROOF-GRÖßEN-ANALYSE:")
    if results:
        first_result = results[0]
        last_result = results[-1]
        
        print(f"   Standard Proofs (pro Item): ~{first_result['standard_avg_proof_kb']:.1f} KB")
        print(f"   Recursive Proof (konstant): ~{first_result['recursive_proof_size_kb']:.1f} KB")
        print(f"   Größenverhältnis bei {first_result['num_items']} Items: {first_result['proof_size_ratio']:.1f}:1")
        print(f"   Größenverhältnis bei {last_result['num_items']} Items: {last_result['proof_size_ratio']:.1f}:1")
    
    return exact_crossover, crossover_candidates

def main():
    """Hauptfunktion"""
    results = find_exact_crossover_88_97()
    
    if results:
        exact_crossover, candidates = analyze_precise_crossover(results)
        
        # Speichere präzise Ergebnisse
        results_dir = project_root / "data" / "precise_crossover_88_97"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "precise_crossover_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "exact_crossover_point": exact_crossover,
                "crossover_candidates": [c["num_items"] for c in candidates],
                "precise_results": results,
                "analysis_summary": {
                    "tested_range": "88-97 items",
                    "total_tests": len(results),
                    "crossover_found": exact_crossover is not None,
                    "close_calls": len(candidates)
                },
                "analysis_timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n💾 Präzise Ergebnisse gespeichert: {results_file}")
        
        # Finale Zusammenfassung
        print(f"\n🎉 PRÄZISE CROSSOVER-ANALYSE ABGESCHLOSSEN!")
        if exact_crossover:
            print(f"✅ EXAKTER CROSSOVER-POINT: {exact_crossover} Items")
        else:
            print(f"⚠️  Crossover-Point liegt außerhalb 88-97 Items")
        
    else:
        print("❌ Keine gültigen Ergebnisse erhalten")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎯 MISSION ACCOMPLISHED!' if success else '❌ MISSION FAILED!'}")
    sys.exit(0 if success else 1)
