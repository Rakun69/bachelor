#!/usr/bin/env python3
"""
🎯 DETAILED CROSSOVER ANALYSIS
Findet den exakten Crossover-Point mit hoher Präzision
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

def quick_standard_test(num_items: int) -> dict:
    """Schneller Standard SNARK Test"""
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
        
        # Messe Zeit
        start_time = time.time()
        successful = 0
        total_proof_size = 0
        
        for reading in temp_readings:
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            result = manager.generate_proof("filter_range", inputs)
            if result.success:
                successful += 1
                total_proof_size += result.metrics.proof_size
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "num_items": num_items,
            "total_time": total_time,
            "successful_proofs": successful,
            "total_proof_size_kb": total_proof_size / 1024,
            "avg_time_per_item": total_time / num_items if num_items > 0 else 0,
            "throughput": successful / total_time if total_time > 0 else 0
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def quick_recursive_test(num_items: int) -> dict:
    """Schneller Recursive SNARK Test"""
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
        
        # Messe Zeit
        start_time = time.time()
        result = nova_manager.prove_recursive_batch(batches)
        total_time = time.time() - start_time
        
        if result.success:
            return {
                "success": True,
                "num_items": num_items,
                "total_time": total_time,
                "steps": len(batches),
                "proof_size_kb": result.proof_size / 1024,
                "verify_time": result.verify_time,
                "throughput": num_items / total_time if total_time > 0 else 0
            }
        else:
            return {"success": False, "error": result.error_message}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def find_precise_crossover():
    """Findet den präzisen Crossover-Point"""
    print("🎯 DETAILED CROSSOVER ANALYSIS")
    print("Findet den exakten Crossover-Point mit hoher Präzision")
    print("=" * 70)
    
    # Grober Scan (10er Schritte)
    print("🔍 Phase 1: Grober Scan (10er Schritte)")
    coarse_range = list(range(50, 151, 10))  # 50, 60, 70, ..., 150
    coarse_results = []
    
    for num_items in coarse_range:
        print(f"\n📊 Teste {num_items} Items:")
        
        std_result = quick_standard_test(num_items)
        rec_result = quick_recursive_test(num_items)
        
        if std_result["success"] and rec_result["success"]:
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if rec_result["total_time"] < std_result["total_time"] else "Standard"
            
            result = {
                "num_items": num_items,
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "ratio": ratio,
                "winner": winner,
                "recursive_wins": rec_result["total_time"] < std_result["total_time"]
            }
            coarse_results.append(result)
            
            print(f"   Standard: {std_result['total_time']:.3f}s")
            print(f"   Recursive: {rec_result['total_time']:.3f}s")
            print(f"   Ratio: {ratio:.3f} - Winner: {winner}")
        
        time.sleep(0.5)
    
    # Finde Crossover-Bereich
    crossover_range = None
    for i in range(len(coarse_results) - 1):
        current = coarse_results[i]
        next_result = coarse_results[i + 1]
        
        if not current["recursive_wins"] and next_result["recursive_wins"]:
            crossover_range = (current["num_items"], next_result["num_items"])
            break
    
    if not crossover_range:
        print("⚠️  Crossover-Bereich nicht in grobem Scan gefunden")
        return coarse_results
    
    # Feiner Scan im Crossover-Bereich
    print(f"\n🔍 Phase 2: Feiner Scan ({crossover_range[0]} - {crossover_range[1]})")
    fine_range = list(range(crossover_range[0], crossover_range[1] + 1, 2))  # 2er Schritte
    fine_results = []
    
    for num_items in fine_range:
        print(f"\n📊 Teste {num_items} Items (fein):")
        
        std_result = quick_standard_test(num_items)
        rec_result = quick_recursive_test(num_items)
        
        if std_result["success"] and rec_result["success"]:
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if rec_result["total_time"] < std_result["total_time"] else "Standard"
            
            result = {
                "num_items": num_items,
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "ratio": ratio,
                "winner": winner,
                "recursive_wins": rec_result["total_time"] < std_result["total_time"],
                "standard_throughput": std_result["throughput"],
                "recursive_throughput": rec_result["throughput"],
                "standard_proof_size_kb": std_result["total_proof_size_kb"],
                "recursive_proof_size_kb": rec_result["proof_size_kb"]
            }
            fine_results.append(result)
            
            print(f"   Standard: {std_result['total_time']:.3f}s")
            print(f"   Recursive: {rec_result['total_time']:.3f}s")
            print(f"   Ratio: {ratio:.3f} - Winner: {winner}")
        
        time.sleep(0.5)
    
    # Kombiniere Ergebnisse
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda x: x["num_items"])
    
    # Finde exakten Crossover-Point
    exact_crossover = None
    for result in all_results:
        if result["recursive_wins"]:
            exact_crossover = result["num_items"]
            break
    
    return all_results, exact_crossover

def analyze_crossover_trends(results):
    """Analysiert Trends um den Crossover-Point"""
    print("\n" + "=" * 70)
    print("📈 CROSSOVER TREND ANALYSE")
    print("=" * 70)
    
    print(f"{'Items':<6} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<7} {'Winner':<10} {'Std_KB':<8} {'Rec_KB':<8}")
    print("-" * 70)
    
    for result in results:
        items = result["num_items"]
        std_time = result["standard_time"]
        rec_time = result["recursive_time"]
        ratio = result["ratio"]
        winner = result["winner"]
        std_kb = result.get("standard_proof_size_kb", 0)
        rec_kb = result.get("recursive_proof_size_kb", 0)
        
        print(f"{items:<6} {std_time:<8.3f} {rec_time:<8.3f} {ratio:<7.3f} {winner:<10} {std_kb:<8.1f} {rec_kb:<8.1f}")
    
    # Berechne Trends
    if len(results) >= 3:
        first_third = results[:len(results)//3]
        last_third = results[-len(results)//3:]
        
        avg_ratio_early = sum(r["ratio"] for r in first_third) / len(first_third)
        avg_ratio_late = sum(r["ratio"] for r in last_third) / len(last_third)
        
        print(f"\n📊 TRENDS:")
        print(f"   Frühe Ratios (Durchschnitt): {avg_ratio_early:.3f}")
        print(f"   Späte Ratios (Durchschnitt): {avg_ratio_late:.3f}")
        print(f"   Trend: {'Recursive wird besser' if avg_ratio_late < avg_ratio_early else 'Standard bleibt besser'}")

def main():
    """Hauptfunktion"""
    results, crossover_point = find_precise_crossover()
    
    if crossover_point:
        print(f"\n🎯 EXAKTER CROSSOVER POINT GEFUNDEN: {crossover_point} Items")
        print(f"   → Ab {crossover_point} Items sind Recursive SNARKs schneller!")
    else:
        print("\n⚠️  Crossover Point nicht im getesteten Bereich gefunden")
    
    analyze_crossover_trends(results)
    
    # Speichere detaillierte Ergebnisse
    results_dir = project_root / "data" / "detailed_crossover_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "detailed_crossover_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "crossover_point": crossover_point,
            "detailed_results": results,
            "analysis_timestamp": time.time(),
            "total_tests": len(results)
        }, f, indent=2)
    
    print(f"\n💾 Detaillierte Ergebnisse gespeichert: {results_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 DETAILED CROSSOVER ANALYSIS ABGESCHLOSSEN!' if success else '❌ DETAILED CROSSOVER ANALYSIS FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
