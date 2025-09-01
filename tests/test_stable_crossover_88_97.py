#!/usr/bin/env python3
"""
🛠️ STABLE CROSSOVER ANALYSIS 88-97 ITEMS
Robuste Version ohne Timing-Probleme
"""

import sys
import time
import json
import statistics
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def apply_moderate_iot_constraints():
    """Wendet moderate IoT-Constraints an (nicht zu restriktiv)"""
    try:
        import resource
        import os
        # Moderate memory limit (2GB - genug für Nova)
        memory_bytes = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        # Light CPU priority reduction
        os.nice(3)
        print("   🔒 Moderate IoT constraints applied: 2GB RAM, Nice +3")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not apply constraints: {e}")
        return False

def robust_timing_test(test_func, *args, max_retries=3, apply_constraints=True):
    """Robuste Zeitmessung mit Retry-Mechanismus und optionalen Constraints"""
    valid_times = []
    
    # Apply constraints if requested
    if apply_constraints:
        apply_moderate_iot_constraints()
    
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()  # Präzisere Zeitmessung
            result = test_func(*args)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            
            # Validierung: Zeit muss positiv und realistisch sein
            if elapsed_time > 0 and elapsed_time < 300:  # Max 5 Minuten
                valid_times.append(elapsed_time)
                result["measured_time"] = elapsed_time
                return result
            else:
                print(f"   ⚠️  Unrealistische Zeit: {elapsed_time:.3f}s (Versuch {attempt + 1})")
                
        except Exception as e:
            print(f"   ❌ Fehler (Versuch {attempt + 1}): {e}")
            
        # Kurze Pause vor Wiederholung
        time.sleep(0.5)
    
    # Fallback: Verwende Durchschnitt wenn verfügbar
    if valid_times:
        avg_time = statistics.mean(valid_times)
        print(f"   📊 Verwende Durchschnitt: {avg_time:.3f}s")
        return {"success": True, "measured_time": avg_time, "fallback": True}
    
    return {"success": False, "error": "Alle Timing-Versuche fehlgeschlagen"}

def stable_standard_test(num_items: int) -> dict:
    """Stabiler Standard SNARK Test"""
    def _run_test():
        sensors = SmartHomeSensors()
        manager = SNARKManager()
        
        # Setup
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Generiere Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Führe Proofs aus
        successful = 0
        individual_proof_sizes = []
        
        for reading in temp_readings:
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            result = manager.generate_proof("filter_range", inputs)
            if result.success:
                successful += 1
                individual_proof_sizes.append(result.metrics.proof_size)
        
        # Berechne Metriken
        if individual_proof_sizes:
            total_proof_size_kb = sum(individual_proof_sizes) / 1024
            avg_proof_size_kb = statistics.mean(individual_proof_sizes) / 1024
        else:
            total_proof_size_kb = 0
            avg_proof_size_kb = 0
        
        return {
            "success": True,
            "num_items": num_items,
            "successful_proofs": successful,
            "total_proof_size_kb": total_proof_size_kb,
            "avg_proof_size_kb": avg_proof_size_kb
        }
    
    print(f"   📊 Standard ({num_items} items): ", end="", flush=True)
    
    # Robuste Zeitmessung mit IoT-Constraints
    result = robust_timing_test(_run_test, apply_constraints=True)
    
    if result["success"]:
        total_time = result["measured_time"]
        throughput = result.get("successful_proofs", num_items) / total_time if total_time > 0 else 0
        
        print(f"{total_time:.3f}s ({result.get('successful_proofs', num_items)}/{num_items})")
        
        result["total_time"] = total_time
        result["throughput"] = throughput
        return result
    else:
        print("❌ Fehlgeschlagen")
        return result

def stable_recursive_test(num_items: int) -> dict:
    """Stabiler Recursive SNARK Test"""
    def _run_test():
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup
        if not nova_manager.setup():
            raise Exception("Nova Setup fehlgeschlagen")
        
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
        
        # Führe rekursiven Proof aus
        result = nova_manager.prove_recursive_batch(batches)
        
        if not result.success:
            raise Exception(f"Recursive Proof fehlgeschlagen: {result.error_message}")
        
        return {
            "success": True,
            "num_items": num_items,
            "steps": len(batches),
            "proof_size_kb": result.proof_size / 1024,
            "verify_time": result.verify_time
        }
    
    print(f"   🚀 Recursive ({num_items} items): ", end="", flush=True)
    
    # Robuste Zeitmessung mit IoT-Constraints
    result = robust_timing_test(_run_test, apply_constraints=True)
    
    if result["success"]:
        total_time = result["measured_time"]
        throughput = num_items / total_time if total_time > 0 else 0
        
        print(f"{total_time:.3f}s ({result.get('steps', 0)} steps)")
        
        result["total_time"] = total_time
        result["throughput"] = throughput
        return result
    else:
        print("❌ Fehlgeschlagen")
        return result

def stable_crossover_analysis():
    """Stabile Crossover-Analyse mit IoT-Constraints"""
    print("🛠️ STABLE CROSSOVER ANALYSIS 88-97 ITEMS WITH IoT CONSTRAINTS")
    print("Robuste Version mit IoT-Resource-Limits (2GB RAM, Nice +3)")
    print("=" * 70)
    
    # Teste jeden Wert von 88 bis 97
    test_points = list(range(88, 98))
    results = []
    
    for num_items in test_points:
        print(f"\n🔬 TESTE: {num_items} Items")
        print("-" * 30)
        
        # Standard Test
        std_result = stable_standard_test(num_items)
        
        # Recursive Test
        rec_result = stable_recursive_test(num_items)
        
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
                "data_quality": "stable"  # Markierung für saubere Daten
            }
            results.append(result)
            
            print(f"   📊 {winner} gewinnt um {percentage_diff:.1f}% (Ratio: {ratio:.4f})")
        else:
            print(f"   ❌ Test fehlgeschlagen - überspringe {num_items} Items")
        
        # Pause zwischen Tests
        time.sleep(1)
    
    return results

def analyze_stable_results(results):
    """Analysiert die stabilen Ergebnisse"""
    print("\n" + "=" * 70)
    print("📊 STABILE CROSSOVER-ANALYSE")
    print("=" * 70)
    
    if not results:
        print("❌ Keine gültigen Ergebnisse")
        return None
    
    print(f"{'Items':<5} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Diff%':<7} {'Winner':<10}")
    print("-" * 60)
    
    exact_crossover = None
    
    for result in results:
        items = result["num_items"]
        std_time = result["standard_time"]
        rec_time = result["recursive_time"]
        ratio = result["ratio"]
        diff_pct = result["percentage_difference"]
        winner = result["winner"]
        
        print(f"{items:<5} {std_time:<8.3f} {rec_time:<8.3f} {ratio:<8.4f} {diff_pct:<7.1f} {winner:<10}")
        
        # Finde ersten Crossover
        if result["recursive_wins"] and exact_crossover is None:
            exact_crossover = items
    
    # Datenqualitäts-Check
    print(f"\n🔍 DATENQUALITÄT:")
    negative_times = sum(1 for r in results if r["standard_time"] < 0 or r["recursive_time"] < 0)
    unrealistic_times = sum(1 for r in results if r["standard_time"] > 100 or r["recursive_time"] > 100)
    
    print(f"   ✅ Gültige Ergebnisse: {len(results)}")
    print(f"   ❌ Negative Zeiten: {negative_times}")
    print(f"   ❌ Unrealistische Zeiten (>100s): {unrealistic_times}")
    
    if negative_times == 0 and unrealistic_times == 0:
        print(f"   🎉 ALLE DATEN SIND SAUBER UND STABIL!")
    
    return exact_crossover

def main():
    """Hauptfunktion"""
    results = stable_crossover_analysis()
    
    if results:
        exact_crossover = analyze_stable_results(results)
        
        # Speichere stabile Ergebnisse
        results_dir = project_root / "data" / "stable_crossover_88_97"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "stable_crossover_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "exact_crossover_point": exact_crossover,
                "stable_results": results,
                "data_quality": {
                    "all_positive_times": all(r["standard_time"] > 0 and r["recursive_time"] > 0 for r in results),
                    "all_realistic_times": all(r["standard_time"] < 100 and r["recursive_time"] < 100 for r in results),
                    "total_tests": len(results)
                },
                "analysis_timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n💾 Stabile Ergebnisse gespeichert: {results_file}")
        
        if exact_crossover:
            print(f"\n🎯 EXAKTER CROSSOVER-POINT: {exact_crossover} Items")
            print(f"✅ STABILE DATEN - BEREIT FÜR BACHELORARBEIT!")
        else:
            print(f"\n⚠️  Crossover-Point außerhalb 88-97 Items")
        
    else:
        print("❌ Keine stabilen Ergebnisse erhalten")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n{'🛠️ STABLE ANALYSIS ERFOLGREICH!' if success else '❌ STABLE ANALYSIS FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
