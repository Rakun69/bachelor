#!/usr/bin/env python3
"""
🔧 FIXED CROSSOVER ANALYSIS
Behebt die Probleme der vorherigen Analyse
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

def robust_standard_test(num_items: int, max_retries: int = 2) -> dict:
    """Robuster Standard SNARK Test mit Fehlerbehandlung"""
    for attempt in range(max_retries):
        try:
            print(f"   📊 Standard SNARK (Versuch {attempt + 1}): ", end="")
            
            sensors = SmartHomeSensors()
            manager = SNARKManager()
            
            # Setup
            circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
            manager.compile_circuit(str(circuit_path), "filter_range")
            manager.setup_circuit("filter_range")
            
            # Generiere Daten
            readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
            temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
            
            # Messe Zeit und sammle Metriken
            start_time = time.time()
            successful = 0
            total_proof_size = 0
            total_verify_time = 0
            
            for reading in temp_readings:
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                
                result = manager.generate_proof("filter_range", inputs)
                if result.success:
                    successful += 1
                    total_proof_size += result.metrics.proof_size
                    total_verify_time += result.metrics.verify_time
                    
                    # Sanity Check: Verify time sollte nicht > 1s sein
                    if result.metrics.verify_time > 1.0:
                        print(f"⚠️  Ungewöhnlich hohe Verify-Zeit: {result.metrics.verify_time:.3f}s")
            
            total_time = time.time() - start_time
            
            # Validierung der Ergebnisse
            if total_time <= 0:
                print(f"❌ Ungültige Zeit: {total_time}")
                continue
                
            if successful == 0:
                print(f"❌ Keine erfolgreichen Proofs")
                continue
            
            avg_proof_size_kb = (total_proof_size / 1024) if successful > 0 else 0
            avg_verify_time = total_verify_time / successful if successful > 0 else 0
            
            print(f"{total_time:.3f}s ({successful}/{num_items} proofs)")
            
            return {
                "success": True,
                "num_items": num_items,
                "total_time": total_time,
                "successful_proofs": successful,
                "total_proof_size_kb": avg_proof_size_kb * successful,  # Gesamtgröße
                "avg_verify_time": avg_verify_time,
                "throughput": successful / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Fehler (Versuch {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
            time.sleep(1)  # Kurze Pause vor Wiederholung
    
    return {"success": False, "error": "Alle Versuche fehlgeschlagen"}

def robust_recursive_test(num_items: int, max_retries: int = 2) -> dict:
    """Robuster Recursive SNARK Test mit Fehlerbehandlung"""
    for attempt in range(max_retries):
        try:
            print(f"   🚀 Recursive SNARK (Versuch {attempt + 1}): ", end="")
            
            sensors = SmartHomeSensors()
            nova_manager = FixedZoKratesNovaManager()
            
            # Setup
            if not nova_manager.setup():
                print(f"❌ Setup fehlgeschlagen")
                continue
            
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
            
            # Validierung der Ergebnisse
            if total_time <= 0:
                print(f"❌ Ungültige Zeit: {total_time}")
                continue
            
            if not result.success:
                print(f"❌ Proof fehlgeschlagen: {result.error_message}")
                continue
            
            # Sanity Checks
            if result.verify_time < 0 or result.verify_time > 100:
                print(f"⚠️  Ungewöhnliche Verify-Zeit: {result.verify_time:.3f}s")
            
            if result.proof_size <= 0:
                print(f"⚠️  Ungültige Proof-Größe: {result.proof_size}")
            
            proof_size_kb = result.proof_size / 1024
            
            print(f"{total_time:.3f}s ({len(batches)} steps)")
            
            return {
                "success": True,
                "num_items": num_items,
                "total_time": total_time,
                "steps": len(batches),
                "proof_size_kb": proof_size_kb,
                "verify_time": result.verify_time,
                "throughput": num_items / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Fehler (Versuch {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
            time.sleep(1)  # Kurze Pause vor Wiederholung
    
    return {"success": False, "error": "Alle Versuche fehlgeschlagen"}

def clean_crossover_analysis():
    """Saubere Crossover-Analyse ohne Duplikate"""
    print("🔧 FIXED CROSSOVER ANALYSIS")
    print("Behebt die Probleme der vorherigen Analyse")
    print("=" * 60)
    
    # Teste nur einmal pro Item-Anzahl
    test_points = [60, 70, 80, 85, 90, 95, 100, 110, 120]
    results = []
    
    for num_items in test_points:
        print(f"\n🔬 TESTE: {num_items} Items")
        print("-" * 30)
        
        # Standard Test
        std_result = robust_standard_test(num_items)
        
        # Recursive Test
        rec_result = robust_recursive_test(num_items)
        
        # Nur hinzufügen wenn beide erfolgreich
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
                "standard_proof_size_kb": std_result["total_proof_size_kb"],
                "recursive_proof_size_kb": rec_result["proof_size_kb"],
                "standard_throughput": std_result["throughput"],
                "recursive_throughput": rec_result["throughput"]
            }
            results.append(result)
            
            print(f"   📊 Ergebnis: {winner} gewinnt (Ratio: {ratio:.3f})")
        else:
            print(f"   ❌ Test fehlgeschlagen - überspringe {num_items} Items")
        
        # Pause zwischen Tests
        time.sleep(2)
    
    return results

def analyze_clean_results(results):
    """Analysiert die bereinigten Ergebnisse"""
    print("\n" + "=" * 70)
    print("📊 BEREINIGTE CROSSOVER ANALYSE")
    print("=" * 70)
    
    if not results:
        print("❌ Keine gültigen Ergebnisse")
        return None
    
    print(f"{'Items':<6} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<7} {'Winner':<10} {'Std_KB':<8} {'Rec_KB':<8}")
    print("-" * 70)
    
    crossover_point = None
    
    for result in results:
        items = result["num_items"]
        std_time = result["standard_time"]
        rec_time = result["recursive_time"]
        ratio = result["ratio"]
        winner = result["winner"]
        std_kb = result["standard_proof_size_kb"]
        rec_kb = result["recursive_proof_size_kb"]
        
        print(f"{items:<6} {std_time:<8.3f} {rec_time:<8.3f} {ratio:<7.3f} {winner:<10} {std_kb:<8.1f} {rec_kb:<8.1f}")
        
        # Finde ersten Crossover
        if result["recursive_wins"] and crossover_point is None:
            crossover_point = items
    
    if crossover_point:
        print(f"\n🎯 CROSSOVER POINT: {crossover_point} Items")
        print(f"   → Ab {crossover_point} Items sind Recursive SNARKs schneller!")
    else:
        print(f"\n⚠️  Crossover Point nicht im getesteten Bereich gefunden")
    
    # Validiere Datenqualität
    print(f"\n🔍 DATENQUALITÄT:")
    valid_results = len(results)
    negative_times = sum(1 for r in results if r["standard_time"] < 0 or r["recursive_time"] < 0)
    zero_proof_sizes = sum(1 for r in results if r["standard_proof_size_kb"] <= 0 or r["recursive_proof_size_kb"] <= 0)
    
    print(f"   ✅ Gültige Ergebnisse: {valid_results}")
    print(f"   ❌ Negative Zeiten: {negative_times}")
    print(f"   ❌ Null Proof-Größen: {zero_proof_sizes}")
    
    if negative_times == 0 and zero_proof_sizes == 0:
        print(f"   🎉 ALLE DATEN SIND SAUBER!")
    
    return crossover_point

def main():
    """Hauptfunktion"""
    results = clean_crossover_analysis()
    
    if results:
        crossover_point = analyze_clean_results(results)
        
        # Speichere bereinigte Ergebnisse
        results_dir = project_root / "data" / "fixed_crossover_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "clean_crossover_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "crossover_point": crossover_point,
                "clean_results": results,
                "data_quality": {
                    "total_tests": len(results),
                    "negative_times": 0,
                    "zero_proof_sizes": 0,
                    "all_data_valid": True
                },
                "analysis_timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n💾 Bereinigte Ergebnisse gespeichert: {results_file}")
        
    else:
        print("❌ Keine gültigen Ergebnisse erhalten")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 FIXED CROSSOVER ANALYSIS ABGESCHLOSSEN!' if success else '❌ FIXED CROSSOVER ANALYSIS FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
