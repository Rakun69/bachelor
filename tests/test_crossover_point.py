#!/usr/bin/env python3
"""
🎯 CROSSOVER POINT TEST
Findet den exakten Punkt, wo Recursive SNARKs besser werden als Standard SNARKs
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

def quick_standard_test(num_items: int) -> float:
    """Schneller Standard SNARK Test - nur Gesamtzeit"""
    print(f"   📊 Standard: {num_items} Items", end=" -> ")
    
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
        
        # Messe nur Gesamtzeit
        start_time = time.time()
        successful = 0
        
        for reading in temp_readings:
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            result = manager.generate_proof("filter_range", inputs)
            if result.success:
                successful += 1
        
        total_time = time.time() - start_time
        print(f"{total_time:.3f}s ({successful}/{num_items} proofs)")
        return total_time
        
    except Exception as e:
        print(f"ERROR: {e}")
        return float('inf')

def quick_recursive_test(num_items: int) -> float:
    """Schneller Recursive SNARK Test - nur Gesamtzeit"""
    print(f"   🚀 Recursive: {num_items} Items", end=" -> ")
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup
        if not nova_manager.setup():
            raise Exception("Nova Setup failed")
        
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
        
        # Messe nur Gesamtzeit
        start_time = time.time()
        result = nova_manager.prove_recursive_batch(batches)
        total_time = time.time() - start_time
        
        if result.success:
            print(f"{total_time:.3f}s ({len(batches)} steps)")
            return total_time
        else:
            print(f"FAILED")
            return float('inf')
        
    except Exception as e:
        print(f"ERROR: {e}")
        return float('inf')

def find_crossover_point():
    """Findet den exakten Crossover Point"""
    print("🎯 CROSSOVER POINT SUCHE")
    print("Sucht den Punkt, wo Recursive SNARKs schneller werden")
    print("=" * 60)
    
    # Basierend auf bisherigen Daten: Crossover bei ~85-90 Items
    # Teste um diesen Bereich herum
    test_points = [60, 70, 80, 85, 90, 95, 100, 110, 120]
    
    results = []
    crossover_found = False
    crossover_point = None
    
    for num_items in test_points:
        print(f"\n🔬 TESTE: {num_items} Items")
        print("-" * 30)
        
        # Standard Test
        std_time = quick_standard_test(num_items)
        
        # Recursive Test  
        rec_time = quick_recursive_test(num_items)
        
        # Analyse
        if std_time != float('inf') and rec_time != float('inf'):
            ratio = rec_time / std_time
            advantage = "✅ RECURSIVE" if rec_time < std_time else "❌ STANDARD"
            
            print(f"   📊 Ratio: {ratio:.2f}x - {advantage}")
            
            results.append({
                "items": num_items,
                "standard_time": std_time,
                "recursive_time": rec_time,
                "ratio": ratio,
                "recursive_wins": rec_time < std_time
            })
            
            # Crossover gefunden?
            if rec_time < std_time and not crossover_found:
                crossover_found = True
                crossover_point = num_items
                print(f"   🎉 CROSSOVER POINT GEFUNDEN!")
        
        # Kurze Pause
        time.sleep(0.5)
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("🎯 CROSSOVER POINT ANALYSE")
    print("=" * 60)
    
    if crossover_found:
        print(f"✅ CROSSOVER POINT: {crossover_point} Items")
        print(f"   → Ab {crossover_point} Items sind Recursive SNARKs schneller!")
        
        # Finde das genaue Verhältnis am Crossover
        crossover_result = next(r for r in results if r["items"] == crossover_point)
        speedup = crossover_result["standard_time"] / crossover_result["recursive_time"]
        print(f"   → Speedup: {speedup:.2f}x")
        
    else:
        print("⚠️  Crossover Point nicht in diesem Bereich gefunden")
        print("   → Teste größere Werte oder verfeinere den Bereich")
    
    # Detaillierte Tabelle
    print(f"\n📊 DETAILLIERTE ERGEBNISSE:")
    print(f"{'Items':<6} {'Standard(s)':<12} {'Recursive(s)':<13} {'Ratio':<8} {'Winner':<10}")
    print("-" * 55)
    
    for result in results:
        items = result["items"]
        std_time = result["standard_time"]
        rec_time = result["recursive_time"]
        ratio = result["ratio"]
        winner = "Recursive" if result["recursive_wins"] else "Standard"
        
        print(f"{items:<6} {std_time:<12.3f} {rec_time:<13.3f} {ratio:<8.2f} {winner:<10}")
    
    # Speichere Ergebnisse
    results_dir = project_root / "data" / "crossover_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "crossover_point.json"
    with open(results_file, 'w') as f:
        json.dump({
            "crossover_found": crossover_found,
            "crossover_point": crossover_point,
            "test_points": test_points,
            "results": results,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n💾 Ergebnisse gespeichert: {results_file}")
    
    return crossover_point, results

def main():
    """Hauptfunktion"""
    print("🎯 CROSSOVER POINT TEST")
    print("Findet den exakten Punkt für deine Bachelorarbeit")
    print("=" * 60)
    
    try:
        crossover_point, results = find_crossover_point()
        
        print("\n🎉 CROSSOVER POINT TEST ABGESCHLOSSEN!")
        
        if crossover_point:
            print(f"✅ Crossover Point gefunden: {crossover_point} Items")
            print("🔥 PERFEKTE DATEN FÜR DEINE BACHELORARBEIT!")
        else:
            print("⚠️  Crossover Point nicht gefunden - teste größere Bereiche")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
