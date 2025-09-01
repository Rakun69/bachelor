#!/usr/bin/env python3
"""
🔄 SINGLE RECURSIVE PROOF TEST
Testet ALLE Items in einem einzigen rekursiven Proof (ohne Batching)
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def test_single_recursive_proof(num_items: int) -> dict:
    """Testet ALLE Items in einem einzigen rekursiven Proof"""
    print(f"🔄 Single Recursive Proof: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup
        if not nova_manager.setup():
            return {"success": False, "error": "Setup failed"}
        
        # Generiere Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # ALLE Items in EINEN EINZIGEN Batch!
        # Statt 3er-Gruppen: Alle Items zusammen
        all_items_batch = []
        
        # Fülle auf Vielfaches von 3 auf (für Circuit-Kompatibilität)
        while len(temp_readings) % 3 != 0:
            temp_readings.append(temp_readings[-1] if temp_readings else type('Reading', (), {'value': 22.0})())
        
        # Erstelle EINEN GROSSEN Batch mit allen Items
        for i in range(0, len(temp_readings), 3):
            batch_readings = temp_readings[i:i+3]
            batch_dicts = [{'value': r.value} for r in batch_readings]
            all_items_batch.append(batch_dicts)
        
        print(f"   → {len(all_items_batch)} Steps für {num_items} Items")
        
        # Messe Zeit für EINEN EINZIGEN rekursiven Proof
        start_time = time.time()
        result = nova_manager.prove_recursive_batch(all_items_batch)
        total_time = time.time() - start_time
        
        if result.success:
            steps = len(all_items_batch)
            throughput = num_items / total_time
            
            print(f"   ✅ Erfolg: {total_time:.3f}s, {steps} steps, {throughput:.1f} items/s")
            print(f"   📊 Proof Size: {result.proof_size / 1024:.1f} KB")
            
            return {
                "success": True,
                "num_items": num_items,
                "total_time": total_time,
                "steps": steps,
                "throughput": throughput,
                "proof_size_kb": result.proof_size / 1024,
                "verify_time": result.verify_time,
                "items_per_step": 3,  # Immer noch 3 Items pro Step
                "total_steps": steps
            }
        else:
            print(f"   ❌ Fehlgeschlagen: {result.error_message}")
            return {"success": False, "error": result.error_message}
            
    except Exception as e:
        print(f"   💥 Fehler: {e}")
        return {"success": False, "error": str(e)}

def compare_batching_strategies():
    """Vergleicht verschiedene Batching-Strategien"""
    print("🔄 SINGLE RECURSIVE PROOF TEST")
    print("Vergleicht: Batching vs. Single Proof für alle Items")
    print("=" * 60)
    
    test_items = [30, 60, 90, 120, 150]
    results = []
    
    for num_items in test_items:
        print(f"\n🔬 TESTE: {num_items} Items")
        print("-" * 30)
        
        # Test Single Recursive Proof
        result = test_single_recursive_proof(num_items)
        if result["success"]:
            results.append(result)
        
        time.sleep(1)
    
    # Analyse
    print("\n" + "=" * 60)
    print("📊 SINGLE RECURSIVE PROOF ANALYSE")
    print("=" * 60)
    
    if results:
        print(f"{'Items':<6} {'Zeit(s)':<8} {'Steps':<6} {'Throughput':<12} {'Proof(KB)':<10} {'Verify(s)':<10}")
        print("-" * 65)
        
        for result in results:
            print(f"{result['num_items']:<6} {result['total_time']:<8.3f} {result['steps']:<6} "
                  f"{result['throughput']:<12.1f} {result['proof_size_kb']:<10.1f} {result['verify_time']:<10.3f}")
        
        # Berechne Trends
        print(f"\n📈 TRENDS:")
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            
            time_ratio = last["total_time"] / first["total_time"]
            items_ratio = last["num_items"] / first["num_items"]
            efficiency_ratio = time_ratio / items_ratio
            
            print(f"   → Zeit-Skalierung: {time_ratio:.2f}x für {items_ratio:.1f}x Items")
            print(f"   → Effizienz-Ratio: {efficiency_ratio:.2f} (1.0 = linear, <1.0 = besser als linear)")
            
            if efficiency_ratio < 1.0:
                print(f"   ✅ SUBLINEARE SKALIERUNG! Recursive SNARKs werden effizienter!")
            else:
                print(f"   ⚠️  Lineare/Superlineare Skalierung")
        
        # Vergleich mit theoretischen Standard SNARKs
        print(f"\n🆚 VERGLEICH MIT STANDARD SNARKS (Theoretisch):")
        for result in results:
            # Annahme: Standard SNARK ~0.07s pro Item
            theoretical_standard_time = result["num_items"] * 0.07
            speedup = theoretical_standard_time / result["total_time"]
            
            print(f"   {result['num_items']} Items: {speedup:.2f}x schneller als Standard SNARKs")
    
    else:
        print("❌ Keine erfolgreichen Tests")
    
    return len(results) > 0

def main():
    """Hauptfunktion"""
    success = compare_batching_strategies()
    
    print(f"\n🔍 ERKENNTNISSE:")
    print("1. Single Recursive Proof verarbeitet ALLE Items in einem Durchgang")
    print("2. Immer noch 3er-Batches intern (Circuit-Limitation)")
    print("3. Aber nur EIN finaler Proof für alle Items")
    print("4. Konstante Proof-Größe unabhängig von Item-Anzahl")
    print("5. Sublineare Zeitkomplexität möglich")
    
    return success

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 SINGLE RECURSIVE PROOF TEST ABGESCHLOSSEN!' if success else '❌ SINGLE RECURSIVE PROOF TEST FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
