#!/usr/bin/env python3
"""
📦 BATCH SIZE IMPACT STUDY
Analysiert den Einfluss verschiedener Batch-Größen auf Performance und Kosten
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
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def test_batch_size_performance(batch_size: int, num_items: int = 120) -> dict:
    """Testet Performance für eine spezifische Batch-Größe"""
    print(f"📦 Teste Batch-Größe {batch_size} mit {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup
        if not nova_manager.setup():
            return {"success": False, "error": "Setup failed"}
        
        # Generiere Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Bereite Batches mit variabler Größe vor
        batches = []
        for i in range(0, len(temp_readings), batch_size):
            batch_readings = temp_readings[i:i+batch_size]
            
            # Fülle auf Batch-Größe auf (wichtig für Circuit-Kompatibilität)
            while len(batch_readings) < batch_size:
                batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
            
            batch_dicts = [{'value': r.value} for r in batch_readings]
            batches.append(batch_dicts)
        
        # Messe Performance
        start_time = time.time()
        result = nova_manager.prove_recursive_batch(batches)
        total_time = time.time() - start_time
        
        if result.success:
            steps = len(batches)
            throughput = num_items / total_time
            time_per_step = total_time / steps if steps > 0 else 0
            items_per_step = batch_size
            efficiency = throughput / batch_size  # Normalisierte Effizienz
            
            print(f"   ✅ Erfolg: {total_time:.3f}s, {steps} steps, {throughput:.1f} items/s")
            
            return {
                "success": True,
                "batch_size": batch_size,
                "num_items": num_items,
                "total_time": total_time,
                "steps": steps,
                "throughput": throughput,
                "time_per_step": time_per_step,
                "items_per_step": items_per_step,
                "efficiency": efficiency,
                "proof_size_kb": result.proof_size / 1024,
                "verify_time": result.verify_time,
                "cost_per_item": total_time / num_items  # Zeit-Kosten pro Item
            }
        else:
            print(f"   ❌ Fehlgeschlagen: {result.error_message}")
            return {"success": False, "error": result.error_message}
            
    except Exception as e:
        print(f"   💥 Fehler: {e}")
        return {"success": False, "error": str(e)}

def comprehensive_batch_analysis():
    """Umfassende Batch-Größen-Analyse"""
    print("📦 BATCH SIZE IMPACT STUDY")
    print("Analysiert den Einfluss verschiedener Batch-Größen")
    print("=" * 60)
    
    # Teste verschiedene Batch-Größen
    batch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    num_items = 120  # Konstante Item-Anzahl für Vergleichbarkeit
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n🔬 BATCH-GRÖßE: {batch_size}")
        print("-" * 30)
        
        result = test_batch_size_performance(batch_size, num_items)
        if result["success"]:
            results.append(result)
        
        # Kurze Pause zwischen Tests
        time.sleep(1)
    
    return results

def analyze_batch_impact(results):
    """Analysiert den Einfluss der Batch-Größe"""
    print("\n" + "=" * 80)
    print("📊 BATCH SIZE IMPACT ANALYSE")
    print("=" * 80)
    
    if not results:
        print("❌ Keine erfolgreichen Tests")
        return
    
    # Sortiere nach Effizienz
    results_by_efficiency = sorted(results, key=lambda x: x["throughput"], reverse=True)
    results_by_time = sorted(results, key=lambda x: x["total_time"])
    
    print("🏆 RANKING NACH THROUGHPUT:")
    print(f"{'Rank':<5} {'Batch':<6} {'Time(s)':<8} {'Steps':<6} {'Throughput':<12} {'Efficiency':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results_by_efficiency[:5], 1):
        batch_size = result["batch_size"]
        total_time = result["total_time"]
        steps = result["steps"]
        throughput = result["throughput"]
        efficiency = result["efficiency"]
        
        print(f"{i:<5} {batch_size:<6} {total_time:<8.3f} {steps:<6} {throughput:<12.1f} {efficiency:<10.2f}")
    
    # Detaillierte Analyse
    print(f"\n📈 DETAILLIERTE ANALYSE:")
    print(f"{'Batch':<6} {'Zeit(s)':<8} {'Steps':<6} {'Zeit/Step':<10} {'Proof(KB)':<10} {'Kosten/Item':<12}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x["batch_size"]):
        batch_size = result["batch_size"]
        total_time = result["total_time"]
        steps = result["steps"]
        time_per_step = result["time_per_step"]
        proof_size = result["proof_size_kb"]
        cost_per_item = result["cost_per_item"]
        
        print(f"{batch_size:<6} {total_time:<8.3f} {steps:<6} {time_per_step:<10.3f} {proof_size:<10.1f} {cost_per_item:<12.4f}")
    
    # Finde optimale Batch-Größe
    best_throughput = results_by_efficiency[0]
    best_time = results_by_time[0]
    
    print(f"\n🎯 OPTIMIERUNGSEMPFEHLUNGEN:")
    print(f"   🚀 Beste Throughput: Batch-Größe {best_throughput['batch_size']} ({best_throughput['throughput']:.1f} items/s)")
    print(f"   ⚡ Schnellste Zeit: Batch-Größe {best_time['batch_size']} ({best_time['total_time']:.3f}s)")
    
    # Analyse von Trends
    small_batches = [r for r in results if r["batch_size"] <= 3]
    large_batches = [r for r in results if r["batch_size"] >= 8]
    
    if small_batches and large_batches:
        avg_small_throughput = sum(r["throughput"] for r in small_batches) / len(small_batches)
        avg_large_throughput = sum(r["throughput"] for r in large_batches) / len(large_batches)
        
        print(f"\n📊 BATCH-GRÖßEN-TRENDS:")
        print(f"   Kleine Batches (≤3): Durchschnitt {avg_small_throughput:.1f} items/s")
        print(f"   Große Batches (≥8): Durchschnitt {avg_large_throughput:.1f} items/s")
        
        if avg_large_throughput > avg_small_throughput:
            improvement = (avg_large_throughput / avg_small_throughput - 1) * 100
            print(f"   → Große Batches sind {improvement:.1f}% effizienter")
        else:
            improvement = (avg_small_throughput / avg_large_throughput - 1) * 100
            print(f"   → Kleine Batches sind {improvement:.1f}% effizienter")

def compare_with_current_implementation(results):
    """Vergleicht mit der aktuellen 3er-Batch Implementierung"""
    print(f"\n🔍 VERGLEICH MIT AKTUELLER IMPLEMENTIERUNG (Batch-Größe 3):")
    
    current_impl = next((r for r in results if r["batch_size"] == 3), None)
    if not current_impl:
        print("❌ Batch-Größe 3 nicht getestet")
        return
    
    best_result = max(results, key=lambda x: x["throughput"])
    
    if best_result["batch_size"] != 3:
        improvement = (best_result["throughput"] / current_impl["throughput"] - 1) * 100
        time_saving = current_impl["total_time"] - best_result["total_time"]
        
        print(f"   📊 Aktuelle Implementierung (Batch 3):")
        print(f"      Zeit: {current_impl['total_time']:.3f}s")
        print(f"      Throughput: {current_impl['throughput']:.1f} items/s")
        print(f"      Steps: {current_impl['steps']}")
        
        print(f"   🚀 Optimale Batch-Größe ({best_result['batch_size']}):")
        print(f"      Zeit: {best_result['total_time']:.3f}s")
        print(f"      Throughput: {best_result['throughput']:.1f} items/s")
        print(f"      Steps: {best_result['steps']}")
        
        print(f"   💡 VERBESSERUNG:")
        print(f"      {improvement:.1f}% schnellerer Throughput")
        print(f"      {time_saving:.3f}s Zeitersparnis")
        print(f"      Empfehlung: Wechsel zu Batch-Größe {best_result['batch_size']}")
    else:
        print(f"   ✅ Aktuelle Batch-Größe 3 ist bereits optimal!")

def main():
    """Hauptfunktion"""
    results = comprehensive_batch_analysis()
    
    if results:
        analyze_batch_impact(results)
        compare_with_current_implementation(results)
        
        # Speichere Ergebnisse
        results_dir = project_root / "data" / "batch_size_impact"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "batch_size_impact_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "batch_sizes_tested": [r["batch_size"] for r in results],
                "results": results,
                "best_throughput": max(results, key=lambda x: x["throughput"]),
                "best_time": min(results, key=lambda x: x["total_time"]),
                "analysis_timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n💾 Ergebnisse gespeichert: {results_file}")
        
    else:
        print("❌ Keine erfolgreichen Tests")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 BATCH SIZE IMPACT STUDY ABGESCHLOSSEN!' if success else '❌ BATCH SIZE IMPACT STUDY FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
