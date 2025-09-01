#!/usr/bin/env python3
"""
🔬 FINAL COMPARISON TEST
Basiert auf den funktionierenden Tests und erstellt eine Vergleichstabelle
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

def run_standard_test(num_items: int) -> dict:
    """Führt Standard SNARK Test durch - basiert auf test_basic_functionality.py"""
    print(f"   📊 Standard SNARK: {num_items} Items")
    
    try:
        # Setup wie in test_basic_functionality.py
        sensors = SmartHomeSensors()
        manager = SNARKManager()
        
        # Kompiliere filter_range circuit
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        
        compile_start = time.time()
        success = manager.compile_circuit(str(circuit_path), "filter_range")
        compile_time = time.time() - compile_start
        
        if not success:
            raise Exception("Circuit compilation failed")
        
        # Setup
        setup_start = time.time()
        manager.setup_circuit("filter_range")
        setup_time = time.time() - setup_start
        
        # Generiere IoT Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Führe Proofs durch
        total_proof_time = 0
        total_verify_time = 0
        total_proof_size = 0
        successful_proofs = 0
        
        for reading in temp_readings:
            try:
                secret_value = max(10, min(50, int(reading.value)))  # Clamp zwischen 10-50
                inputs = ["10", "50", str(secret_value)]
                
                # Generate proof (includes witness, proof, verify)
                proof_start = time.time()
                circuit_result = manager.generate_proof("filter_range", inputs)
                proof_time = time.time() - proof_start
                total_proof_time += proof_time
                
                if circuit_result.success:
                    successful_proofs += 1
                    # Extract metrics
                    metrics = circuit_result.metrics
                    total_verify_time += metrics.verify_time
                    total_proof_size += metrics.proof_size
                    
            except Exception as e:
                print(f"      ⚠️  Proof fehlgeschlagen: {e}")
        
        total_time = compile_time + setup_time + total_proof_time + total_verify_time
        throughput = successful_proofs / total_time if total_time > 0 else 0
        
        return {
            "type": "Standard",
            "num_items": num_items,
            "successful_proofs": successful_proofs,
            "compile_time": compile_time,
            "setup_time": setup_time,
            "total_proof_time": total_proof_time,
            "total_verify_time": total_verify_time,
            "total_proof_size": total_proof_size,
            "total_time": total_time,
            "throughput": throughput,
            "success": True
        }
        
    except Exception as e:
        print(f"      ❌ Standard SNARK Fehler: {e}")
        return {
            "type": "Standard",
            "num_items": num_items,
            "error": str(e),
            "success": False
        }

def run_recursive_test(num_items: int) -> dict:
    """Führt Recursive SNARK Test durch - basiert auf test_nova_functionality.py"""
    print(f"   🚀 Recursive SNARK: {num_items} Items")
    
    try:
        # Setup wie in test_nova_functionality.py
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Setup Nova
        setup_start = time.time()
        setup_success = nova_manager.setup()
        setup_time = time.time() - setup_start
        
        if not setup_success:
            raise Exception("Nova Setup fehlgeschlagen")
        
        # Generiere IoT Daten
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Bereite Batches vor (3 Werte pro Batch im richtigen Format)
        batches = []
        for i in range(0, len(temp_readings), 3):
            batch_readings = temp_readings[i:i+3]
            # Fülle auf 3 Werte auf falls nötig
            while len(batch_readings) < 3:
                if batch_readings:
                    batch_readings.append(batch_readings[-1])
                else:
                    # Fallback: erstelle dummy reading
                    batch_readings.append(type('Reading', (), {'value': 22.0})())
            
            # Konvertiere zu Dict-Format für Nova
            batch_dicts = [{'value': r.value} for r in batch_readings]
            batches.append(batch_dicts)
        
        # Führe Recursive Proof durch
        proof_start = time.time()
        proof_result = nova_manager.prove_recursive_batch(batches)
        proof_time = time.time() - proof_start
        
        if not proof_result.success:
            raise Exception(f"Nova Proof fehlgeschlagen: {getattr(proof_result, 'error', 'Unbekannter Fehler')}")
        
        verify_time = proof_result.verify_time
        proof_size = proof_result.proof_size
        
        total_time = setup_time + proof_time + verify_time
        throughput = num_items / total_time if total_time > 0 else 0
        
        return {
            "type": "Recursive",
            "num_items": num_items,
            "setup_time": setup_time,
            "total_proof_time": proof_time,
            "total_verify_time": verify_time,
            "total_proof_size": proof_size,
            "total_time": total_time,
            "throughput": throughput,
            "steps": len(batches),
            "success": True
        }
        
    except Exception as e:
        print(f"      ❌ Recursive SNARK Fehler: {e}")
        return {
            "type": "Recursive",
            "num_items": num_items,
            "error": str(e),
            "success": False
        }

def create_comparison_table(results: list) -> str:
    """Erstellt detaillierte Vergleichstabelle"""
    table = []
    table.append("=" * 140)
    table.append("🔬 FINAL COMPARISON: Standard vs Recursive ZK-SNARKs")
    table.append("=" * 140)
    
    # Header
    header = f"{'Items':<6} {'Type':<10} {'Setup(s)':<9} {'Proof(s)':<10} {'Verify(s)':<10} {'Size(KB)':<10} {'Throughput':<11} {'Total(s)':<9} {'Advantage':<12} {'Cost($)':<10}"
    table.append(header)
    table.append("-" * 140)
    
    # Gruppiere nach Items
    items_groups = {}
    for result in results:
        items = result["num_items"]
        if items not in items_groups:
            items_groups[items] = {"standard": None, "recursive": None}
        
        if result["type"] == "Standard":
            items_groups[items]["standard"] = result
        else:
            items_groups[items]["recursive"] = result
    
    # Erstelle Zeilen
    for items in sorted(items_groups.keys()):
        group = items_groups[items]
        
        # Standard SNARK Zeile
        std = group["standard"]
        if std and std["success"]:
            setup_time = std.get("setup_time", std.get("compile_time", 0))
            proof_time = std["total_proof_time"]
            verify_time = std["total_verify_time"]
            size_kb = std["total_proof_size"] / 1024
            throughput = std["throughput"]
            total_time = std["total_time"]
            
            # Geschätzte Kosten (Zeit * $0.001/s + Größe * $0.0001/MB)
            cost = total_time * 0.001 + (std["total_proof_size"] / (1024*1024)) * 0.0001
            
            std_line = f"{items:<6} {'Standard':<10} {setup_time:<9.3f} {proof_time:<10.3f} {verify_time:<10.3f} {size_kb:<10.1f} {throughput:<11.2f} {total_time:<9.3f} {'-':<12} {cost:<10.6f}"
            table.append(std_line)
        else:
            error_msg = std.get("error", "ERROR") if std else "NO DATA"
            std_line = f"{items:<6} {'Standard':<10} {error_msg[:9]:<9} {error_msg[:10]:<10} {error_msg[:10]:<10} {error_msg[:10]:<10} {error_msg[:11]:<11} {error_msg[:9]:<9} {error_msg[:12]:<12} {error_msg[:10]:<10}"
            table.append(std_line)
        
        # Recursive SNARK Zeile
        rec = group["recursive"]
        if rec and rec["success"]:
            setup_time = rec.get("setup_time", 0)
            proof_time = rec["total_proof_time"]
            verify_time = rec["total_verify_time"]
            size_kb = rec["total_proof_size"] / 1024
            throughput = rec["throughput"]
            total_time = rec["total_time"]
            
            # Geschätzte Kosten
            cost = total_time * 0.001 + (rec["total_proof_size"] / (1024*1024)) * 0.0001
            
            # Bestimme Advantage
            advantage = "❌ NEIN"
            if std and std["success"]:
                if rec["total_time"] < std["total_time"]:
                    speedup = std["total_time"] / rec["total_time"]
                    advantage = f"✅ {speedup:.1f}x"
                elif cost < (std["total_time"] * 0.001 + (std["total_proof_size"] / (1024*1024)) * 0.0001):
                    advantage = "💰 KOSTEN"
            
            rec_line = f"{items:<6} {'Recursive':<10} {setup_time:<9.3f} {proof_time:<10.3f} {verify_time:<10.3f} {size_kb:<10.1f} {throughput:<11.2f} {total_time:<9.3f} {advantage:<12} {cost:<10.6f}"
            table.append(rec_line)
        else:
            error_msg = rec.get("error", "ERROR") if rec else "NO DATA"
            rec_line = f"{items:<6} {'Recursive':<10} {error_msg[:9]:<9} {error_msg[:10]:<10} {error_msg[:10]:<10} {error_msg[:10]:<10} {error_msg[:11]:<11} {error_msg[:9]:<9} {error_msg[:12]:<12} {error_msg[:10]:<10}"
            table.append(rec_line)
        
        table.append("")  # Leerzeile
    
    return "\n".join(table)

def main():
    """Hauptfunktion"""
    print("🔬 FINAL COMPARISON TEST")
    print("Erstellt detaillierte Vergleichstabelle für verschiedene Datenmengen")
    print("=" * 80)
    
    # Test verschiedene Größen
    test_sizes = [6, 12, 20, 30, 50]  # Beginne mit kleineren Größen
    
    results = []
    
    for num_items in test_sizes:
        print(f"\n📊 TESTE: {num_items} Items")
        print("-" * 50)
        
        # Standard SNARK Test
        std_result = run_standard_test(num_items)
        results.append(std_result)
        
        if std_result["success"]:
            print(f"      ✅ Standard: {std_result['total_time']:.3f}s, {std_result['successful_proofs']}/{num_items} proofs")
        else:
            print(f"      ❌ Standard: {std_result.get('error', 'Unbekannter Fehler')}")
        
        # Recursive SNARK Test
        rec_result = run_recursive_test(num_items)
        results.append(rec_result)
        
        if rec_result["success"]:
            print(f"      ✅ Recursive: {rec_result['total_time']:.3f}s, {rec_result['steps']} steps")
        else:
            print(f"      ❌ Recursive: {rec_result.get('error', 'Unbekannter Fehler')}")
        
        # Kurze Pause
        time.sleep(0.5)
    
    # Erstelle und zeige Tabelle
    print("\n" + "=" * 80)
    print("📊 DETAILLIERTE VERGLEICHSTABELLE")
    print("=" * 80)
    table = create_comparison_table(results)
    print(table)
    
    # Analyse
    print("\n" + "=" * 80)
    print("🎯 ANALYSE & EMPFEHLUNGEN")
    print("=" * 80)
    
    successful_comparisons = 0
    crossover_found = False
    
    for i in range(0, len(results), 2):
        if i+1 < len(results):
            std = results[i]
            rec = results[i+1]
            
            if std["success"] and rec["success"]:
                successful_comparisons += 1
                items = std["num_items"]
                
                if rec["total_time"] < std["total_time"]:
                    speedup = std["total_time"] / rec["total_time"]
                    print(f"✅ Bei {items} Items: Recursive ist {speedup:.2f}x schneller!")
                    crossover_found = True
                else:
                    ratio = rec["total_time"] / std["total_time"]
                    print(f"⚠️  Bei {items} Items: Recursive ist {ratio:.2f}x langsamer")
    
    if not crossover_found:
        print("\n🔍 CROSSOVER POINT: Nicht in diesem Bereich gefunden")
        print("   → Teste größere Datenmengen (100+) für möglichen Crossover")
    
    print(f"\n📊 STATISTIK: {successful_comparisons} erfolgreiche Vergleiche")
    
    # Speichere Ergebnisse
    results_dir = project_root / "data" / "final_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "comparison_table.json"
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "test_sizes": test_sizes,
            "successful_comparisons": successful_comparisons,
            "crossover_found": crossover_found,
            "timestamp": time.time()
        }, f, indent=2)
    
    # Speichere auch als Text
    table_file = results_dir / "comparison_table.txt"
    with open(table_file, 'w') as f:
        f.write(table)
    
    print(f"\n💾 Ergebnisse gespeichert:")
    print(f"   📄 JSON: {results_file}")
    print(f"   📄 Tabelle: {table_file}")
    
    print("\n🎉 FINAL COMPARISON TEST ABGESCHLOSSEN!")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Test erfolgreich abgeschlossen!")
        print(f"📊 {len(results)} Tests durchgeführt")
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
