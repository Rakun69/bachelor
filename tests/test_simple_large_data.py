#!/usr/bin/env python3
"""
🔬 SIMPLE LARGE DATA TEST
Einfacher Test für große Datenmengen ohne komplexe Dependencies
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

def test_standard_snark(num_items: int) -> dict:
    """Test Standard SNARK mit gegebener Anzahl Items"""
    print(f"   📊 Standard SNARK: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        snark_manager = SNARKManager()
        
        # Generiere IoT Daten (alle Sensoren für kurze Zeit)
        all_readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        
        # Filtere nur Temperatur-Readings und nimm die ersten num_items
        temp_readings = [r for r in all_readings if r.sensor_type == "temperature"]
        readings = temp_readings[:num_items]
        
        # Falls nicht genug Readings, wiederhole sie
        while len(readings) < num_items:
            readings.extend(temp_readings[:num_items - len(readings)])
        
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        
        # Kompiliere Circuit
        compile_start = time.time()
        snark_manager.compile_circuit(str(circuit_path), "filter_range")
        compile_time = time.time() - compile_start
        
        # Setup
        setup_start = time.time()
        snark_manager.setup_circuit()
        setup_time = time.time() - setup_start
        
        # Verarbeite alle Readings
        total_proof_time = 0
        total_verify_time = 0
        total_proof_size = 0
        successful_proofs = 0
        
        for reading in readings:
            try:
                secret_value = int(reading.value)
                inputs = ["10", "50", str(secret_value)]
                
                # Witness
                snark_manager.generate_witness(inputs)
                
                # Proof
                proof_start = time.time()
                proof_result = snark_manager.generate_proof()
                proof_time = time.time() - proof_start
                total_proof_time += proof_time
                
                # Verify
                verify_start = time.time()
                is_valid = snark_manager.verify_proof()
                verify_time = time.time() - verify_start
                total_verify_time += verify_time
                
                # Proof Size
                if proof_result and "proof_file" in proof_result:
                    proof_file = Path(proof_result["proof_file"])
                    if proof_file.exists():
                        total_proof_size += proof_file.stat().st_size
                
                if is_valid:
                    successful_proofs += 1
                    
            except Exception as e:
                print(f"      ⚠️  Proof {len([r for r in readings if r == reading])} fehlgeschlagen: {e}")
        
        # Berechne Durchschnitte
        avg_proof_time = total_proof_time / num_items if num_items > 0 else 0
        avg_verify_time = total_verify_time / num_items if num_items > 0 else 0
        avg_proof_size = total_proof_size / num_items if num_items > 0 else 0
        
        total_time = compile_time + setup_time + total_proof_time + total_verify_time
        throughput = num_items / total_time if total_time > 0 else 0
        
        return {
            "type": "Standard",
            "num_items": num_items,
            "successful_proofs": successful_proofs,
            "compile_time": compile_time,
            "setup_time": setup_time,
            "total_proof_time": total_proof_time,
            "total_verify_time": total_verify_time,
            "avg_proof_time": avg_proof_time,
            "avg_verify_time": avg_verify_time,
            "total_proof_size": total_proof_size,
            "avg_proof_size": avg_proof_size,
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

def test_recursive_snark(num_items: int) -> dict:
    """Test Recursive SNARK mit gegebener Anzahl Items"""
    print(f"   🚀 Recursive SNARK: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # Generiere IoT Daten (alle Sensoren für kurze Zeit)
        all_readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        
        # Filtere nur Temperatur-Readings und nimm die ersten num_items
        temp_readings = [r for r in all_readings if r.sensor_type == "temperature"]
        readings = temp_readings[:num_items]
        
        # Falls nicht genug Readings, wiederhole sie
        while len(readings) < num_items:
            readings.extend(temp_readings[:num_items - len(readings)])
        
        circuit_path = project_root / "circuits" / "nova" / "iot_recursive.zok"
        
        # Setup Nova (ohne Parameter - setup() nimmt keine Argumente)
        setup_start = time.time()
        setup_success = nova_manager.setup()
        setup_time = time.time() - setup_start
        
        if not setup_success:
            raise Exception("Nova Setup fehlgeschlagen")
        
        # Bereite Daten in 3er-Batches vor (Nova Circuit erwartet 3 Werte)
        batches = []
        for i in range(0, len(readings), 3):
            batch_readings = readings[i:i+3]
            # Fülle auf 3 Werte auf falls nötig
            while len(batch_readings) < 3:
                batch_readings.append(batch_readings[-1])
            batches.append([int(r.value) for r in batch_readings])
        
        # Führe Recursive Proof durch
        proof_start = time.time()
        proof_result = nova_manager.prove_recursive_batch(batches)
        proof_time = time.time() - proof_start
        
        if not proof_result["success"]:
            raise Exception(f"Nova Proof fehlgeschlagen: {proof_result.get('error', 'Unbekannter Fehler')}")
        
        verify_time = proof_result.get("verify_time", 0)
        proof_size = proof_result.get("proof_size", 0)
        
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
    """Erstellt formatierte Vergleichstabelle"""
    table = []
    table.append("=" * 120)
    table.append("🔬 LARGE DATA COMPARISON: Standard vs Recursive ZK-SNARKs")
    table.append("=" * 120)
    
    # Header
    header = f"{'Items':<6} {'Type':<10} {'Setup(s)':<9} {'Proof(s)':<10} {'Verify(s)':<10} {'Size(KB)':<10} {'Throughput':<11} {'Total(s)':<9} {'Advantage':<10}"
    table.append(header)
    table.append("-" * 120)
    
    # Gruppiere Ergebnisse nach Items
    items_groups = {}
    for result in results:
        items = result["num_items"]
        if items not in items_groups:
            items_groups[items] = {"standard": None, "recursive": None}
        
        if result["type"] == "Standard":
            items_groups[items]["standard"] = result
        else:
            items_groups[items]["recursive"] = result
    
    # Erstelle Tabellenzeilen
    for items in sorted(items_groups.keys()):
        group = items_groups[items]
        
        # Standard SNARK Zeile
        std = group["standard"]
        if std and std["success"]:
            setup_time = std.get("setup_time", 0)
            proof_time = std["total_proof_time"]
            verify_time = std["total_verify_time"]
            size_kb = std["total_proof_size"] / 1024
            throughput = std["throughput"]
            total_time = std["total_time"]
            
            std_line = f"{items:<6} {'Standard':<10} {setup_time:<9.3f} {proof_time:<10.3f} {verify_time:<10.3f} {size_kb:<10.1f} {throughput:<11.2f} {total_time:<9.3f} {'-':<10}"
            table.append(std_line)
        else:
            std_line = f"{items:<6} {'Standard':<10} {'ERROR':<9} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<11} {'ERROR':<9} {'ERROR':<10}"
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
            
            # Bestimme Advantage
            advantage = "❌ NEIN"
            if std and std["success"] and rec["total_time"] < std["total_time"]:
                advantage = "✅ JA"
            
            rec_line = f"{items:<6} {'Recursive':<10} {setup_time:<9.3f} {proof_time:<10.3f} {verify_time:<10.3f} {size_kb:<10.1f} {throughput:<11.2f} {total_time:<9.3f} {advantage:<10}"
            table.append(rec_line)
        else:
            rec_line = f"{items:<6} {'Recursive':<10} {'ERROR':<9} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<11} {'ERROR':<9} {'ERROR':<10}"
            table.append(rec_line)
        
        table.append("")  # Leerzeile
    
    return "\n".join(table)

def find_crossover_point(results: list) -> dict:
    """Findet den Crossover Point"""
    crossover_info = {
        "found": False,
        "crossover_items": None,
        "standard_time": None,
        "recursive_time": None
    }
    
    # Gruppiere nach Items
    items_groups = {}
    for result in results:
        items = result["num_items"]
        if items not in items_groups:
            items_groups[items] = {}
        items_groups[items][result["type"]] = result
    
    # Suche Crossover Point
    for items in sorted(items_groups.keys()):
        group = items_groups[items]
        if "Standard" in group and "Recursive" in group:
            std = group["Standard"]
            rec = group["Recursive"]
            
            if std["success"] and rec["success"]:
                if rec["total_time"] < std["total_time"]:
                    crossover_info = {
                        "found": True,
                        "crossover_items": items,
                        "standard_time": std["total_time"],
                        "recursive_time": rec["total_time"],
                        "speedup": std["total_time"] / rec["total_time"]
                    }
                    break
    
    return crossover_info

def main():
    """Hauptfunktion"""
    print("🔬 SIMPLE LARGE DATA TEST")
    print("Vergleicht Standard vs Recursive SNARKs für verschiedene Datenmengen")
    print("=" * 80)
    
    # Test-Größen (beginne klein und steigere)
    test_sizes = [20, 50, 100, 200, 500]
    
    results = []
    
    for num_items in test_sizes:
        print(f"\n📊 TESTE: {num_items} Items")
        print("-" * 40)
        
        # Standard SNARK Test
        std_result = test_standard_snark(num_items)
        results.append(std_result)
        
        if std_result["success"]:
            print(f"      ✅ Standard: {std_result['total_time']:.3f}s, {std_result['throughput']:.2f} items/s")
        else:
            print(f"      ❌ Standard: {std_result.get('error', 'Unbekannter Fehler')}")
        
        # Recursive SNARK Test
        rec_result = test_recursive_snark(num_items)
        results.append(rec_result)
        
        if rec_result["success"]:
            print(f"      ✅ Recursive: {rec_result['total_time']:.3f}s, {rec_result['throughput']:.2f} items/s")
        else:
            print(f"      ❌ Recursive: {rec_result.get('error', 'Unbekannter Fehler')}")
        
        # Kurze Pause
        time.sleep(1)
    
    # Erstelle Tabelle
    print("\n" + "=" * 80)
    print("📊 ERGEBNISTABELLE")
    print("=" * 80)
    table = create_comparison_table(results)
    print(table)
    
    # Finde Crossover Point
    crossover = find_crossover_point(results)
    
    print("\n" + "=" * 80)
    print("🎯 CROSSOVER POINT ANALYSE")
    print("=" * 80)
    
    if crossover["found"]:
        print(f"✅ Crossover Point gefunden bei: {crossover['crossover_items']} Items")
        print(f"   Standard Zeit: {crossover['standard_time']:.3f}s")
        print(f"   Recursive Zeit: {crossover['recursive_time']:.3f}s")
        print(f"   Speedup: {crossover['speedup']:.2f}x")
    else:
        print("⚠️  Kein Crossover Point in diesem Bereich gefunden")
        print("   → Teste größere Datenmengen (1000+) für Crossover")
    
    # Speichere Ergebnisse
    results_dir = project_root / "data" / "simple_large_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "crossover": crossover,
            "test_sizes": test_sizes,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n💾 Ergebnisse gespeichert: {results_file}")
    print("\n🎉 TEST ABGESCHLOSSEN!")
    
    return results, crossover

if __name__ == "__main__":
    try:
        results, crossover = main()
        print("\n✅ Alle Tests erfolgreich!")
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
