#!/usr/bin/env python3
"""
🔬 LARGE DATA COMPARISON TEST
Umfassender Vergleich Standard vs Recursive SNARKs für große Datenmengen
Erstellt detaillierte Tabelle mit allen wichtigen Metriken
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager
from src.evaluation.benchmark_framework import BenchmarkFramework, BenchmarkConfig

class LargeDataComparison:
    def __init__(self):
        self.sensors = SmartHomeSensors()
        self.snark_manager = SNARKManager()
        self.nova_manager = FixedZoKratesNovaManager()
        
        # Erstelle BenchmarkConfig
        benchmark_config = BenchmarkConfig(
            circuit_types=["filter_range", "iot_recursive"],
            data_sizes=[20, 50, 100, 200, 500, 1000],
            batch_sizes=[5, 10, 20, 50],
            privacy_levels=[1, 2, 3],
            iterations=1,
            output_dir="data/large_data_results"
        )
        self.benchmark = BenchmarkFramework(benchmark_config)
        
        # Test-Parameter für große Datenmengen
        self.test_sizes = [20, 50, 100, 150, 200, 300, 500, 750, 1000]
        self.batch_sizes = [5, 10, 20, 25, 50]
        
        self.results = []
        
    def calculate_costs(self, proof_time: float, verify_time: float, proof_size: int) -> Dict:
        """Berechnet geschätzte Kosten für verschiedene Szenarien"""
        # Geschätzte Kosten pro Sekunde Rechenzeit (in USD)
        compute_cost_per_second = 0.001  # $0.001 pro Sekunde
        
        # Geschätzte Kosten pro MB Storage (in USD pro Monat)
        storage_cost_per_mb_month = 0.00002  # $0.00002 pro MB/Monat
        
        # Geschätzte Netzwerk-Kosten pro MB (in USD)
        network_cost_per_mb = 0.0001  # $0.0001 pro MB
        
        proof_size_mb = proof_size / (1024 * 1024)
        
        return {
            "compute_cost": (proof_time + verify_time) * compute_cost_per_second,
            "storage_cost_monthly": proof_size_mb * storage_cost_per_mb_month,
            "network_cost": proof_size_mb * network_cost_per_mb,
            "total_cost": ((proof_time + verify_time) * compute_cost_per_second + 
                          proof_size_mb * network_cost_per_mb)
        }
    
    def run_standard_snark_test(self, num_items: int, batch_size: int) -> Dict:
        """Führt Standard SNARK Test durch"""
        print(f"   📊 Standard SNARK: {num_items} Items, Batch {batch_size}")
        
        # Generiere IoT Daten
        readings = self.sensors.generate_readings(
            sensor_type="temperature",
            num_readings=num_items,
            base_value=22.0
        )
        
        # Bereite Daten in Batches vor
        batches = [readings[i:i+batch_size] for i in range(0, len(readings), batch_size)]
        
        total_proof_time = 0
        total_verify_time = 0
        total_proof_size = 0
        total_witness_time = 0
        
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        
        try:
            # Kompiliere Circuit einmal
            compile_start = time.time()
            self.snark_manager.compile_circuit(str(circuit_path))
            compile_time = time.time() - compile_start
            
            # Setup einmal
            setup_start = time.time()
            self.snark_manager.setup_circuit()
            setup_time = time.time() - setup_start
            
            # Verarbeite jeden Batch
            for batch_idx, batch in enumerate(batches):
                for reading in batch:
                    # Verwende Temperaturwert als secret_value
                    secret_value = int(reading.value)
                    inputs = ["10", "50", str(secret_value)]  # min, max, secret
                    
                    # Witness Generation
                    witness_start = time.time()
                    self.snark_manager.generate_witness(inputs)
                    witness_time = time.time() - witness_start
                    total_witness_time += witness_time
                    
                    # Proof Generation
                    proof_start = time.time()
                    proof_result = self.snark_manager.generate_proof()
                    proof_time = time.time() - proof_start
                    total_proof_time += proof_time
                    
                    # Verification
                    verify_start = time.time()
                    is_valid = self.snark_manager.verify_proof()
                    verify_time = time.time() - verify_start
                    total_verify_time += verify_time
                    
                    # Proof Size
                    if proof_result and "proof_file" in proof_result:
                        proof_file = Path(proof_result["proof_file"])
                        if proof_file.exists():
                            total_proof_size += proof_file.stat().st_size
            
            # Berechne Durchschnittswerte
            num_proofs = len(readings)
            avg_proof_time = total_proof_time / num_proofs if num_proofs > 0 else 0
            avg_verify_time = total_verify_time / num_proofs if num_proofs > 0 else 0
            avg_proof_size = total_proof_size / num_proofs if num_proofs > 0 else 0
            
            # Berechne Kosten
            costs = self.calculate_costs(total_proof_time, total_verify_time, total_proof_size)
            
            return {
                "type": "Standard",
                "num_items": num_items,
                "batch_size": batch_size,
                "num_batches": len(batches),
                "compile_time": compile_time,
                "setup_time": setup_time,
                "total_proof_time": total_proof_time,
                "total_verify_time": total_verify_time,
                "total_witness_time": total_witness_time,
                "avg_proof_time": avg_proof_time,
                "avg_verify_time": avg_verify_time,
                "total_proof_size": total_proof_size,
                "avg_proof_size": avg_proof_size,
                "throughput": num_items / (total_proof_time + total_verify_time) if (total_proof_time + total_verify_time) > 0 else 0,
                "costs": costs,
                "success": True
            }
            
        except Exception as e:
            print(f"      ❌ Standard SNARK Fehler: {e}")
            return {
                "type": "Standard",
                "num_items": num_items,
                "batch_size": batch_size,
                "error": str(e),
                "success": False
            }
    
    def run_recursive_snark_test(self, num_items: int, batch_size: int) -> Dict:
        """Führt Recursive SNARK (Nova) Test durch"""
        print(f"   🚀 Recursive SNARK: {num_items} Items, Batch {batch_size}")
        
        # Generiere IoT Daten
        readings = self.sensors.generate_readings(
            sensor_type="temperature",
            num_readings=num_items,
            base_value=22.0
        )
        
        circuit_path = project_root / "circuits" / "nova" / "iot_recursive.zok"
        
        try:
            # Setup Nova
            setup_start = time.time()
            setup_result = self.nova_manager.setup(str(circuit_path))
            setup_time = time.time() - setup_start
            
            if not setup_result["success"]:
                raise Exception(f"Nova Setup fehlgeschlagen: {setup_result.get('error', 'Unbekannter Fehler')}")
            
            # Bereite Daten in Batches vor (3 Werte pro Batch für Nova Circuit)
            nova_batch_size = 3  # Nova Circuit erwartet 3 Werte
            batches = []
            for i in range(0, len(readings), nova_batch_size):
                batch_readings = readings[i:i+nova_batch_size]
                # Fülle auf 3 Werte auf falls nötig
                while len(batch_readings) < nova_batch_size:
                    batch_readings.append(batch_readings[-1])  # Wiederhole letzten Wert
                batches.append([int(r.value) for r in batch_readings])
            
            # Führe Recursive Proof durch
            proof_start = time.time()
            proof_result = self.nova_manager.prove_recursive_batch(batches)
            proof_time = time.time() - proof_start
            
            if not proof_result["success"]:
                raise Exception(f"Nova Proof fehlgeschlagen: {proof_result.get('error', 'Unbekannter Fehler')}")
            
            # Verification (in prove_recursive_batch enthalten)
            verify_time = proof_result.get("verify_time", 0)
            
            # Proof Size
            proof_size = proof_result.get("proof_size", 0)
            
            # Berechne Kosten
            costs = self.calculate_costs(proof_time, verify_time, proof_size)
            
            return {
                "type": "Recursive",
                "num_items": num_items,
                "batch_size": batch_size,
                "num_batches": len(batches),
                "setup_time": setup_time,
                "total_proof_time": proof_time,
                "total_verify_time": verify_time,
                "avg_proof_time": proof_time,  # Nova macht einen großen Proof
                "avg_verify_time": verify_time,
                "total_proof_size": proof_size,
                "avg_proof_size": proof_size,
                "throughput": num_items / (proof_time + verify_time) if (proof_time + verify_time) > 0 else 0,
                "costs": costs,
                "steps": len(batches),
                "success": True
            }
            
        except Exception as e:
            print(f"      ❌ Recursive SNARK Fehler: {e}")
            return {
                "type": "Recursive",
                "num_items": num_items,
                "batch_size": batch_size,
                "error": str(e),
                "success": False
            }
    
    def run_comparison_test(self, num_items: int, batch_size: int) -> Dict:
        """Führt Vergleichstest durch"""
        print(f"\n🔬 TESTE: {num_items} Items mit Batch Size {batch_size}")
        
        # Standard SNARK Test
        standard_result = self.run_standard_snark_test(num_items, batch_size)
        
        # Recursive SNARK Test
        recursive_result = self.run_recursive_snark_test(num_items, batch_size)
        
        # Vergleichsanalyse
        comparison = {
            "num_items": num_items,
            "batch_size": batch_size,
            "standard": standard_result,
            "recursive": recursive_result,
            "timestamp": time.time()
        }
        
        # Berechne Verhältnisse wenn beide erfolgreich
        if standard_result["success"] and recursive_result["success"]:
            comparison["analysis"] = {
                "time_ratio": (recursive_result["total_proof_time"] + recursive_result["total_verify_time"]) / 
                             (standard_result["total_proof_time"] + standard_result["total_verify_time"]),
                "size_ratio": recursive_result["total_proof_size"] / standard_result["total_proof_size"] if standard_result["total_proof_size"] > 0 else float('inf'),
                "cost_ratio": recursive_result["costs"]["total_cost"] / standard_result["costs"]["total_cost"] if standard_result["costs"]["total_cost"] > 0 else float('inf'),
                "throughput_ratio": recursive_result["throughput"] / standard_result["throughput"] if standard_result["throughput"] > 0 else 0,
                "recursive_advantage": (standard_result["total_proof_time"] + standard_result["total_verify_time"]) > 
                                     (recursive_result["total_proof_time"] + recursive_result["total_verify_time"])
            }
        
        return comparison
    
    def create_results_table(self) -> str:
        """Erstellt formatierte Tabelle mit allen Ergebnissen"""
        table = []
        table.append("=" * 150)
        table.append("🔬 LARGE DATA COMPARISON: Standard vs Recursive ZK-SNARKs")
        table.append("=" * 150)
        
        # Header
        header = f"{'Items':<6} {'Batch':<6} {'Type':<10} {'Setup(s)':<9} {'Proof(s)':<10} {'Verify(s)':<10} {'Size(KB)':<10} {'Throughput':<11} {'Cost($)':<10} {'Advantage':<10}"
        table.append(header)
        table.append("-" * 150)
        
        # Sortiere Ergebnisse nach Items, dann Batch Size
        sorted_results = sorted(self.results, key=lambda x: (x["num_items"], x["batch_size"]))
        
        for result in sorted_results:
            items = result["num_items"]
            batch = result["batch_size"]
            
            # Standard SNARK Zeile
            std = result["standard"]
            if std["success"]:
                setup_time = std.get("setup_time", 0)
                proof_time = std["total_proof_time"]
                verify_time = std["total_verify_time"]
                size_kb = std["total_proof_size"] / 1024
                throughput = std["throughput"]
                cost = std["costs"]["total_cost"]
                
                std_line = f"{items:<6} {batch:<6} {'Standard':<10} {setup_time:<9.3f} {proof_time:<10.3f} {verify_time:<10.3f} {size_kb:<10.1f} {throughput:<11.2f} {cost:<10.6f} {'-':<10}"
                table.append(std_line)
            else:
                std_line = f"{items:<6} {batch:<6} {'Standard':<10} {'ERROR':<9} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<11} {'ERROR':<10} {'ERROR':<10}"
                table.append(std_line)
            
            # Recursive SNARK Zeile
            rec = result["recursive"]
            if rec["success"]:
                setup_time = rec.get("setup_time", 0)
                proof_time = rec["total_proof_time"]
                verify_time = rec["total_verify_time"]
                size_kb = rec["total_proof_size"] / 1024
                throughput = rec["throughput"]
                cost = rec["costs"]["total_cost"]
                
                # Advantage bestimmen
                advantage = "❌ NEIN"
                if "analysis" in result and result["analysis"]["recursive_advantage"]:
                    advantage = "✅ JA"
                
                rec_line = f"{items:<6} {batch:<6} {'Recursive':<10} {setup_time:<9.3f} {proof_time:<10.3f} {verify_time:<10.3f} {size_kb:<10.1f} {throughput:<11.2f} {cost:<10.6f} {advantage:<10}"
                table.append(rec_line)
            else:
                rec_line = f"{items:<6} {batch:<6} {'Recursive':<10} {'ERROR':<9} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<11} {'ERROR':<10} {'ERROR':<10}"
                table.append(rec_line)
            
            table.append("")  # Leerzeile zwischen verschiedenen Item-Counts
        
        return "\n".join(table)
    
    def find_crossover_points(self) -> Dict:
        """Findet Crossover Points in den Daten"""
        crossover_analysis = {
            "time_crossover": None,
            "cost_crossover": None,
            "size_crossover": None,
            "throughput_crossover": None
        }
        
        for result in sorted(self.results, key=lambda x: x["num_items"]):
            if "analysis" in result and result["analysis"]["recursive_advantage"]:
                items = result["num_items"]
                
                # Zeit Crossover
                if crossover_analysis["time_crossover"] is None:
                    crossover_analysis["time_crossover"] = items
                
                # Kosten Crossover
                if result["analysis"]["cost_ratio"] < 1.0 and crossover_analysis["cost_crossover"] is None:
                    crossover_analysis["cost_crossover"] = items
                
                # Größe Crossover
                if result["analysis"]["size_ratio"] < 1.0 and crossover_analysis["size_crossover"] is None:
                    crossover_analysis["size_crossover"] = items
                
                # Throughput Crossover
                if result["analysis"]["throughput_ratio"] > 1.0 and crossover_analysis["throughput_crossover"] is None:
                    crossover_analysis["throughput_crossover"] = items
        
        return crossover_analysis
    
    def run_comprehensive_test(self):
        """Führt umfassenden Test durch"""
        print("🚀 STARTE LARGE DATA COMPARISON TEST")
        print("=" * 80)
        
        total_tests = len(self.test_sizes) * len(self.batch_sizes)
        current_test = 0
        
        for num_items in self.test_sizes:
            for batch_size in self.batch_sizes:
                current_test += 1
                print(f"\n📊 TEST {current_test}/{total_tests}")
                
                # Führe Vergleichstest durch
                result = self.run_comparison_test(num_items, batch_size)
                self.results.append(result)
                
                # Kurze Pause zwischen Tests
                time.sleep(1)
        
        # Erstelle Ergebnistabelle
        print("\n" + "=" * 80)
        print("📊 ERSTELLE ERGEBNISTABELLE...")
        table = self.create_results_table()
        print(table)
        
        # Finde Crossover Points
        crossover = self.find_crossover_points()
        
        print("\n" + "=" * 80)
        print("🎯 CROSSOVER POINT ANALYSE")
        print("=" * 80)
        
        if crossover["time_crossover"]:
            print(f"⏱️  Zeit Crossover bei: {crossover['time_crossover']} Items")
        else:
            print("⏱️  Zeit Crossover: Nicht gefunden (teste größere Datenmengen)")
        
        if crossover["cost_crossover"]:
            print(f"💰 Kosten Crossover bei: {crossover['cost_crossover']} Items")
        else:
            print("💰 Kosten Crossover: Nicht gefunden")
        
        if crossover["size_crossover"]:
            print(f"📦 Größe Crossover bei: {crossover['size_crossover']} Items")
        else:
            print("📦 Größe Crossover: Nicht gefunden")
        
        if crossover["throughput_crossover"]:
            print(f"🚀 Throughput Crossover bei: {crossover['throughput_crossover']} Items")
        else:
            print("🚀 Throughput Crossover: Nicht gefunden")
        
        # Speichere Ergebnisse
        results_dir = project_root / "data" / "large_data_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON Report
        json_file = results_dir / "large_data_comparison.json"
        with open(json_file, 'w') as f:
            json.dump({
                "results": self.results,
                "crossover_analysis": crossover,
                "test_parameters": {
                    "test_sizes": self.test_sizes,
                    "batch_sizes": self.batch_sizes
                },
                "timestamp": time.time()
            }, f, indent=2)
        
        # Text Report
        text_file = results_dir / "large_data_comparison.txt"
        with open(text_file, 'w') as f:
            f.write(table)
            f.write(f"\n\nCrossover Analysis:\n{crossover}")
        
        print(f"\n💾 Ergebnisse gespeichert:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📄 Text: {text_file}")
        
        print("\n🎉 LARGE DATA COMPARISON TEST ABGESCHLOSSEN!")
        return self.results, crossover

def main():
    """Hauptfunktion"""
    print("🔬 LARGE DATA COMPARISON TEST")
    print("Vergleicht Standard vs Recursive SNARKs für große Datenmengen")
    print("=" * 80)
    
    try:
        comparison = LargeDataComparison()
        results, crossover = comparison.run_comprehensive_test()
        
        print("\n✅ Test erfolgreich abgeschlossen!")
        print(f"📊 {len(results)} Vergleichstests durchgeführt")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
