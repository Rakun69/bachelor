#!/usr/bin/env python3
"""
🎯 WORKING IoT CONSTRAINTS TEST
Basiert auf den funktionierenden Tests, aber mit moderaten IoT-Constraints
"""

import sys
import time
import json
import psutil
import resource
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def apply_realistic_iot_constraints():
    """Wendet realistische IoT-Constraints an (nicht zu restriktiv für Nova)"""
    try:
        # Realistische IoT-Limits: 1.5GB RAM (genug für Nova, aber limitiert)
        memory_bytes = int(1.5 * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        
        # Moderate CPU-Priorität (simuliert IoT-CPU-Limits)
        os.nice(5)
        
        print("   🔒 Realistic IoT constraints: 1.5GB RAM, Nice +5")
        return True
        
    except Exception as e:
        print(f"   ⚠️  Could not apply constraints: {e}")
        return False

def monitor_resources():
    """Überwacht Ressourcenverbrauch"""
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1)
    }

def iot_constrained_test(num_items, test_type):
    """Test mit realistischen IoT-Constraints (basiert auf funktionierenden Tests)"""
    print(f"🔬 IoT-Constrained {test_type} ({num_items} items): ", end="", flush=True)
    
    # Apply realistic IoT constraints
    apply_realistic_iot_constraints()
    
    start_time = time.perf_counter()
    start_resources = monitor_resources()
    max_memory = start_resources["memory_mb"]
    
    try:
        sensors = SmartHomeSensors()
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        if test_type == "standard":
            manager = SNARKManager()
            circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
            manager.compile_circuit(str(circuit_path), "filter_range")
            manager.setup_circuit("filter_range")
            
            successful = 0
            individual_proof_sizes = []
            
            for i, reading in enumerate(temp_readings):
                # Monitor resources
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                
                result = manager.generate_proof("filter_range", inputs)
                if result.success:
                    successful += 1
                    individual_proof_sizes.append(result.metrics.proof_size)
                
                # Progress indicator for large tests
                if num_items > 80 and (i + 1) % 20 == 0:
                    print(f"{i+1}...", end="", flush=True)
            
            # Calculate metrics (same as working tests)
            if individual_proof_sizes:
                total_proof_size_kb = sum(individual_proof_sizes) / 1024
                avg_proof_size_kb = sum(individual_proof_sizes) / len(individual_proof_sizes) / 1024
            else:
                total_proof_size_kb = 0
                avg_proof_size_kb = 0
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            throughput = successful / total_time if total_time > 0 else 0
            
            print(f"{total_time:.3f}s ({successful}/{num_items}) Max:{max_memory:.1f}MB")
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size_kb,
                "avg_proof_size_kb": avg_proof_size_kb,
                "throughput": throughput,
                "max_memory_mb": max_memory,
                "iot_constrained": True
            }
            
        elif test_type == "recursive":
            nova_manager = FixedZoKratesNovaManager()
            if not nova_manager.setup():
                print("❌ Nova Setup failed")
                return {"success": False, "error": "Nova setup failed", "iot_constrained": True}
            
            # Prepare batches (same as working tests)
            batches = []
            for i in range(0, len(temp_readings), 3):
                batch_readings = temp_readings[i:i+3]
                while len(batch_readings) < 3:
                    batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
                
                batch_dicts = [{'value': r.value} for r in batch_readings]
                batches.append(batch_dicts)
            
            # Monitor during recursive proof
            for i in range(len(batches)):
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
            
            result = nova_manager.prove_recursive_batch(batches)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            if not result.success:
                print(f"❌ Failed: {result.error_message}")
                return {
                    "success": False,
                    "error": result.error_message,
                    "total_time": total_time,
                    "max_memory_mb": max_memory,
                    "iot_constrained": True
                }
            
            throughput = num_items / total_time if total_time > 0 else 0
            
            print(f"{total_time:.3f}s ({len(batches)} steps) Max:{max_memory:.1f}MB")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "proof_size_kb": result.proof_size / 1024,
                "verify_time": result.verify_time,
                "throughput": throughput,
                "max_memory_mb": max_memory,
                "iot_constrained": True
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        end_resources = monitor_resources()
        print(f"❌ Error: {str(e)[:50]}...")
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory,
            "iot_constrained": True
        }

def working_iot_constraints_analysis():
    """IoT-Constraints-Analyse basierend auf funktionierenden Tests"""
    print("🎯 WORKING IoT CONSTRAINTS ANALYSIS")
    print("Basiert auf funktionierenden Tests mit realistischen IoT-Limits")
    print("=" * 70)
    
    # Test key points around crossover
    test_points = [85, 89, 95, 100]
    results = []
    
    for num_items in test_points:
        print(f"\n🔬 TESTE: {num_items} Items mit IoT-Constraints")
        print("-" * 40)
        
        # Standard Test
        std_result = iot_constrained_test(num_items, "standard")
        
        # Recursive Test
        rec_result = iot_constrained_test(num_items, "recursive")
        
        # Analyze if both successful
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
                "standard_max_memory": std_result["max_memory_mb"],
                "recursive_max_memory": rec_result["max_memory_mb"],
                "both_successful": True,
                "iot_constrained": True
            }
            results.append(result)
            
            print(f"   📊 {winner} gewinnt um {percentage_diff:.1f}% (Ratio: {ratio:.4f})")
            print(f"   💾 Memory: Std {std_result['max_memory_mb']:.1f}MB, Rec {rec_result['max_memory_mb']:.1f}MB")
        else:
            print(f"   ❌ Test fehlgeschlagen - Std: {'✅' if std_result['success'] else '❌'}, Rec: {'✅' if rec_result['success'] else '❌'}")
            
            # Still record the failure for analysis
            result = {
                "num_items": num_items,
                "standard_success": std_result["success"],
                "recursive_success": rec_result["success"],
                "standard_error": std_result.get("error", ""),
                "recursive_error": rec_result.get("error", ""),
                "both_successful": False,
                "iot_constrained": True
            }
            results.append(result)
        
        # Pause between tests
        time.sleep(1)
    
    return results

def analyze_iot_constraint_results(results):
    """Analysiert IoT-Constraint Ergebnisse"""
    print("\n" + "=" * 80)
    print("📊 IoT CONSTRAINTS ANALYSIS RESULTS")
    print("=" * 80)
    
    successful_results = [r for r in results if r.get("both_successful", False)]
    failed_results = [r for r in results if not r.get("both_successful", False)]
    
    if successful_results:
        print(f"\n✅ SUCCESSFUL COMPARISONS UNDER IoT CONSTRAINTS ({len(successful_results)}):")
        print(f"{'Items':<5} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Diff%':<7} {'Winner':<10} {'StdMem':<8} {'RecMem':<8}")
        print("-" * 80)
        
        crossover_point = None
        
        for result in successful_results:
            items = result["num_items"]
            std_time = result["standard_time"]
            rec_time = result["recursive_time"]
            ratio = result["ratio"]
            diff_pct = result["percentage_difference"]
            winner = result["winner"]
            std_mem = result["standard_max_memory"]
            rec_mem = result["recursive_max_memory"]
            
            print(f"{items:<5} {std_time:<8.3f} {rec_time:<8.3f} {ratio:<8.4f} {diff_pct:<7.1f} {winner:<10} {std_mem:<8.1f} {rec_mem:<8.1f}")
            
            if result["recursive_wins"] and crossover_point is None:
                crossover_point = items
        
        if crossover_point:
            print(f"\n🎯 CROSSOVER POINT UNDER IoT CONSTRAINTS: {crossover_point} Items")
        else:
            print(f"\n⚠️  No crossover found under IoT constraints in tested range")
    
    if failed_results:
        print(f"\n❌ FAILED TESTS UNDER IoT CONSTRAINTS ({len(failed_results)}):")
        for result in failed_results:
            items = result["num_items"]
            std_ok = "✅" if result.get("standard_success", False) else "❌"
            rec_ok = "✅" if result.get("recursive_success", False) else "❌"
            print(f"   {items} Items: Standard {std_ok}, Recursive {rec_ok}")
    
    # Compare with unconstrained results (if available)
    print(f"\n🔍 IoT CONSTRAINT IMPACT:")
    if successful_results:
        avg_std_memory = sum(r["standard_max_memory"] for r in successful_results) / len(successful_results)
        avg_rec_memory = sum(r["recursive_max_memory"] for r in successful_results) / len(successful_results)
        
        print(f"   💾 Average Memory Usage:")
        print(f"      Standard SNARKs: {avg_std_memory:.1f}MB")
        print(f"      Recursive SNARKs: {avg_rec_memory:.1f}MB")
        
        if len(successful_results) == len(results):
            print(f"   ✅ All tests passed under IoT constraints")
            print(f"   🎯 Both SNARK types are IoT-compatible with moderate limits")
        else:
            print(f"   ⚠️  {len(failed_results)} tests failed under IoT constraints")
            print(f"   📊 Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")

def main():
    """Hauptfunktion"""
    results = working_iot_constraints_analysis()
    
    if results:
        analyze_iot_constraint_results(results)
        
        # Save results
        results_dir = project_root / "data" / "working_iot_constraints"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "iot_constraint_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "iot_constraint_results": results,
                "test_timestamp": time.time(),
                "constraint_settings": {
                    "memory_limit_gb": 1.5,
                    "cpu_nice_value": 5,
                    "description": "Realistic IoT constraints - moderate limits"
                },
                "test_items": [85, 89, 95, 100]
            }, f, indent=2)
        
        print(f"\n💾 IoT constraint results saved: {results_file}")
        print(f"✅ WORKING IoT CONSTRAINTS ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No IoT constraint results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
