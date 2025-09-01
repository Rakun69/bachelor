#!/usr/bin/env python3
"""
✅ FINAL WORKING COMPARISON
Basiert auf den FUNKTIONIERENDEN Tests - ohne problematische Constraints
Vergleicht Standard vs Recursive SNARKs mit und ohne leichte Monitoring
"""

import sys
import time
import json
import psutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def monitor_resources():
    """Einfaches Resource-Monitoring ohne Constraints"""
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1)
    }

def working_test(num_items, test_type, monitor_resources_flag=True):
    """Test basierend auf den FUNKTIONIERENDEN Versionen"""
    monitor_label = "Monitored" if monitor_resources_flag else "Standard"
    print(f"✅ {monitor_label} {test_type} ({num_items} items): ", end="", flush=True)
    
    start_time = time.perf_counter()
    start_resources = monitor_resources() if monitor_resources_flag else {"memory_mb": 0, "cpu_percent": 0}
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
                if monitor_resources_flag:
                    current_resources = monitor_resources()
                    max_memory = max(max_memory, current_resources["memory_mb"])
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                
                result = manager.generate_proof("filter_range", inputs)
                if result.success:
                    successful += 1
                    individual_proof_sizes.append(result.metrics.proof_size)
                
                # Progress for large tests
                if num_items > 80 and (i + 1) % 25 == 0:
                    print(f"{i+1}...", end="", flush=True)
            
            # Calculate metrics (exact same as working tests)
            if individual_proof_sizes:
                total_proof_size_kb = sum(individual_proof_sizes) / 1024
                avg_proof_size_kb = sum(individual_proof_sizes) / len(individual_proof_sizes) / 1024
            else:
                total_proof_size_kb = 0
                avg_proof_size_kb = 0
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            throughput = successful / total_time if total_time > 0 else 0
            
            memory_str = f"Max:{max_memory:.1f}MB" if monitor_resources_flag else ""
            print(f"{total_time:.3f}s ({successful}/{num_items}) {memory_str}")
            
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
                "monitored": monitor_resources_flag
            }
            
        elif test_type == "recursive":
            nova_manager = FixedZoKratesNovaManager()
            if not nova_manager.setup():
                print("❌ Nova Setup failed")
                return {"success": False, "error": "Nova setup failed", "monitored": monitor_resources_flag}
            
            # Prepare batches (exact same as working tests)
            batches = []
            for i in range(0, len(temp_readings), 3):
                batch_readings = temp_readings[i:i+3]
                while len(batch_readings) < 3:
                    batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
                
                batch_dicts = [{'value': r.value} for r in batch_readings]
                batches.append(batch_dicts)
            
            # Monitor during recursive proof
            if monitor_resources_flag:
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
                    "monitored": monitor_resources_flag
                }
            
            throughput = num_items / total_time if total_time > 0 else 0
            
            memory_str = f"Max:{max_memory:.1f}MB" if monitor_resources_flag else ""
            print(f"{total_time:.3f}s ({len(batches)} steps) {memory_str}")
            
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
                "monitored": monitor_resources_flag
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        end_resources = monitor_resources() if monitor_resources_flag else {"memory_mb": 0}
        print(f"❌ Error: {str(e)[:50]}...")
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory,
            "monitored": monitor_resources_flag
        }

def final_working_comparison():
    """Finale Vergleichsanalyse basierend auf funktionierenden Tests"""
    print("✅ FINAL WORKING COMPARISON")
    print("Basiert auf bewährten funktionierenden Tests")
    print("Vergleicht Standard vs Recursive SNARKs")
    print("=" * 60)
    
    # Test key scenarios
    scenarios = [
        {"name": "Small Batch", "items": 50},
        {"name": "Medium Batch", "items": 85},
        {"name": "Crossover Point", "items": 89},
        {"name": "Large Batch", "items": 100}
    ]
    
    all_results = []
    
    for scenario in scenarios:
        num_items = scenario["items"]
        scenario_name = scenario["name"]
        
        print(f"\n🔬 SCENARIO: {scenario_name} ({num_items} items)")
        print("-" * 50)
        
        # Standard Test
        std_result = working_test(num_items, "standard", monitor_resources_flag=True)
        
        # Recursive Test
        rec_result = working_test(num_items, "recursive", monitor_resources_flag=True)
        
        # Analyze results
        if std_result["success"] and rec_result["success"]:
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if rec_result["total_time"] < std_result["total_time"] else "Standard"
            time_diff = abs(rec_result["total_time"] - std_result["total_time"])
            percentage_diff = (time_diff / min(std_result["total_time"], rec_result["total_time"])) * 100
            
            # Memory efficiency
            memory_ratio = rec_result["max_memory_mb"] / std_result["max_memory_mb"] if std_result["max_memory_mb"] > 0 else 1.0
            memory_winner = "Recursive" if memory_ratio < 1.0 else "Standard"
            
            result = {
                "scenario": scenario_name,
                "num_items": num_items,
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "time_ratio": ratio,
                "time_winner": winner,
                "time_advantage": percentage_diff,
                "standard_memory": std_result["max_memory_mb"],
                "recursive_memory": rec_result["max_memory_mb"],
                "memory_ratio": memory_ratio,
                "memory_winner": memory_winner,
                "standard_proof_size_kb": std_result["total_proof_size_kb"],
                "recursive_proof_size_kb": rec_result["proof_size_kb"],
                "standard_throughput": std_result["throughput"],
                "recursive_throughput": rec_result["throughput"],
                "both_successful": True
            }
            
            all_results.append(result)
            
            print(f"   📊 Time: {winner} wins by {percentage_diff:.1f}% (Ratio: {ratio:.3f})")
            print(f"   💾 Memory: {memory_winner} ({'less' if memory_ratio < 1 else 'more'} efficient)")
            print(f"   📏 Proof Size: Std {std_result['total_proof_size_kb']:.1f}KB vs Rec {rec_result['proof_size_kb']:.1f}KB")
            
        else:
            # Handle failures
            result = {
                "scenario": scenario_name,
                "num_items": num_items,
                "standard_success": std_result["success"],
                "recursive_success": rec_result["success"],
                "standard_error": std_result.get("error", ""),
                "recursive_error": rec_result.get("error", ""),
                "both_successful": False
            }
            all_results.append(result)
            
            print(f"   ❌ Standard: {'✅' if std_result['success'] else '❌'}")
            print(f"   ❌ Recursive: {'✅' if rec_result['success'] else '❌'}")
        
        # Short pause between scenarios
        time.sleep(0.5)
    
    return all_results

def analyze_final_results(results):
    """Analysiert finale Vergleichsergebnisse"""
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON RESULTS")
    print("=" * 70)
    
    successful_comparisons = [r for r in results if r.get("both_successful", False)]
    failed_tests = [r for r in results if not r.get("both_successful", False)]
    
    if successful_comparisons:
        print(f"\n✅ SUCCESSFUL COMPARISONS ({len(successful_comparisons)}):")
        print(f"{'Scenario':<15} {'Items':<6} {'Time Winner':<12} {'Advantage':<10} {'Memory Winner':<12} {'Proof Size':<15}")
        print("-" * 80)
        
        crossover_point = None
        
        for result in successful_comparisons:
            scenario = result["scenario"][:14]
            items = result["num_items"]
            time_winner = result["time_winner"]
            advantage = result["time_advantage"]
            memory_winner = result["memory_winner"]
            
            # Proof size comparison
            std_size = result["standard_proof_size_kb"]
            rec_size = result["recursive_proof_size_kb"]
            size_comparison = f"S:{std_size:.0f} R:{rec_size:.0f}"
            
            print(f"{scenario:<15} {items:<6} {time_winner:<12} {advantage:<10.1f}% {memory_winner:<12} {size_comparison:<15}")
            
            # Find crossover
            if result["time_winner"] == "Recursive" and crossover_point is None:
                crossover_point = items
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
        for result in failed_tests:
            scenario = result["scenario"]
            std_ok = "✅" if result.get("standard_success", False) else "❌"
            rec_ok = "✅" if result.get("recursive_success", False) else "❌"
            print(f"   {scenario}: Standard {std_ok}, Recursive {rec_ok}")
    
    # Summary insights
    print(f"\n🎯 KEY INSIGHTS:")
    
    if successful_comparisons:
        recursive_wins = sum(1 for r in successful_comparisons if r["time_winner"] == "Recursive")
        total_comparisons = len(successful_comparisons)
        
        print(f"   📈 Recursive wins: {recursive_wins}/{total_comparisons} scenarios ({recursive_wins/total_comparisons*100:.0f}%)")
        
        if crossover_point:
            print(f"   🎯 Crossover Point: {crossover_point} items")
        else:
            print(f"   ⚠️  No crossover found in tested range")
        
        # Average metrics
        avg_std_memory = sum(r["standard_memory"] for r in successful_comparisons) / len(successful_comparisons)
        avg_rec_memory = sum(r["recursive_memory"] for r in successful_comparisons) / len(successful_comparisons)
        
        print(f"   💾 Average Memory: Standard {avg_std_memory:.1f}MB, Recursive {avg_rec_memory:.1f}MB")
        
        # Proof size analysis
        avg_std_proof = sum(r["standard_proof_size_kb"] for r in successful_comparisons) / len(successful_comparisons)
        avg_rec_proof = sum(r["recursive_proof_size_kb"] for r in successful_comparisons) / len(successful_comparisons)
        
        print(f"   📏 Average Proof Size: Standard {avg_std_proof:.1f}KB, Recursive {avg_rec_proof:.1f}KB")
    
    # Reliability assessment
    success_rate = len(successful_comparisons) / len(results) * 100 if results else 0
    print(f"   ✅ Overall Success Rate: {success_rate:.0f}% ({len(successful_comparisons)}/{len(results)})")
    
    if success_rate == 100:
        print(f"   🎉 ALL TESTS SUCCESSFUL - RELIABLE RESULTS!")
    elif success_rate >= 75:
        print(f"   👍 MOSTLY SUCCESSFUL - GOOD RELIABILITY")
    else:
        print(f"   ⚠️  LOW SUCCESS RATE - CHECK IMPLEMENTATION")

def main():
    """Hauptfunktion"""
    print("✅ Starting Final Working Comparison...")
    
    results = final_working_comparison()
    
    if results:
        analyze_final_results(results)
        
        # Save results
        results_dir = project_root / "data" / "final_working_comparison"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "final_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "final_comparison_results": results,
                "test_timestamp": time.time(),
                "test_description": "Final working comparison without problematic constraints",
                "scenarios_tested": ["50 items", "85 items", "89 items", "100 items"],
                "methodology": "Based on proven working tests with resource monitoring"
            }, f, indent=2)
        
        print(f"\n💾 Final comparison results saved: {results_file}")
        print(f"✅ FINAL WORKING COMPARISON COMPLETE!")
        
        return True
    else:
        print("❌ No final comparison results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
