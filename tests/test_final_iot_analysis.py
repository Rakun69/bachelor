#!/usr/bin/env python3
"""
🎯 FINAL IoT ANALYSIS
Kompakte Analyse ohne komplizierte Resource-Limits
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

def monitor_memory():
    """Einfaches Memory-Monitoring"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def simple_test(num_items, test_type, max_time=120):
    """Einfacher Test mit Timeout"""
    print(f"🔬 Testing {num_items} items ({test_type}): ", end="", flush=True)
    
    start_time = time.perf_counter()
    start_memory = monitor_memory()
    max_memory = start_memory
    
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
            total_proof_size = 0
            
            for i, reading in enumerate(temp_readings):
                # Check timeout
                elapsed = time.perf_counter() - start_time
                if elapsed > max_time:
                    print(f"⏰ TIMEOUT after {elapsed:.1f}s")
                    break
                
                # Monitor memory
                current_memory = monitor_memory()
                max_memory = max(max_memory, current_memory)
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                    total_proof_size += result.metrics.proof_size
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"{total_time:.2f}s ({successful}/{num_items}) Max:{max_memory:.1f}MB")
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size / 1024,
                "max_memory_mb": max_memory,
                "timeout": elapsed > max_time
            }
            
        elif test_type == "recursive":
            nova_manager = FixedZoKratesNovaManager()
            if not nova_manager.setup():
                print("❌ Nova Setup failed")
                return {"success": False, "error": "Nova setup failed"}
            
            # Prepare batches
            batches = []
            for i in range(0, len(temp_readings), 3):
                batch_readings = temp_readings[i:i+3]
                while len(batch_readings) < 3:
                    batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
                batch_dicts = [{'value': r.value} for r in batch_readings]
                batches.append(batch_dicts)
            
            # Monitor memory during recursive proof
            for i in range(len(batches)):
                current_memory = monitor_memory()
                max_memory = max(max_memory, current_memory)
                
                # Check timeout
                elapsed = time.perf_counter() - start_time
                if elapsed > max_time:
                    print(f"⏰ TIMEOUT after {elapsed:.1f}s")
                    return {"success": False, "error": "Timeout", "total_time": elapsed}
            
            result = nova_manager.prove_recursive_batch(batches)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            if not result.success:
                print(f"❌ Failed: {result.error_message}")
                return {"success": False, "error": result.error_message, "total_time": total_time}
            
            print(f"{total_time:.2f}s ({len(batches)} steps) Max:{max_memory:.1f}MB")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "proof_size_kb": result.proof_size / 1024,
                "max_memory_mb": max_memory,
                "timeout": False
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        print(f"❌ Error: {str(e)[:50]}...")
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory
        }

def final_iot_analysis():
    """Finale IoT-Analyse ohne komplizierte Constraints"""
    print("🎯 FINAL IoT ANALYSIS")
    print("Realistische Performance-Analyse für IoT-Devices")
    print("=" * 60)
    
    # Test verschiedene Szenarien
    test_scenarios = [
        {"name": "Small IoT Batch", "items": 50},
        {"name": "Medium IoT Batch", "items": 85},
        {"name": "Large IoT Batch", "items": 100},
        {"name": "Crossover Point", "items": 89}
    ]
    
    all_results = []
    
    for scenario in test_scenarios:
        print(f"\n🔬 SCENARIO: {scenario['name']} ({scenario['items']} items)")
        print("-" * 50)
        
        # Standard test
        std_result = simple_test(scenario['items'], "standard")
        
        # Recursive test
        rec_result = simple_test(scenario['items'], "recursive")
        
        # Analyze results
        if std_result["success"] and rec_result["success"]:
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if ratio < 1.0 else "Standard"
            advantage = abs(1.0 - ratio) * 100
            
            memory_ratio = rec_result["max_memory_mb"] / std_result["max_memory_mb"]
            memory_winner = "Recursive" if memory_ratio < 1.0 else "Standard"
            
            result = {
                "scenario": scenario["name"],
                "num_items": scenario["items"],
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "time_ratio": ratio,
                "time_winner": winner,
                "time_advantage": advantage,
                "standard_memory": std_result["max_memory_mb"],
                "recursive_memory": rec_result["max_memory_mb"],
                "memory_ratio": memory_ratio,
                "memory_winner": memory_winner,
                "standard_success_rate": std_result.get("successful_proofs", 0) / scenario["items"] * 100,
                "recursive_steps": rec_result.get("steps", 0),
                "both_successful": True
            }
            
            all_results.append(result)
            
            print(f"   📊 Time: {winner} wins by {advantage:.1f}% (Ratio: {ratio:.3f})")
            print(f"   💾 Memory: {memory_winner} uses {abs(1-memory_ratio)*100:.1f}% {'less' if memory_ratio < 1 else 'more'}")
            
        else:
            # Handle failures
            result = {
                "scenario": scenario["name"],
                "num_items": scenario["items"],
                "standard_success": std_result["success"],
                "recursive_success": rec_result["success"],
                "standard_error": std_result.get("error", ""),
                "recursive_error": rec_result.get("error", ""),
                "both_successful": False
            }
            all_results.append(result)
            
            print(f"   ❌ Standard: {'✅' if std_result['success'] else '❌'}")
            print(f"   ❌ Recursive: {'✅' if rec_result['success'] else '❌'}")
    
    return all_results

def analyze_final_results(results):
    """Analysiert finale Ergebnisse"""
    print("\n" + "=" * 70)
    print("📊 FINAL IoT ANALYSIS RESULTS")
    print("=" * 70)
    
    successful_comparisons = [r for r in results if r.get("both_successful", False)]
    
    if successful_comparisons:
        print(f"\n✅ SUCCESSFUL COMPARISONS ({len(successful_comparisons)}):")
        print(f"{'Scenario':<20} {'Items':<6} {'Time Winner':<12} {'Advantage':<10} {'Memory Winner':<12}")
        print("-" * 70)
        
        for result in successful_comparisons:
            scenario = result["scenario"][:19]
            items = result["num_items"]
            time_winner = result["time_winner"]
            advantage = result["time_advantage"]
            memory_winner = result["memory_winner"]
            
            print(f"{scenario:<20} {items:<6} {time_winner:<12} {advantage:<10.1f}% {memory_winner:<12}")
    
    failed_tests = [r for r in results if not r.get("both_successful", False)]
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
        for result in failed_tests:
            print(f"   {result['scenario']}: Std={'✅' if result.get('standard_success') else '❌'}, Rec={'✅' if result.get('recursive_success') else '❌'}")
            if result.get('recursive_error'):
                print(f"      Recursive Error: {result['recursive_error'][:60]}...")
    
    # Key insights
    print(f"\n🎯 KEY INSIGHTS:")
    
    if successful_comparisons:
        recursive_wins = sum(1 for r in successful_comparisons if r["time_winner"] == "Recursive")
        crossover_result = next((r for r in successful_comparisons if "Crossover" in r["scenario"]), None)
        
        print(f"   📈 Recursive wins in {recursive_wins}/{len(successful_comparisons)} scenarios")
        
        if crossover_result:
            if crossover_result["time_winner"] == "Recursive":
                print(f"   ✅ Crossover confirmed at 89 items (Recursive {crossover_result['time_advantage']:.1f}% faster)")
            else:
                print(f"   ⚠️  Crossover not confirmed at 89 items (Standard {crossover_result['time_advantage']:.1f}% faster)")
    
    # Memory analysis
    if successful_comparisons:
        avg_memory_ratio = sum(r["memory_ratio"] for r in successful_comparisons) / len(successful_comparisons)
        if avg_memory_ratio < 1.0:
            print(f"   💾 Recursive uses {(1-avg_memory_ratio)*100:.1f}% less memory on average")
        else:
            print(f"   💾 Recursive uses {(avg_memory_ratio-1)*100:.1f}% more memory on average")
    
    # IoT suitability
    memory_efficient_scenarios = sum(1 for r in successful_comparisons if r["memory_ratio"] < 1.0)
    if memory_efficient_scenarios > len(successful_comparisons) / 2:
        print(f"   🏠 Recursive SNARKs are memory-efficient for IoT devices")
    else:
        print(f"   ⚠️  Recursive SNARKs may be memory-intensive for IoT devices")

def main():
    """Hauptfunktion"""
    print("🎯 Starting Final IoT Analysis...")
    
    results = final_iot_analysis()
    
    if results:
        analyze_final_results(results)
        
        # Save results
        results_dir = project_root / "data" / "final_iot_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "final_iot_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "final_iot_results": results,
                "test_timestamp": time.time(),
                "scenarios_tested": ["50 items", "85 items", "100 items", "89 items (crossover)"],
                "analysis_summary": {
                    "successful_comparisons": len([r for r in results if r.get("both_successful", False)]),
                    "failed_tests": len([r for r in results if not r.get("both_successful", False)])
                }
            }, f, indent=2)
        
        print(f"\n💾 Final IoT results saved: {results_file}")
        print(f"✅ FINAL IoT ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
