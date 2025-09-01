#!/usr/bin/env python3
"""
🔧 BATCH SIZE OPTIMIZATION
Findet optimale Batch-Größe für Recursive SNARKs
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

def test_batch_size(num_items, batch_size):
    """Testet Recursive SNARKs mit spezifischer Batch-Größe"""
    print(f"🔬 Testing {num_items} items with batch size {batch_size}: ", end="", flush=True)
    
    start_time = time.perf_counter()
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        if not nova_manager.setup():
            print("❌ Nova Setup failed")
            return None
        
        # Generate data
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Prepare batches with custom batch size
        batches = []
        for i in range(0, len(temp_readings), batch_size):
            batch_readings = temp_readings[i:i+batch_size]
            
            # Pad batch to required size
            while len(batch_readings) < batch_size:
                batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
            
            batch_dicts = [{'value': r.value} for r in batch_readings]
            batches.append(batch_dicts)
        
        # Execute recursive proof
        result = nova_manager.prove_recursive_batch(batches)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        if result.success:
            throughput = num_items / total_time
            efficiency = throughput / batch_size  # Items per second per batch size
            
            print(f"{total_time:.2f}s ({len(batches)} batches, {throughput:.1f} items/s)")
            
            return {
                "success": True,
                "num_items": num_items,
                "batch_size": batch_size,
                "total_time": total_time,
                "num_batches": len(batches),
                "throughput": throughput,
                "efficiency": efficiency,
                "proof_size_kb": result.proof_size / 1024,
                "verify_time": result.verify_time
            }
        else:
            print(f"❌ Failed: {result.error_message}")
            return None
            
    except Exception as e:
        end_time = time.perf_counter()
        print(f"❌ Error: {str(e)[:50]}...")
        return None

def batch_size_optimization():
    """Optimiert Batch-Größe für verschiedene Szenarien"""
    print("🔧 BATCH SIZE OPTIMIZATION")
    print("Findet optimale Batch-Größe für Recursive SNARKs")
    print("=" * 60)
    
    # Test verschiedene Batch-Größen
    batch_sizes = [1, 2, 3, 5, 8, 10]
    test_scenarios = [
        {"name": "Small Dataset", "items": 50},
        {"name": "Medium Dataset", "items": 100},
        {"name": "Large Dataset", "items": 150}
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n🔬 SCENARIO: {scenario['name']} ({scenario['items']} items)")
        print("-" * 50)
        
        scenario_results = []
        
        for batch_size in batch_sizes:
            result = test_batch_size(scenario['items'], batch_size)
            
            if result:
                scenario_results.append(result)
        
        all_results[scenario['name']] = scenario_results
        
        # Find optimal batch size for this scenario
        if scenario_results:
            best_result = min(scenario_results, key=lambda x: x['total_time'])
            most_efficient = max(scenario_results, key=lambda x: x['efficiency'])
            
            print(f"\n   🏆 Fastest: Batch size {best_result['batch_size']} ({best_result['total_time']:.2f}s)")
            print(f"   ⚡ Most Efficient: Batch size {most_efficient['batch_size']} ({most_efficient['efficiency']:.2f} items/s/batch)")
    
    return all_results

def analyze_batch_optimization(results):
    """Analysiert Batch-Size-Optimierung"""
    print("\n" + "=" * 70)
    print("📊 BATCH SIZE OPTIMIZATION RESULTS")
    print("=" * 70)
    
    for scenario_name, scenario_results in results.items():
        if not scenario_results:
            continue
            
        print(f"\n🔬 {scenario_name.upper()}:")
        print(f"{'Batch':<6} {'Time(s)':<8} {'Batches':<8} {'Throughput':<12} {'Efficiency':<12}")
        print("-" * 60)
        
        best_time = float('inf')
        best_batch_size = None
        best_efficiency = 0
        most_efficient_batch = None
        
        for result in scenario_results:
            batch_size = result['batch_size']
            total_time = result['total_time']
            num_batches = result['num_batches']
            throughput = result['throughput']
            efficiency = result['efficiency']
            
            print(f"{batch_size:<6} {total_time:<8.2f} {num_batches:<8} {throughput:<12.1f} {efficiency:<12.2f}")
            
            if total_time < best_time:
                best_time = total_time
                best_batch_size = batch_size
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                most_efficient_batch = batch_size
        
        print(f"   🏆 Fastest: Batch size {best_batch_size} ({best_time:.2f}s)")
        print(f"   ⚡ Most Efficient: Batch size {most_efficient_batch} ({best_efficiency:.2f})")
    
    # Overall recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    all_results_flat = []
    for scenario_results in results.values():
        all_results_flat.extend(scenario_results)
    
    if all_results_flat:
        # Find most commonly optimal batch sizes
        batch_performance = {}
        for result in all_results_flat:
            batch_size = result['batch_size']
            if batch_size not in batch_performance:
                batch_performance[batch_size] = []
            batch_performance[batch_size].append(result['efficiency'])
        
        # Calculate average efficiency per batch size
        avg_efficiency = {}
        for batch_size, efficiencies in batch_performance.items():
            avg_efficiency[batch_size] = sum(efficiencies) / len(efficiencies)
        
        best_overall = max(avg_efficiency.items(), key=lambda x: x[1])
        
        print(f"   📈 Overall Best Batch Size: {best_overall[0]} (avg efficiency: {best_overall[1]:.2f})")
        
        # Current default is 3 - compare
        current_default = avg_efficiency.get(3, 0)
        if best_overall[0] != 3:
            improvement = ((best_overall[1] - current_default) / current_default) * 100
            print(f"   🚀 Improvement over current default (3): {improvement:.1f}%")
        else:
            print(f"   ✅ Current default (3) is optimal!")

def main():
    """Hauptfunktion"""
    results = batch_size_optimization()
    
    if results:
        analyze_batch_optimization(results)
        
        # Save results
        results_dir = project_root / "data" / "batch_size_optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "batch_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "batch_optimization_results": results,
                "test_timestamp": time.time(),
                "batch_sizes_tested": [1, 2, 3, 5, 8, 10],
                "scenarios": ["50 items", "100 items", "150 items"]
            }, f, indent=2)
        
        print(f"\n💾 Batch optimization results saved: {results_file}")
        print(f"✅ BATCH SIZE OPTIMIZATION COMPLETE!")
        
        return True
    else:
        print("❌ No batch optimization results")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)