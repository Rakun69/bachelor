#!/usr/bin/env python3
"""
🎯 RESOURCE-CONSTRAINED CROSSOVER ANALYSIS
Vergleicht Crossover-Point mit und ohne IoT-Resource-Limits
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

def apply_iot_constraints(memory_mb=256, cpu_nice=15):
    """Wendet IoT-ähnliche Resource-Constraints an"""
    try:
        # Memory limit (soft limit in bytes)
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes * 2))
        
        # CPU priority (higher nice = lower priority)
        os.nice(cpu_nice)
        
        print(f"   🔒 IoT Constraints applied: {memory_mb}MB RAM, Nice +{cpu_nice}")
        return True
        
    except Exception as e:
        print(f"   ⚠️  Could not apply constraints: {e}")
        return False

def monitor_resources():
    """Überwacht aktuelle Ressourcennutzung"""
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1)
    }

def constrained_test(num_items, test_type, apply_constraints=True):
    """Test mit optionalen Resource-Constraints"""
    constraint_label = "Constrained" if apply_constraints else "Unconstrained"
    print(f"🔬 {constraint_label} {test_type} ({num_items} items): ", end="", flush=True)
    
    # Apply constraints if requested
    if apply_constraints:
        apply_iot_constraints(memory_mb=512, cpu_nice=10)  # Moderate constraints
    
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
            total_proof_size = 0
            
            for i, reading in enumerate(temp_readings):
                # Monitor resources
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                # Check for memory issues
                if apply_constraints and current_resources["memory_mb"] > 500:  # Near limit
                    print(f"⚠️ Memory warning: {current_resources['memory_mb']:.1f}MB")
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                    total_proof_size += result.metrics.proof_size
                
                # Progress for large tests
                if num_items > 80 and (i + 1) % 20 == 0:
                    print(f"{i+1}/{num_items}...", end="", flush=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"{total_time:.2f}s ({successful}/{num_items}) Max:{max_memory:.1f}MB")
            
            return {
                "success": True,
                "type": "standard",
                "constrained": apply_constraints,
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size / 1024,
                "max_memory_mb": max_memory,
                "success_rate": successful / num_items * 100
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
            
            # Monitor during recursive proof
            for i in range(len(batches)):
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                if apply_constraints and current_resources["memory_mb"] > 500:
                    print(f"⚠️ Memory warning: {current_resources['memory_mb']:.1f}MB")
            
            result = nova_manager.prove_recursive_batch(batches)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            if not result.success:
                print(f"❌ Failed: {result.error_message}")
                return {
                    "success": False,
                    "error": result.error_message,
                    "constrained": apply_constraints,
                    "total_time": total_time,
                    "max_memory_mb": max_memory
                }
            
            print(f"{total_time:.2f}s ({len(batches)} steps) Max:{max_memory:.1f}MB")
            
            return {
                "success": True,
                "type": "recursive",
                "constrained": apply_constraints,
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "proof_size_kb": result.proof_size / 1024,
                "max_memory_mb": max_memory
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        print(f"❌ Error: {str(e)[:50]}...")
        return {
            "success": False,
            "error": str(e),
            "constrained": apply_constraints,
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory
        }

def resource_constrained_crossover_analysis():
    """Vergleicht Crossover mit und ohne Resource-Constraints"""
    print("🎯 RESOURCE-CONSTRAINED CROSSOVER ANALYSIS")
    print("Vergleicht Performance mit und ohne IoT-Resource-Limits")
    print("=" * 70)
    
    # Test items around suspected crossover point
    test_items = [80, 85, 90, 95, 100, 105]
    
    all_results = {
        "unconstrained": [],
        "constrained": []
    }
    
    for constraint_mode in ["unconstrained", "constrained"]:
        apply_constraints = (constraint_mode == "constrained")
        
        print(f"\n🔬 TESTING: {constraint_mode.upper()} MODE")
        print("-" * 50)
        
        mode_results = []
        
        for num_items in test_items:
            print(f"\n📊 Testing {num_items} items:")
            
            # Standard test
            std_result = constrained_test(num_items, "standard", apply_constraints)
            
            # Recursive test
            rec_result = constrained_test(num_items, "recursive", apply_constraints)
            
            # Analyze if both successful
            if std_result["success"] and rec_result["success"]:
                ratio = rec_result["total_time"] / std_result["total_time"]
                winner = "Recursive" if ratio < 1.0 else "Standard"
                advantage = abs(1.0 - ratio) * 100
                
                result = {
                    "num_items": num_items,
                    "constrained": apply_constraints,
                    "standard_time": std_result["total_time"],
                    "recursive_time": rec_result["total_time"],
                    "ratio": ratio,
                    "winner": winner,
                    "advantage": advantage,
                    "standard_memory": std_result["max_memory_mb"],
                    "recursive_memory": rec_result["max_memory_mb"],
                    "both_successful": True
                }
                
                mode_results.append(result)
                print(f"   📊 {winner} wins by {advantage:.1f}%")
                
            else:
                # Handle failures
                result = {
                    "num_items": num_items,
                    "constrained": apply_constraints,
                    "standard_success": std_result["success"],
                    "recursive_success": rec_result["success"],
                    "standard_error": std_result.get("error", ""),
                    "recursive_error": rec_result.get("error", ""),
                    "both_successful": False
                }
                mode_results.append(result)
                print(f"   ❌ Std: {'✅' if std_result['success'] else '❌'}, Rec: {'✅' if rec_result['success'] else '❌'}")
        
        all_results[constraint_mode] = mode_results
    
    return all_results

def analyze_constraint_impact(results):
    """Analysiert den Einfluss von Resource-Constraints"""
    print("\n" + "=" * 80)
    print("📊 RESOURCE CONSTRAINT IMPACT ANALYSIS")
    print("=" * 80)
    
    unconstrained = results["unconstrained"]
    constrained = results["constrained"]
    
    # Compare crossover points
    print(f"\n🎯 CROSSOVER POINT COMPARISON:")
    
    for mode_name, mode_results in [("Unconstrained", unconstrained), ("Constrained", constrained)]:
        successful = [r for r in mode_results if r.get("both_successful", False)]
        
        if successful:
            crossover_point = None
            for result in successful:
                if result["winner"] == "Recursive":
                    crossover_point = result["num_items"]
                    break
            
            print(f"   {mode_name}: {crossover_point if crossover_point else 'No crossover found'}")
        else:
            print(f"   {mode_name}: No successful comparisons")
    
    # Detailed comparison table
    print(f"\n📊 DETAILED COMPARISON:")
    print(f"{'Items':<6} {'Unconstrained Winner':<18} {'Constrained Winner':<18} {'Impact':<15}")
    print("-" * 70)
    
    constraint_impact_count = 0
    
    for i, num_items in enumerate([80, 85, 90, 95, 100, 105]):
        if i < len(unconstrained) and i < len(constrained):
            unc_result = unconstrained[i]
            con_result = constrained[i]
            
            unc_winner = unc_result.get("winner", "Failed")
            con_winner = con_result.get("winner", "Failed")
            
            if unc_winner != "Failed" and con_winner != "Failed":
                if unc_winner != con_winner:
                    impact = "CHANGED"
                    constraint_impact_count += 1
                else:
                    impact = "Same"
            else:
                impact = "Failed"
            
            print(f"{num_items:<6} {unc_winner:<18} {con_winner:<18} {impact:<15}")
    
    # Summary insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    if constraint_impact_count > 0:
        print(f"   🚨 Resource constraints changed winner in {constraint_impact_count} cases!")
        print(f"   📈 This shows IoT constraints significantly impact performance!")
    else:
        print(f"   ✅ Resource constraints did not change winners")
        print(f"   📊 Performance scales similarly under constraints")
    
    # Memory impact
    unc_successful = [r for r in unconstrained if r.get("both_successful", False)]
    con_successful = [r for r in constrained if r.get("both_successful", False)]
    
    if unc_successful and con_successful:
        unc_avg_memory = sum(r["standard_memory"] + r["recursive_memory"] for r in unc_successful) / len(unc_successful) / 2
        con_avg_memory = sum(r["standard_memory"] + r["recursive_memory"] for r in con_successful) / len(con_successful) / 2
        
        memory_reduction = ((unc_avg_memory - con_avg_memory) / unc_avg_memory) * 100
        
        if memory_reduction > 5:
            print(f"   💾 Constraints reduced memory usage by {memory_reduction:.1f}%")
        else:
            print(f"   💾 Memory usage similar under constraints ({memory_reduction:.1f}% difference)")

def main():
    """Hauptfunktion"""
    print("🎯 Starting Resource-Constrained Crossover Analysis...")
    
    results = resource_constrained_crossover_analysis()
    
    if results:
        analyze_constraint_impact(results)
        
        # Save results
        results_dir = project_root / "data" / "resource_constrained_crossover"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "constraint_impact_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "constraint_impact_results": results,
                "test_timestamp": time.time(),
                "test_items": [80, 85, 90, 95, 100, 105],
                "constraint_settings": {
                    "memory_limit_mb": 512,
                    "cpu_nice_value": 10
                }
            }, f, indent=2)
        
        print(f"\n💾 Constraint impact results saved: {results_file}")
        print(f"✅ RESOURCE-CONSTRAINED ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No constraint impact results")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
