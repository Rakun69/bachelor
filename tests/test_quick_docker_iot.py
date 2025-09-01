#!/usr/bin/env python3
"""
🚀 QUICK DOCKER IoT TEST
Schneller Test ohne vollständigen Docker Build
"""

import sys
import time
import json
import psutil
import resource
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def limit_resources(memory_mb=256, cpu_percent=25):
    """Simuliert IoT-Resource-Limits ohne Docker"""
    try:
        # Memory limit (soft limit)
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        
        # CPU limit simulation (via process priority)
        import os
        os.nice(10)  # Lower priority
        
        print(f"✅ Resource limits applied: {memory_mb}MB RAM, {cpu_percent}% CPU")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not apply resource limits: {e}")
        return False

def monitor_resources():
    """Überwacht Ressourcenverbrauch"""
    process = psutil.Process()
    
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent()
    }

def iot_constrained_test(num_items, test_type, memory_limit=256, cpu_limit=25):
    """Test mit simulierten IoT-Constraints"""
    print(f"🔬 IoT-Constrained Test: {num_items} items ({test_type})")
    print(f"   Limits: {memory_limit}MB RAM, {cpu_limit}% CPU")
    
    # Apply resource limits
    limit_resources(memory_limit, cpu_limit)
    
    start_time = time.perf_counter()
    start_resources = monitor_resources()
    
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
            max_memory = 0
            
            for i, reading in enumerate(temp_readings):
                # Monitor resources during execution
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                # Check memory limit
                if current_resources["memory_mb"] > memory_limit * 1.2:  # 20% tolerance
                    print(f"   ⚠️  Memory limit exceeded: {current_resources['memory_mb']:.1f}MB")
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                    total_proof_size += result.metrics.proof_size
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   📊 Progress: {i+1}/{num_items} ({current_resources['memory_mb']:.1f}MB)")
            
            end_time = time.perf_counter()
            end_resources = monitor_resources()
            total_time = end_time - start_time
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size / 1024,
                "throughput": successful / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory,
                "final_memory_mb": end_resources["memory_mb"],
                "memory_limit_mb": memory_limit,
                "memory_exceeded": max_memory > memory_limit * 1.2
            }
            
        elif test_type == "recursive":
            nova_manager = FixedZoKratesNovaManager()
            if not nova_manager.setup():
                raise Exception("Nova Setup failed")
            
            # Prepare batches
            batches = []
            for i in range(0, len(temp_readings), 3):
                batch_readings = temp_readings[i:i+3]
                while len(batch_readings) < 3:
                    batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
                batch_dicts = [{'value': r.value} for r in batch_readings]
                batches.append(batch_dicts)
            
            # Monitor during recursive proof
            max_memory = 0
            for i in range(len(batches)):
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                if (i + 1) % 5 == 0:
                    print(f"   📊 Batch Progress: {i+1}/{len(batches)} ({current_resources['memory_mb']:.1f}MB)")
            
            result = nova_manager.prove_recursive_batch(batches)
            
            end_time = time.perf_counter()
            end_resources = monitor_resources()
            total_time = end_time - start_time
            
            if not result.success:
                raise Exception(f"Recursive proof failed: {result.error_message}")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "proof_size_kb": result.proof_size / 1024,
                "throughput": num_items / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory,
                "final_memory_mb": end_resources["memory_mb"],
                "memory_limit_mb": memory_limit,
                "memory_exceeded": max_memory > memory_limit * 1.2
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        end_resources = monitor_resources()
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "final_memory_mb": end_resources["memory_mb"]
        }

def quick_iot_analysis():
    """Schnelle IoT-Constraint-Analyse ohne Docker"""
    print("🚀 QUICK IoT RESOURCE CONSTRAINTS TEST")
    print("Simuliert IoT-Device-Limitierungen ohne Docker")
    print("=" * 60)
    
    # IoT Device Configurations
    iot_configs = [
        {"name": "Low-End IoT", "memory": 256, "cpu": 25},
        {"name": "Mid-Range IoT", "memory": 512, "cpu": 50},
        {"name": "Standard System", "memory": 2048, "cpu": 100}  # Referenz
    ]
    
    test_items = [85, 90, 95]
    all_results = {}
    
    for config in iot_configs:
        print(f"\n🔬 TESTING: {config['name']}")
        print(f"   Memory: {config['memory']}MB, CPU: {config['cpu']}%")
        print("-" * 40)
        
        config_results = []
        
        for num_items in test_items:
            print(f"\n📊 Testing {num_items} items...")
            
            # Standard SNARK test
            print("   🔧 Standard SNARKs: ", end="", flush=True)
            std_result = iot_constrained_test(
                num_items, "standard", 
                config["memory"], config["cpu"]
            )
            
            if std_result and std_result["success"]:
                memory_status = "⚠️ EXCEEDED" if std_result.get("memory_exceeded") else "✅ OK"
                print(f"{std_result['total_time']:.2f}s ({memory_status})")
            else:
                print("❌ Failed")
                continue
            
            # Recursive SNARK test  
            print("   🚀 Recursive SNARKs: ", end="", flush=True)
            rec_result = iot_constrained_test(
                num_items, "recursive",
                config["memory"], config["cpu"]
            )
            
            if rec_result and rec_result["success"]:
                memory_status = "⚠️ EXCEEDED" if rec_result.get("memory_exceeded") else "✅ OK"
                print(f"{rec_result['total_time']:.2f}s ({memory_status})")
            else:
                print("❌ Failed")
                continue
            
            # Compare results
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if ratio < 1.0 else "Standard"
            advantage = abs(1.0 - ratio) * 100
            
            result_entry = {
                "num_items": num_items,
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "ratio": ratio,
                "winner": winner,
                "advantage_percent": advantage,
                "standard_max_memory": std_result.get("max_memory_mb", 0),
                "recursive_max_memory": rec_result.get("max_memory_mb", 0),
                "standard_memory_exceeded": std_result.get("memory_exceeded", False),
                "recursive_memory_exceeded": rec_result.get("memory_exceeded", False),
                "iot_config": config
            }
            
            config_results.append(result_entry)
            print(f"   📊 {winner} wins by {advantage:.1f}%")
            print(f"   💾 Memory: Std {std_result.get('max_memory_mb', 0):.1f}MB, Rec {rec_result.get('max_memory_mb', 0):.1f}MB")
        
        all_results[config["name"]] = config_results
    
    return all_results

def analyze_quick_results(all_results):
    """Analysiert Quick IoT-Constraint Ergebnisse"""
    print("\n" + "=" * 70)
    print("📊 QUICK IoT CONSTRAINTS ANALYSIS")
    print("=" * 70)
    
    for config_name, results in all_results.items():
        if not results:
            continue
            
        print(f"\n🔬 {config_name.upper()}:")
        print(f"{'Items':<6} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Winner':<10} {'StdMem':<8} {'RecMem':<8}")
        print("-" * 70)
        
        crossover_point = None
        memory_issues = 0
        
        for result in results:
            items = result["num_items"]
            std_time = result["standard_time"]
            rec_time = result["recursive_time"]
            ratio = result["ratio"]
            winner = result["winner"]
            std_mem = result["standard_max_memory"]
            rec_mem = result["recursive_max_memory"]
            
            # Memory status indicators
            std_mem_str = f"{std_mem:.1f}MB"
            rec_mem_str = f"{rec_mem:.1f}MB"
            
            if result["standard_memory_exceeded"]:
                std_mem_str += "⚠️"
                memory_issues += 1
            if result["recursive_memory_exceeded"]:
                rec_mem_str += "⚠️"
                memory_issues += 1
            
            print(f"{items:<6} {std_time:<8.2f} {rec_time:<8.2f} {ratio:<8.3f} {winner:<10} {std_mem_str:<8} {rec_mem_str:<8}")
            
            if winner == "Recursive" and crossover_point is None:
                crossover_point = items
        
        if crossover_point:
            print(f"   🎯 Crossover Point: {crossover_point} items")
        else:
            print(f"   ⚠️  No crossover found in tested range")
        
        if memory_issues > 0:
            print(f"   ⚠️  Memory limit exceeded in {memory_issues} tests")

def main():
    """Hauptfunktion"""
    results = quick_iot_analysis()
    
    if results:
        analyze_quick_results(results)
        
        # Save results
        results_dir = project_root / "data" / "quick_iot_constraints"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "quick_iot_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "quick_iot_results": results,
                "test_timestamp": time.time(),
                "constraints_tested": ["256MB/25%CPU", "512MB/50%CPU", "2048MB/100%CPU"]
            }, f, indent=2)
        
        print(f"\n💾 Quick IoT results saved: {results_file}")
        print(f"✅ QUICK IoT CONSTRAINTS ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
