#!/usr/bin/env python3
"""
⚡ QUICK CONSTRAINT CHECK
Schneller Test um zu prüfen ob Resource-Constraints funktionieren
"""

import sys
import time
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

def apply_light_constraints():
    """Wendet leichte Constraints an (nicht zu restriktiv)"""
    try:
        # Moderate memory limit (1GB)
        memory_bytes = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes * 2))
        
        # Light CPU priority reduction
        os.nice(5)
        
        print("✅ Light constraints applied: 1GB RAM, Nice +5")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not apply constraints: {e}")
        return False

def monitor_resources():
    """Überwacht Ressourcen"""
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1)
    }

def quick_test(test_type, num_items=20, apply_constraints=False):
    """Sehr schneller Test mit wenigen Items"""
    constraint_label = "Constrained" if apply_constraints else "Normal"
    print(f"⚡ Quick {constraint_label} {test_type} ({num_items} items): ", end="", flush=True)
    
    if apply_constraints:
        apply_light_constraints()
    
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
            
            # Test nur 3 Proofs für Geschwindigkeit
            successful = 0
            for i, reading in enumerate(temp_readings[:3]):
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                
                print(f"{i+1}...", end="", flush=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            end_resources = monitor_resources()
            
            print(f" {total_time:.2f}s ({successful}/3) {end_resources['memory_mb']:.1f}MB")
            
            return {
                "success": True,
                "type": "standard",
                "constrained": apply_constraints,
                "total_time": total_time,
                "successful_proofs": successful,
                "memory_mb": end_resources["memory_mb"]
            }
            
        elif test_type == "recursive":
            nova_manager = FixedZoKratesNovaManager()
            
            print("setup...", end="", flush=True)
            if not nova_manager.setup():
                print(" ❌ Setup failed")
                return {"success": False, "error": "Nova setup failed", "constrained": apply_constraints}
            
            # Nur 2 kleine Batches für Geschwindigkeit
            batches = [
                [{'value': 25.0}, {'value': 26.0}, {'value': 27.0}],
                [{'value': 28.0}, {'value': 29.0}, {'value': 30.0}]
            ]
            
            print("prove...", end="", flush=True)
            result = nova_manager.prove_recursive_batch(batches)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            end_resources = monitor_resources()
            
            if result.success:
                print(f" {total_time:.2f}s (2 batches) {end_resources['memory_mb']:.1f}MB")
                return {
                    "success": True,
                    "type": "recursive",
                    "constrained": apply_constraints,
                    "total_time": total_time,
                    "steps": len(batches),
                    "memory_mb": end_resources["memory_mb"]
                }
            else:
                print(f" ❌ Failed: {result.error_message}")
                return {
                    "success": False,
                    "error": result.error_message,
                    "constrained": apply_constraints,
                    "total_time": total_time,
                    "memory_mb": end_resources["memory_mb"]
                }
            
    except Exception as e:
        end_time = time.perf_counter()
        end_resources = monitor_resources()
        print(f" ❌ Error: {str(e)[:30]}...")
        return {
            "success": False,
            "error": str(e),
            "constrained": apply_constraints,
            "total_time": end_time - start_time,
            "memory_mb": end_resources["memory_mb"]
        }

def quick_constraint_check():
    """Schneller Check ob Constraints funktionieren"""
    print("⚡ QUICK CONSTRAINT CHECK")
    print("Schneller Test um Constraint-Funktionalität zu prüfen")
    print("=" * 60)
    
    results = []
    
    # Test 1: Normal Standard
    print("\n🔬 TEST 1: Normal Standard SNARKs")
    std_normal = quick_test("standard", num_items=20, apply_constraints=False)
    results.append(std_normal)
    
    # Test 2: Constrained Standard
    print("\n🔬 TEST 2: Constrained Standard SNARKs")
    std_constrained = quick_test("standard", num_items=20, apply_constraints=True)
    results.append(std_constrained)
    
    # Test 3: Normal Recursive
    print("\n🔬 TEST 3: Normal Recursive SNARKs")
    rec_normal = quick_test("recursive", num_items=6, apply_constraints=False)
    results.append(rec_normal)
    
    # Test 4: Constrained Recursive
    print("\n🔬 TEST 4: Constrained Recursive SNARKs")
    rec_constrained = quick_test("recursive", num_items=6, apply_constraints=True)
    results.append(rec_constrained)
    
    return results

def analyze_quick_results(results):
    """Analysiert Quick-Test Ergebnisse"""
    print("\n" + "=" * 60)
    print("📊 QUICK TEST RESULTS")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r["success"])
    
    print(f"\n✅ SUCCESSFUL TESTS: {success_count}/4")
    
    for i, result in enumerate(results, 1):
        test_names = [
            "Normal Standard",
            "Constrained Standard", 
            "Normal Recursive",
            "Constrained Recursive"
        ]
        
        status = "✅" if result["success"] else "❌"
        time_str = f"{result.get('total_time', 0):.2f}s" if result["success"] else "Failed"
        memory_str = f"{result.get('memory_mb', 0):.1f}MB"
        
        print(f"   {status} Test {i} ({test_names[i-1]}): {time_str}, {memory_str}")
        
        if not result["success"]:
            error = result.get("error", "Unknown error")
            print(f"      Error: {error[:60]}...")
    
    # Check if constraints work
    std_normal = results[0] if len(results) > 0 else None
    std_constrained = results[1] if len(results) > 1 else None
    rec_normal = results[2] if len(results) > 2 else None
    rec_constrained = results[3] if len(results) > 3 else None
    
    print(f"\n🔍 CONSTRAINT IMPACT:")
    
    if std_normal and std_normal["success"] and std_constrained and std_constrained["success"]:
        std_impact = ((std_constrained["total_time"] - std_normal["total_time"]) / std_normal["total_time"]) * 100
        print(f"   📊 Standard SNARKs: {std_impact:+.1f}% time change under constraints")
    else:
        print(f"   ❌ Standard SNARKs: Could not compare (one failed)")
    
    if rec_normal and rec_normal["success"] and rec_constrained and rec_constrained["success"]:
        rec_impact = ((rec_constrained["total_time"] - rec_normal["total_time"]) / rec_normal["total_time"]) * 100
        print(f"   📊 Recursive SNARKs: {rec_impact:+.1f}% time change under constraints")
    elif rec_normal and rec_normal["success"] and rec_constrained and not rec_constrained["success"]:
        print(f"   🚨 Recursive SNARKs: FAILED under constraints!")
    elif not rec_normal or not rec_normal["success"]:
        print(f"   ❌ Recursive SNARKs: Failed even without constraints")
    else:
        print(f"   ❌ Recursive SNARKs: Could not compare")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if success_count == 4:
        print(f"   ✅ All tests passed - constraints work correctly")
        print(f"   🚀 Ready for full constraint analysis")
    elif success_count >= 2:
        print(f"   ⚠️  Some tests failed - investigate constraint settings")
        print(f"   🔧 Consider lighter constraints or fix Nova issues")
    else:
        print(f"   🚨 Most tests failed - check basic functionality first")
        print(f"   🛠️  Fix core issues before testing constraints")

def main():
    """Hauptfunktion"""
    print("⚡ Starting Quick Constraint Check...")
    
    results = quick_constraint_check()
    
    if results:
        analyze_quick_results(results)
        print(f"\n✅ QUICK CHECK COMPLETE! (Total time: ~30-60 seconds)")
        return True
    else:
        print("❌ No quick check results")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
