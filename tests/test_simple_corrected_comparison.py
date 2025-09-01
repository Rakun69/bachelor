#!/usr/bin/env python3
"""
🎯 SIMPLE CORRECTED COMPARISON TEST
Direkter Test ohne Docker - zeigt korrigierte Ergebnisse sofort
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

def test_corrected_data_generation():
    """Testet die korrigierte Daten-Generierung"""
    print("🔧 Testing corrected IoT data generation...")
    
    sensors = SmartHomeSensors()
    readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
    temp_readings = [r for r in readings if r.sensor_type == "temperature"]
    
    print(f"📊 Total readings generated: {len(readings)}")
    print(f"📊 Temperature readings: {len(temp_readings)}")
    
    # Count by sensor
    temp_by_sensor = {}
    for reading in temp_readings:
        sensor_id = reading.sensor_id
        if sensor_id not in temp_by_sensor:
            temp_by_sensor[sensor_id] = 0
        temp_by_sensor[sensor_id] += 1
    
    print(f"📊 Temperature sensors found: {len(temp_by_sensor)}")
    for sensor_id, count in temp_by_sensor.items():
        print(f"   {sensor_id}: {count} readings")
    
    expected_per_sensor = 60  # 1 hour / 60 seconds = 60 readings per sensor
    expected_total = len(temp_by_sensor) * expected_per_sensor
    
    print(f"\n✅ Expected: {len(temp_by_sensor)} sensors × {expected_per_sensor} readings = {expected_total} total")
    print(f"✅ Actual: {len(temp_readings)} total")
    print(f"✅ Match: {'YES' if len(temp_readings) == expected_total else 'NO'}")
    
    return len(temp_readings) >= 500  # Should have at least 500 for good testing

def simple_standard_test(num_items):
    """Einfacher Standard SNARK Test"""
    print(f"\n🔧 Simple Standard SNARK Test: {num_items} items")
    
    try:
        # Generate data
        sensors = SmartHomeSensors()
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"]
        
        if len(temp_readings) < num_items:
            print(f"⚠️  Only {len(temp_readings)} readings available, using all")
            num_items = len(temp_readings)
        
        temp_readings = temp_readings[:num_items]
        
        # Setup SNARK manager
        manager = SNARKManager()
        circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
        
        start_time = time.perf_counter()
        
        # Compile and setup
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Generate proofs
        successful = 0
        total_proof_size = 0
        
        for i, reading in enumerate(temp_readings):
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            result = manager.generate_proof("filter_range", inputs)
            
            if result.success:
                successful += 1
                total_proof_size += result.metrics.proof_size
            
            # Show progress
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{num_items} ({(i+1)/num_items*100:.1f}%)")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print(f"\n📊 Standard SNARK Results:")
        print(f"   ✅ Successful: {successful}/{num_items}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   ⚡ Throughput: {successful/total_time:.2f} proofs/sec")
        print(f"   📏 Total proof size: {total_proof_size/1024:.2f}KB")
        print(f"   📏 Avg proof size: {total_proof_size/successful/1024:.3f}KB per proof")
        
        return {
            "success": True,
            "type": "standard",
            "num_items": num_items,
            "successful_proofs": successful,
            "total_time": total_time,
            "total_proof_size_kb": total_proof_size / 1024,
            "avg_proof_size_kb": total_proof_size / successful / 1024 if successful > 0 else 0,
            "throughput": successful / total_time if total_time > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Standard test failed: {e}")
        return {"success": False, "error": str(e)}

def simple_recursive_test(num_items):
    """Einfacher Recursive SNARK Test"""
    print(f"\n🚀 Simple Recursive SNARK Test: {num_items} items")
    
    try:
        # Generate data
        sensors = SmartHomeSensors()
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"]
        
        if len(temp_readings) < num_items:
            print(f"⚠️  Only {len(temp_readings)} readings available, using all")
            num_items = len(temp_readings)
        
        temp_readings = temp_readings[:num_items]
        
        # Setup Nova manager
        nova_manager = FixedZoKratesNovaManager()
        
        start_time = time.perf_counter()
        
        if not nova_manager.setup():
            raise Exception("Nova setup failed")
        
        # Prepare batches
        batches = []
        for i in range(0, len(temp_readings), 3):
            batch_readings = temp_readings[i:i+3]
            while len(batch_readings) < 3:
                batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
            batch_dicts = [{'value': r.value} for r in batch_readings]
            batches.append(batch_dicts)
        
        print(f"   📦 Prepared {len(batches)} batches for {num_items} items")
        
        # Generate recursive proof
        result = nova_manager.prove_recursive_batch(batches)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        if result.success:
            print(f"\n📊 Recursive SNARK Results:")
            print(f"   ✅ Recursive proof successful")
            print(f"   🔄 Steps: {len(batches)}")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   ⚡ Throughput: {num_items/total_time:.2f} items/sec")
            print(f"   📏 Proof size: {result.proof_size/1024:.2f}KB (CONSTANT!)")
            print(f"   ✅ Verify time: {result.verify_time:.3f}s")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "proof_size_kb": result.proof_size / 1024,
                "verify_time": result.verify_time,
                "throughput": num_items / total_time if total_time > 0 else 0
            }
        else:
            raise Exception(f"Recursive proof failed: {result.error_message}")
        
    except Exception as e:
        print(f"❌ Recursive test failed: {e}")
        return {"success": False, "error": str(e)}

def compare_results(std_result, rec_result):
    """Vergleicht Standard vs Recursive Ergebnisse"""
    print(f"\n" + "="*60)
    print("📊 CORRECTED COMPARISON RESULTS")
    print("="*60)
    
    if std_result.get("success") and rec_result.get("success"):
        std_time = std_result["total_time"]
        rec_time = rec_result["total_time"]
        
        std_size = std_result["total_proof_size_kb"]
        rec_size = rec_result["proof_size_kb"]
        
        time_ratio = rec_time / std_time
        size_ratio = rec_size / std_size if std_size > 0 else 0
        
        print(f"⏱️  TIME COMPARISON:")
        print(f"   Standard: {std_time:.2f}s")
        print(f"   Recursive: {rec_time:.2f}s")
        print(f"   Ratio: {time_ratio:.2f}x ({'Recursive faster' if time_ratio < 1 else 'Standard faster'})")
        
        print(f"\n📏 SIZE COMPARISON:")
        print(f"   Standard: {std_size:.2f}KB")
        print(f"   Recursive: {rec_size:.2f}KB")
        print(f"   Ratio: {size_ratio:.2f}x ({'Recursive smaller' if size_ratio < 1 else 'Standard smaller'})")
        
        print(f"\n🎯 CROSSOVER ANALYSIS:")
        if size_ratio < 1:
            print(f"   📏 Recursive WINS on proof size ({size_ratio:.2f}x smaller)")
        if time_ratio < 1:
            print(f"   ⏱️  Recursive WINS on time ({time_ratio:.2f}x faster)")
        
        if time_ratio >= 1 and size_ratio < 1:
            print(f"   📊 Trade-off: Recursive slower but much smaller proofs")
            print(f"   🎯 Recursive better for: Storage, Network, Verification")
            print(f"   🎯 Standard better for: Generation time")
        
        return {
            "time_ratio": time_ratio,
            "size_ratio": size_ratio,
            "recursive_time_winner": time_ratio < 1,
            "recursive_size_winner": size_ratio < 1
        }
    else:
        print("❌ Cannot compare - one or both tests failed")
        return None

def main():
    """Hauptfunktion für einfachen korrigierten Test"""
    print("🎯 SIMPLE CORRECTED COMPARISON TEST")
    print("Testing corrected IoT simulation and fair comparisons")
    print("="*60)
    
    # Test data generation first
    if not test_corrected_data_generation():
        print("❌ Data generation test failed")
        return False
    
    # Test with moderate item count
    num_items = 200
    
    print(f"\n🎯 Running comparison test with {num_items} items...")
    
    # Run both tests
    std_result = simple_standard_test(num_items)
    rec_result = simple_recursive_test(num_items)
    
    # Compare results
    comparison = compare_results(std_result, rec_result)
    
    # Save results
    results_dir = project_root / "data" / "simple_corrected_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "simple_corrected_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "standard_result": std_result,
            "recursive_result": rec_result,
            "comparison": comparison,
            "test_timestamp": time.time(),
            "corrections_applied": [
                "Fixed IoT simulation to generate 600 temperature readings per hour",
                "Added 5 additional temperature sensors",
                "Fixed outdoor sensor update frequency",
                "Ensured fair comparison with same item counts"
            ]
        }, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print("✅ SIMPLE CORRECTED TEST COMPLETE!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
