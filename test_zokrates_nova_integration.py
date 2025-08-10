#!/usr/bin/env python3
"""
Test Suite for ZoKrates Nova Integration
Tests the ZoKrates Nova recursive SNARK implementation
"""

import sys
import os
import json
import time
import tempfile
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_zokrates_installation():
    """Test if ZoKrates is installed and has Nova support"""
    print("🔧 Testing ZoKrates installation and Nova support...")
    
    try:
        # Check ZoKrates is installed
        result = subprocess.run(['zokrates', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ZoKrates not found")
            return False
        
        print(f"✅ ZoKrates found: {result.stdout.strip()}")
        
        # Check Nova support
        result = subprocess.run(['zokrates', 'nova', '--help'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ZoKrates Nova support not found")
            return False
        
        print("✅ ZoKrates Nova support confirmed")
        return True
        
    except Exception as e:
        print(f"❌ ZoKrates check failed: {e}")
        return False

def test_nova_manager_import():
    """Test ZoKrates Nova manager import and initialization"""
    print("📦 Testing ZoKrates Nova Manager import...")
    
    try:
        from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
        print("✅ ZoKratesNovaManager imported successfully")
        
        # Test manager creation
        manager = ZoKratesNovaManager(
            circuit_path="circuits/nova/iot_recursive.zok",
            batch_size=5
        )
        print("✅ Nova manager created successfully")
        
        # Test Nova support check
        if manager.check_zokrates_nova_support():
            print("✅ ZoKrates Nova support detected")
            return True
        else:
            print("⚠️  ZoKrates Nova support not detected")
            return False
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Manager test failed: {e}")
        return False

def test_circuit_compilation():
    """Test Nova circuit compilation"""
    print("⚙️  Testing Nova circuit compilation...")
    
    try:
        from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
        
        manager = ZoKratesNovaManager(
            circuit_path="circuits/nova/iot_recursive.zok",
            batch_size=5
        )
        
        # Test circuit compilation (setup)
        print("📋 Compiling Nova circuit...")
        setup_success = manager.setup()
        
        if setup_success:
            print("✅ Nova circuit compiled successfully")
            return True
        else:
            print("❌ Nova circuit compilation failed")
            return False
        
    except Exception as e:
        print(f"❌ Circuit compilation test failed: {e}")
        return False

def test_data_conversion():
    """Test IoT data to Nova format conversion"""
    print("🔄 Testing IoT data conversion...")
    
    try:
        from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
        
        manager = ZoKratesNovaManager(batch_size=3)
        
        # Create test IoT data
        test_data = [
            {
                'sensor_id': 'test_sensor_1',
                'sensor_type': 'temperature',
                'room': 'living_room',
                'value': 22.5,
                'privacy_level': 1,
                'timestamp': int(time.time())
            },
            {
                'sensor_id': 'test_sensor_2', 
                'sensor_type': 'humidity',
                'room': 'bedroom',
                'value': 45.0,
                'privacy_level': 2,
                'timestamp': int(time.time())
            }
        ]
        
        # Test initial state preparation
        init_state = manager.prepare_initial_state()
        print(f"✅ Initial state prepared: {len(init_state)} fields")
        
        # Test step input preparation
        step_input = manager.prepare_step_input(test_data)
        print(f"✅ Step input prepared: {len(step_input)} fields")
        
        # Validate conversion (simplified format)
        assert len(step_input['values']) == 3  # Fixed to 3 values
        assert 'batch_id' in step_input
        
        print("✅ Data conversion validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Data conversion test failed: {e}")
        return False

def test_recursive_proof_generation():
    """Test Nova recursive proof generation"""
    print("🔐 Testing Nova recursive proof generation...")
    
    try:
        from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
        
        manager = ZoKratesNovaManager(
            circuit_path="circuits/nova/iot_recursive.zok",
            batch_size=3
        )
        
        # Setup first
        if not manager.setup():
            print("⚠️  Setup failed, skipping proof test")
            return False
        
        # Create test batches
        batches = []
        for batch_num in range(2):  # 2 batches
            batch = []
            for i in range(3):  # 3 readings per batch
                batch.append({
                    'sensor_id': f'test_sensor_{batch_num}_{i}',
                    'sensor_type': 'temperature',
                    'room': f'room_{i+1}',
                    'value': 20.0 + batch_num + i * 0.5,
                    'privacy_level': (i % 3) + 1,
                    'timestamp': int(time.time()) + batch_num * 100 + i * 10
                })
            batches.append(batch)
        
        print(f"📦 Created {len(batches)} test batches")
        
        # Generate recursive proof
        start_time = time.time()
        result = manager.prove_recursive_batch(batches)
        end_time = time.time()
        
        if result.success:
            print("✅ Nova recursive proof generation successful!")
            print(f"   📈 Steps: {result.step_count}")
            print(f"   ⏱️  Time: {result.total_time:.3f}s")
            print(f"   ✅ Verify time: {result.verify_time:.3f}s")
            print(f"   📦 Proof size: {result.proof_size} bytes")
            return True
        else:
            print(f"❌ Nova proof failed: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Recursive proof test failed: {e}")
        return False

def test_benchmark_comparison():
    """Test Nova vs traditional benchmark"""
    print("📊 Testing Nova vs traditional benchmark...")
    
    try:
        from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
        
        manager = ZoKratesNovaManager(batch_size=5)
        
        # Create test data
        test_data = []
        for i in range(15):  # 3 batches worth
            test_data.append({
                'sensor_id': f'benchmark_sensor_{i}',
                'sensor_type': 'temperature',
                'room': f'room_{(i % 3) + 1}',
                'value': 18.0 + i * 0.3,
                'privacy_level': (i % 3) + 1,
                'timestamp': int(time.time()) + i * 30
            })
        
        # Mock traditional proof time (normally from ZoKrates SNARK)
        traditional_time = 2.0  # 2 seconds baseline
        
        # Run benchmark
        comparison = manager.benchmark_vs_traditional(test_data, traditional_time)
        
        if comparison.get('nova_available', False):
            print("✅ Nova benchmark completed")
            metrics = comparison['nova_metrics']
            print(f"   ⏱️  Nova time: {metrics['total_time']:.3f}s")
            print(f"   📦 Proof size: {metrics['proof_size']} bytes")
            print(f"   📈 Throughput: {metrics['throughput']:.2f} readings/s")
            
            improvements = comparison['improvements']
            print(f"   🚀 Speedup: {improvements['time_speedup']:.2f}x")
            return True
        else:
            print(f"❌ Nova benchmark failed: {comparison.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        return False

def test_orchestrator_integration():
    """Test integration with main orchestrator"""
    print("🎛️  Testing orchestrator integration...")
    
    try:
        # Ensure we have seaborn (should be available in venv)
        import seaborn
        from orchestrator import IoTZKOrchestrator
        
        # Create test config
        test_config = {
            "nova_config": {
                "circuit_path": "circuits/nova/iot_recursive.zok",
                "batch_size": 5
            }
        }
        
        # Test orchestrator creation
        orchestrator = IoTZKOrchestrator()
        print("✅ Orchestrator created with Nova manager")
        
        # Check Nova manager type
        if hasattr(orchestrator, 'nova_manager'):
            manager_type = type(orchestrator.nova_manager).__name__
            if manager_type == 'ZoKratesNovaManager':
                print("✅ ZoKratesNovaManager integrated correctly")
                return True
            else:
                print(f"❌ Wrong manager type: {manager_type}")
                return False
        else:
            print("❌ Nova manager not found in orchestrator")
            return False
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False

def run_all_tests():
    """Run complete test suite"""
    print("🚀 ZoKrates Nova Integration Test Suite")
    print("="*50)
    
    tests = [
        ("ZoKrates Installation", test_zokrates_installation),
        ("Nova Manager Import", test_nova_manager_import),
        ("Circuit Compilation", test_circuit_compilation),
        ("Data Conversion", test_data_conversion),
        ("Recursive Proof Generation", test_recursive_proof_generation),
        ("Benchmark Comparison", test_benchmark_comparison),
        ("Orchestrator Integration", test_orchestrator_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ PASS: {test_name}")
            else:
                print(f"❌ FAIL: {test_name}")
                
        except Exception as e:
            print(f"💥 ERROR: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 Test Summary")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} ({passed/total*100:.1f}% success rate)")
    print()
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if passed == total:
        print("\n🎉 All tests passed! ZoKrates Nova integration is ready.")
    elif passed > total / 2:
        print(f"\n⚠️  {total-passed} tests failed. Check configuration and try again.")
    else:
        print(f"\n❌ Many tests failed. ZoKrates Nova may not be properly configured.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
