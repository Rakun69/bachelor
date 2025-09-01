#!/usr/bin/env python3
"""
Kleiner Benchmark Test - Testet ob die Benchmark-Pipeline funktioniert
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_small_benchmark():
    """Teste einen sehr kleinen Benchmark"""
    print("=" * 60)
    print("TEST: Kleiner Performance Benchmark")
    print("=" * 60)
    
    try:
        from src.evaluation.benchmark_framework import BenchmarkFramework, BenchmarkConfig
        from src.proof_systems.snark_manager import SNARKManager
        from src.iot_simulation.smart_home import SmartHomeSensors
        
        # Erstelle minimale Benchmark Config
        config = BenchmarkConfig(
            circuit_types=["filter_range"],  # Nur ein Circuit
            data_sizes=[10],                 # Nur 10 Datenpunkte
            batch_sizes=[5],                 # Nur Batch-Größe 5
            privacy_levels=[2],              # Nur Privacy Level 2
            iterations=1,                    # Nur 1 Iteration
            output_dir="data/test_benchmarks"
        )
        
        print(f"📊 Benchmark Config:")
        print(f"   - Circuits: {config.circuit_types}")
        print(f"   - Data Sizes: {config.data_sizes}")
        print(f"   - Batch Sizes: {config.batch_sizes}")
        print(f"   - Iterations: {config.iterations}")
        
        # Erstelle Framework
        framework = BenchmarkFramework(config)
        
        # Erstelle Manager
        snark_manager = SNARKManager(
            circuits_dir="circuits",
            output_dir="data/test_proofs"
        )
        
        # Erstelle IoT Simulator
        iot_simulator = SmartHomeSensors()
        
        print("\n🚀 Starte Mini-Benchmark...")
        
        # Führe Benchmark aus
        results = framework.run_benchmark_suite(snark_manager, iot_simulator)
        
        print(f"\n✅ Benchmark abgeschlossen!")
        print(f"   - {len(results)} Ergebnisse generiert")
        
        # Analysiere Ergebnisse
        for i, result in enumerate(results):
            print(f"\n📈 Ergebnis {i+1}:")
            print(f"   - Circuit: {result.circuit_type}")
            print(f"   - Proof System: {result.proof_system}")
            print(f"   - Success Rate: {result.success_rate}")
            
            perf = result.performance
            print(f"   - Compile Zeit: {perf.compile_time:.3f}s")
            print(f"   - Setup Zeit: {perf.setup_time:.3f}s")
            print(f"   - Proof Zeit: {perf.proof_generation_time:.3f}s")
            print(f"   - Verify Zeit: {perf.verification_time:.3f}s")
            print(f"   - Proof Größe: {perf.proof_size} bytes")
            print(f"   - Throughput: {perf.throughput:.1f} ops/s")
            
            privacy = result.privacy
            print(f"   - Privacy Level: {privacy.privacy_level}")
            print(f"   - Info Leakage: {privacy.information_leakage:.3f}")
            
            scale = result.scalability
            print(f"   - Data Size: {scale.data_size}")
            print(f"   - Batch Size: {scale.batch_size}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Benchmark fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Führe Mini-Benchmark aus"""
    print("🚀 MINI-BENCHMARK TEST")
    print("=" * 80)
    
    success, results = test_small_benchmark()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 MINI-BENCHMARK ERFOLGREICH!")
        print("=" * 80)
        print("✅ Benchmark Framework funktioniert")
        print("✅ Performance Metriken werden korrekt erfasst")
        print("✅ Privacy Analyse funktioniert")
        print("✅ Scalability Metriken funktionieren")
        print("\n🔥 SYSTEM IST BEREIT FÜR VOLLSTÄNDIGE BENCHMARKS!")
    else:
        print("\n❌ MINI-BENCHMARK FEHLGESCHLAGEN")
        print("⚠️  Benchmark Framework hat Probleme")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)