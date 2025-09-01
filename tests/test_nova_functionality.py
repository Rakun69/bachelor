#!/usr/bin/env python3
"""
Nova Recursive SNARK Funktionalitätstest
Testet ob die experimentellen Nova Features wirklich funktionieren
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_nova_setup():
    """Test Nova Circuit Setup"""
    print("=" * 60)
    print("TEST: Nova Circuit Setup")
    print("=" * 60)
    
    try:
        from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager
        
        # Erstelle Nova Manager
        nova_manager = FixedZoKratesNovaManager(
            circuit_path="circuits/nova/iot_recursive.zok",
            batch_size=3
        )
        
        # Prüfe ob Nova Circuit existiert
        if not nova_manager.circuit_path.exists():
            print(f"❌ Nova Circuit nicht gefunden: {nova_manager.circuit_path}")
            return False, None
        
        print(f"📁 Nova Circuit gefunden: {nova_manager.circuit_path}")
        
        # Teste Nova Support
        if not nova_manager.check_zokrates_nova_support():
            print("❌ ZoKrates Nova Support nicht verfügbar")
            return False, None
        
        print("✅ ZoKrates Nova Support bestätigt")
        
        # Setup Nova Circuit
        print("🔨 Setup Nova Circuit...")
        setup_success = nova_manager.setup()
        
        if setup_success:
            print("✅ Nova Circuit Setup erfolgreich!")
            return True, nova_manager
        else:
            print("❌ Nova Circuit Setup fehlgeschlagen")
            return False, None
            
    except Exception as e:
        print(f"❌ Nova Setup fehlgeschlagen: {e}")
        return False, None

def test_nova_proof_generation(nova_manager):
    """Test Nova Proof Generation mit echten IoT Daten"""
    print("\n" + "=" * 60)
    print("TEST: Nova Proof Generation")
    print("=" * 60)
    
    try:
        # Erstelle Test IoT Daten (3 Batches mit je 3 Werten)
        test_iot_batches = [
            [
                {"value": 22.5, "sensor_type": "temperature"},
                {"value": 45.0, "sensor_type": "humidity"}, 
                {"value": 1.0, "sensor_type": "motion"}
            ],
            [
                {"value": 23.1, "sensor_type": "temperature"},
                {"value": 47.2, "sensor_type": "humidity"},
                {"value": 0.0, "sensor_type": "motion"}
            ]
        ]
        
        print(f"🔐 Generiere Nova Recursive Proof für {len(test_iot_batches)} Batches")
        print(f"   Jeder Batch: 3 IoT Werte")
        print(f"   Total: {len(test_iot_batches) * 3} IoT Readings")
        
        # Generiere Nova Proof
        result = nova_manager.prove_recursive_batch(test_iot_batches)
        
        if result.success:
            print("✅ Nova Recursive Proof erfolgreich generiert!")
            print(f"   - Steps processed: {result.step_count}")
            print(f"   - Total Zeit: {result.total_time:.3f}s")
            print(f"   - Verify Zeit: {result.verify_time:.3f}s")
            print(f"   - Proof Größe: {result.proof_size} bytes")
            print(f"   - Zeit pro Step: {result.total_time/result.step_count:.3f}s")
            return True, result
        else:
            print(f"❌ Nova Proof fehlgeschlagen: {result.error_message}")
            return False, None
            
    except Exception as e:
        print(f"❌ Nova Proof Generation fehlgeschlagen: {e}")
        return False, None

def test_nova_vs_standard_comparison():
    """Test Vergleich Nova vs Standard SNARKs"""
    print("\n" + "=" * 60)
    print("TEST: Nova vs Standard SNARK Vergleich")
    print("=" * 60)
    
    try:
        from src.proof_systems.snark_manager import SNARKManager
        from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager
        
        # Erstelle Test IoT Daten
        test_iot_data = [
            {"value": 22.5, "sensor_type": "temperature"},
            {"value": 23.1, "sensor_type": "temperature"}, 
            {"value": 21.8, "sensor_type": "temperature"},
            {"value": 24.2, "sensor_type": "temperature"},
            {"value": 22.9, "sensor_type": "temperature"},
            {"value": 23.7, "sensor_type": "temperature"}
        ]
        
        print(f"📊 Vergleiche Performance für {len(test_iot_data)} IoT Readings")
        
        # Standard SNARK Test (ein einfacher Proof)
        standard_manager = SNARKManager(circuits_dir="circuits", output_dir="data/test_proofs")
        
        # Kompiliere filter_range falls nötig
        circuit_path = Path("circuits/basic/filter_range.zok")
        standard_manager.compile_circuit(str(circuit_path), "filter_range")
        standard_manager.setup_circuit("filter_range")
        
        # Generiere einen Standard Proof als Baseline
        import time
        start_time = time.time()
        standard_result = standard_manager.generate_proof("filter_range", ["20", "25", "22"])
        standard_time = time.time() - start_time
        
        if not standard_result.success:
            print("❌ Standard SNARK Baseline fehlgeschlagen")
            return False, None
        
        print(f"✅ Standard SNARK Baseline: {standard_time:.3f}s")
        
        # Nova Manager Test
        nova_manager = FixedZoKratesNovaManager(
            circuit_path="circuits/nova/iot_recursive.zok",
            batch_size=3
        )
        
        nova_manager.setup()
        
        # Teste Nova mit den IoT Daten
        batches = []
        for i in range(0, len(test_iot_data), 3):
            batch = test_iot_data[i:i+3]
            while len(batch) < 3:
                batch.append({"value": 0, "sensor_type": "padding"})
            batches.append(batch)
        
        nova_result = nova_manager.prove_recursive_batch(batches)
        
        if nova_result.success:
            print("✅ Nova vs Standard Vergleich erfolgreich!")
            
            print(f"   📈 Nova Metriken:")
            print(f"      - Proof Zeit: {nova_result.total_time:.3f}s")
            print(f"      - Verify Zeit: {nova_result.verify_time:.3f}s")
            print(f"      - Proof Größe: {nova_result.proof_size} bytes")
            print(f"      - Steps: {nova_result.step_count}")
            
            time_speedup = standard_time / nova_result.total_time if nova_result.total_time > 0 else 0
            
            print(f"   🚀 Verbesserungen:")
            print(f"      - Zeit Speedup: {time_speedup:.2f}x")
            print(f"      - Compression Factor: {len(batches)}")
            
            return True, {
                "nova_time": nova_result.total_time,
                "standard_time": standard_time,
                "speedup": time_speedup
            }
        else:
            print(f"❌ Nova Vergleich fehlgeschlagen: {nova_result.error_message}")
            return False, None
            
    except Exception as e:
        print(f"❌ Nova vs Standard Vergleich fehlgeschlagen: {e}")
        return False, None

def main():
    """Führe Nova Tests aus"""
    print("🚀 STARTE NOVA RECURSIVE SNARK TESTS")
    print("=" * 80)
    
    # Test 1: Nova Setup
    setup_success, nova_manager = test_nova_setup()
    if not setup_success:
        print("\n❌ STOPP: Nova Setup fehlgeschlagen")
        print("⚠️  Mögliche Ursachen:")
        print("   - Nova Circuit fehlt")
        print("   - ZoKrates Nova nicht korrekt installiert")
        print("   - Pallas Curve Support fehlt")
        return False
    
    # Test 2: Nova Proof Generation
    proof_success, proof_result = test_nova_proof_generation(nova_manager)
    if not proof_success:
        print("\n❌ WARNUNG: Nova Proof Generation fehlgeschlagen")
        print("⚠️  Nova ist experimentell - das kann normal sein")
    
    # Test 3: Nova vs Standard Vergleich
    comparison_success, comparison_result = test_nova_vs_standard_comparison()
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("📊 NOVA TEST ZUSAMMENFASSUNG")
    print("=" * 80)
    
    if setup_success:
        print("✅ Nova Setup funktioniert")
    else:
        print("❌ Nova Setup fehlgeschlagen")
    
    if proof_success:
        print("✅ Nova Proof Generation funktioniert")
        print("🎉 RECURSIVE SNARKs sind voll funktional!")
    else:
        print("⚠️  Nova Proof Generation problematisch")
        print("📝 Fallback: Nur Standard SNARKs für Bachelorarbeit verwenden")
    
    if comparison_success:
        print("✅ Nova vs Standard Vergleich funktioniert")
    else:
        print("⚠️  Nova Vergleich problematisch")
    
    # Empfehlung
    print("\n🎯 EMPFEHLUNG FÜR BACHELORARBEIT:")
    if setup_success and proof_success and comparison_success:
        print("✅ Nova Recursive SNARKs können verwendet werden")
        print("📊 Vollständiger Standard vs Recursive Vergleich möglich")
    elif setup_success:
        print("⚠️  Nova Setup funktioniert, aber Proofs problematisch")
        print("📝 Fokus auf Standard SNARKs + theoretische Nova Diskussion")
    else:
        print("❌ Nova nicht verwendbar")
        print("📝 Fokus auf Standard SNARKs + Batch Processing")
    
    return setup_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)