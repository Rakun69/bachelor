#!/usr/bin/env python3
"""
Grundlegende Funktionalitätstests für das IoT ZK-SNARK System
Testet Schritt für Schritt ob die Kernkomponenten funktionieren
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_1_iot_simulation():
    """Test 1: IoT Sensor Simulation - Generiert realistische Sensordaten"""
    print("=" * 60)
    print("TEST 1: IoT Sensor Simulation")
    print("=" * 60)
    
    try:
        from src.iot_simulation.smart_home import SmartHomeSensors
        
        # Erstelle Smart Home Simulator
        simulator = SmartHomeSensors()
        
        # Generiere 1 Stunde Daten (60 Readings)
        readings = simulator.generate_readings(duration_hours=1, time_step_seconds=60)
        
        # Validierungen
        assert len(readings) > 0, "Keine Sensordaten generiert!"
        
        # Prüfe Sensor-Typen
        sensor_types = set(r.sensor_type for r in readings)
        expected_types = {'temperature', 'humidity', 'motion', 'light'}
        assert len(sensor_types.intersection(expected_types)) > 0, "Keine erwarteten Sensor-Typen gefunden!"
        
        # Prüfe Temperatur-Werte (sollten realistisch sein)
        temp_readings = [r for r in readings if r.sensor_type == 'temperature']
        if temp_readings:
            temp_values = [r.value for r in temp_readings]
            assert min(temp_values) >= 0, "Unrealistische Temperatur (< 0°C)"
            assert max(temp_values) <= 50, "Unrealistische Temperatur (> 50°C)"
        
        # Prüfe Privacy Levels
        privacy_levels = set(r.privacy_level for r in readings)
        assert privacy_levels.issubset({1, 2, 3}), "Ungültige Privacy Levels!"
        
        print(f"✅ IoT Simulation erfolgreich!")
        print(f"   - {len(readings)} Sensordaten generiert")
        print(f"   - {len(sensor_types)} verschiedene Sensor-Typen: {sensor_types}")
        print(f"   - Privacy Levels: {sorted(privacy_levels)}")
        if temp_readings:
            print(f"   - Temperatur-Bereich: {min(temp_values):.1f}°C - {max(temp_values):.1f}°C")
        
        return True, readings
        
    except Exception as e:
        print(f"❌ IoT Simulation FEHLGESCHLAGEN: {e}")
        return False, None

def test_2_zokrates_availability():
    """Test 2: ZoKrates Installation - Prüft ob ZoKrates verfügbar ist"""
    print("\n" + "=" * 60)
    print("TEST 2: ZoKrates Verfügbarkeit")
    print("=" * 60)
    
    import subprocess
    
    try:
        # Teste ZoKrates Version
        result = subprocess.run(['zokrates', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ ZoKrates verfügbar: {result.stdout.strip()}")
            
            # Teste ZoKrates Nova Support
            nova_result = subprocess.run(['zokrates', 'nova', '--help'],
                                       capture_output=True, text=True, timeout=10)
            
            if nova_result.returncode == 0:
                print("✅ ZoKrates Nova Support verfügbar")
                return True, "nova_available"
            else:
                print("⚠️  ZoKrates Nova Support NICHT verfügbar")
                return True, "nova_not_available"
        else:
            print(f"❌ ZoKrates nicht verfügbar: {result.stderr}")
            return False, None
            
    except FileNotFoundError:
        print("❌ ZoKrates nicht installiert!")
        return False, None
    except Exception as e:
        print(f"❌ ZoKrates Test fehlgeschlagen: {e}")
        return False, None

def test_3_circuit_compilation():
    """Test 3: ZK Circuit Kompilierung - Testet einfachstes Circuit"""
    print("\n" + "=" * 60)
    print("TEST 3: ZK Circuit Kompilierung")
    print("=" * 60)
    
    try:
        from src.proof_systems.snark_manager import SNARKManager
        
        # Erstelle SNARK Manager
        manager = SNARKManager(
            circuits_dir="circuits",
            output_dir="data/test_proofs"
        )
        
        # Teste einfachstes Circuit: filter_range
        circuit_path = Path("circuits/basic/filter_range.zok")
        
        if not circuit_path.exists():
            print(f"❌ Circuit nicht gefunden: {circuit_path}")
            return False, None
        
        print(f"📁 Circuit gefunden: {circuit_path}")
        
        # Kompiliere Circuit
        print("🔨 Kompiliere filter_range Circuit...")
        compile_success = manager.compile_circuit(str(circuit_path), "filter_range")
        
        if not compile_success:
            print("❌ Circuit Kompilierung fehlgeschlagen!")
            return False, None
        
        print("✅ Circuit erfolgreich kompiliert!")
        
        # Setup Circuit
        print("⚙️  Setup Circuit...")
        setup_success = manager.setup_circuit("filter_range")
        
        if not setup_success:
            print("❌ Circuit Setup fehlgeschlagen!")
            return False, None
        
        print("✅ Circuit Setup erfolgreich!")
        return True, manager
        
    except Exception as e:
        print(f"❌ Circuit Kompilierung fehlgeschlagen: {e}")
        return False, None

def test_4_simple_proof():
    """Test 4: Einfacher ZK Proof - Generiert einen Test-Proof"""
    print("\n" + "=" * 60)
    print("TEST 4: Einfacher ZK Proof")
    print("=" * 60)
    
    try:
        from src.proof_systems.snark_manager import SNARKManager
        
        # Erstelle SNARK Manager (reuse from test 3)
        manager = SNARKManager(
            circuits_dir="circuits", 
            output_dir="data/test_proofs"
        )
        
        # Kompiliere und Setup (falls noch nicht gemacht)
        circuit_path = Path("circuits/basic/filter_range.zok")
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Teste einfachen Proof: Beweise dass 25 zwischen 10 und 50 liegt
        # filter_range.zok erwartet: min_val, max_val, secret_value
        test_inputs = ["10", "50", "25"]
        
        print(f"🔐 Generiere Proof für: secret_value=25 ist zwischen 10 und 50")
        print(f"   Inputs: {test_inputs}")
        
        result = manager.generate_proof("filter_range", test_inputs)
        
        if result.success:
            print("✅ ZK Proof erfolgreich generiert!")
            print(f"   - Proof Zeit: {result.metrics.proof_time:.3f}s")
            print(f"   - Verify Zeit: {result.metrics.verify_time:.3f}s") 
            print(f"   - Proof Größe: {result.metrics.proof_size} bytes")
            print(f"   - Witness Zeit: {result.metrics.witness_time:.3f}s")
            return True, result
        else:
            print(f"❌ ZK Proof fehlgeschlagen: {result.error_message}")
            return False, None
            
    except Exception as e:
        print(f"❌ Proof Generation fehlgeschlagen: {e}")
        return False, None

def main():
    """Führe alle Tests nacheinander aus"""
    print("🚀 STARTE SYSTEMATISCHE TESTS FÜR IOT ZK-SNARK SYSTEM")
    print("=" * 80)
    
    # Test 1: IoT Simulation
    iot_success, iot_data = test_1_iot_simulation()
    if not iot_success:
        print("\n❌ STOPP: IoT Simulation fehlgeschlagen - weitere Tests nicht möglich")
        return False
    
    # Test 2: ZoKrates
    zk_success, zk_status = test_2_zokrates_availability()
    if not zk_success:
        print("\n❌ STOPP: ZoKrates nicht verfügbar - ZK Tests nicht möglich")
        return False
    
    # Test 3: Circuit Kompilierung
    compile_success, manager = test_3_circuit_compilation()
    if not compile_success:
        print("\n❌ STOPP: Circuit Kompilierung fehlgeschlagen")
        return False
    
    # Test 4: Einfacher Proof
    proof_success, proof_result = test_4_simple_proof()
    if not proof_success:
        print("\n❌ STOPP: Proof Generation fehlgeschlagen")
        return False
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("🎉 ALLE GRUNDTESTS ERFOLGREICH!")
    print("=" * 80)
    print("✅ IoT Simulation funktioniert")
    print("✅ ZoKrates ist verfügbar")
    print("✅ Circuit Kompilierung funktioniert")
    print("✅ ZK Proof Generation funktioniert")
    
    if zk_status == "nova_available":
        print("✅ Nova Support verfügbar - Recursive SNARKs testbar")
    else:
        print("⚠️  Nova Support nicht verfügbar - nur Standard SNARKs")
    
    print("\n🔥 SYSTEM IST BEREIT FÜR VOLLSTÄNDIGE EVALUATION!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)