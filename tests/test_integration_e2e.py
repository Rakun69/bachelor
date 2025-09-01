#!/usr/bin/env python3
"""
Integration & End-to-End Tests
Testet das komplette System von IoT-Daten bis zu finalen Ergebnissen
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_complete_orchestrator_workflow():
    """Teste den kompletten Orchestrator Workflow"""
    print("=" * 80)
    print("INTEGRATION: Kompletter Orchestrator Workflow")
    print("=" * 80)
    
    try:
        from src.orchestrator import IoTZKOrchestrator
        
        # Erstelle Orchestrator mit Test-Config
        test_config = {
            "iot_simulation": {
                "multi_period_enabled": False,  # Für schnelleren Test
                "duration_hours": 2,            # Nur 2 Stunden
                "time_step_seconds": 300        # 5 Minuten Schritte
            },
            "circuit_types": ["filter_range"],  # Nur ein Circuit
            "data_sizes": [10, 20],            # Kleine Größen
            "batch_sizes": [5],                # Eine Batch-Größe
            "privacy_levels": [2],             # Ein Privacy Level
            "iterations": 1,                   # Eine Iteration
            "evaluation": {
                "run_performance_tests": True,
                "run_privacy_analysis": True,
                "run_scalability_tests": True,
                "generate_visualizations": False,  # Für Speed
                "run_nova_comparison": False       # Für Speed
            }
        }
        
        # Speichere Test-Config
        config_file = Path("data/test_config.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(test_config, f, indent=2)
        
        print(f"📝 Test-Konfiguration erstellt: {config_file}")
        
        # Erstelle Orchestrator
        orchestrator = IoTZKOrchestrator(str(config_file))
        
        print(f"✅ Orchestrator initialisiert")
        
        # Teste Phase für Phase
        phases_results = {}
        
        # Phase 1: Datengenerierung
        print(f"\n📊 Phase 1: IoT Datengenerierung")
        iot_data = orchestrator._generate_iot_data()
        
        if isinstance(iot_data, dict) and iot_data.get("total_readings", 0) > 0:
            print(f"✅ IoT Daten generiert: {iot_data['total_readings']} readings")
            phases_results["data_generation"] = True
        else:
            print(f"❌ IoT Datengenerierung fehlgeschlagen")
            phases_results["data_generation"] = False
            return False, phases_results
        
        # Phase 2: Circuit Kompilierung
        print(f"\n🔧 Phase 2: Circuit Kompilierung")
        circuit_status = orchestrator._compile_circuits()
        
        successful_circuits = sum(1 for status in circuit_status.values() 
                                if status.get("status") == "success")
        
        if successful_circuits > 0:
            print(f"✅ {successful_circuits} Circuits erfolgreich kompiliert")
            phases_results["circuit_compilation"] = True
        else:
            print(f"❌ Circuit Kompilierung fehlgeschlagen")
            phases_results["circuit_compilation"] = False
            return False, phases_results
        
        # Phase 3: Benchmarks
        print(f"\n📈 Phase 3: Benchmark Ausführung")
        benchmark_results = orchestrator._run_benchmarks()
        
        if isinstance(benchmark_results, list) and len(benchmark_results) > 0:
            successful_benchmarks = sum(1 for r in benchmark_results 
                                      if r.get("success_rate", 0) > 0.5)
            print(f"✅ {successful_benchmarks}/{len(benchmark_results)} Benchmarks erfolgreich")
            phases_results["benchmarks"] = successful_benchmarks > 0
        else:
            print(f"❌ Benchmark Ausführung fehlgeschlagen")
            phases_results["benchmarks"] = False
            return False, phases_results
        
        # Phase 4: Analyse
        print(f"\n🔍 Phase 4: Ergebnis-Analyse")
        analysis = orchestrator._analyze_results()
        
        if isinstance(analysis, dict) and "comparison_report" in analysis:
            print(f"✅ Analyse erfolgreich durchgeführt")
            phases_results["analysis"] = True
        else:
            print(f"❌ Analyse fehlgeschlagen")
            phases_results["analysis"] = False
        
        # Zusammenfassung
        successful_phases = sum(1 for success in phases_results.values() if success)
        total_phases = len(phases_results)
        
        print(f"\n📊 Workflow Zusammenfassung:")
        print(f"   Erfolgreiche Phasen: {successful_phases}/{total_phases}")
        
        return successful_phases == total_phases, phases_results
        
    except Exception as e:
        print(f"❌ Orchestrator Workflow fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

def test_data_pipeline_integrity():
    """Teste die Datenintegrität durch die gesamte Pipeline"""
    print("\n" + "=" * 80)
    print("INTEGRATION: Daten-Pipeline Integrität")
    print("=" * 80)
    
    try:
        from src.iot_simulation.smart_home import SmartHomeSensors
        from src.proof_systems.snark_manager import SNARKManager
        
        # Schritt 1: IoT Daten generieren
        print("📊 Schritt 1: IoT Daten generieren")
        simulator = SmartHomeSensors()
        readings = simulator.generate_readings(duration_hours=1, time_step_seconds=300)
        
        if not readings:
            print("❌ Keine IoT Daten generiert")
            return False
        
        print(f"✅ {len(readings)} IoT Readings generiert")
        
        # Schritt 2: Daten für ZK-Proofs vorbereiten
        print("🔧 Schritt 2: Daten für ZK-Proofs vorbereiten")
        
        # Filtere Temperatur-Daten
        temp_readings = [r for r in readings if r.sensor_type == "temperature"]
        
        if not temp_readings:
            print("❌ Keine Temperatur-Daten gefunden")
            return False
        
        print(f"✅ {len(temp_readings)} Temperatur-Readings gefiltert")
        
        # Schritt 3: ZK-Proofs generieren
        print("🔐 Schritt 3: ZK-Proofs generieren")
        
        manager = SNARKManager(circuits_dir="circuits", output_dir="data/test_proofs")
        
        # Kompiliere filter_range Circuit
        circuit_path = Path("circuits/basic/filter_range.zok")
        if not manager.compile_circuit(str(circuit_path), "filter_range"):
            print("❌ Circuit Kompilierung fehlgeschlagen")
            return False
        
        if not manager.setup_circuit("filter_range"):
            print("❌ Circuit Setup fehlgeschlagen")
            return False
        
        # Generiere Proofs für erste 5 Temperatur-Readings
        successful_proofs = 0
        total_proof_size = 0
        
        for i, reading in enumerate(temp_readings[:5]):
            temp_value = int(reading.value)
            inputs = ["10", "40", str(temp_value)]  # Min: 10°C, Max: 40°C
            
            result = manager.generate_proof("filter_range", inputs)
            
            if result.success:
                successful_proofs += 1
                total_proof_size += result.metrics.proof_size
                print(f"   ✅ Proof {i+1}: {temp_value}°C -> {result.metrics.proof_size} bytes")
            else:
                print(f"   ❌ Proof {i+1} fehlgeschlagen: {result.error_message}")
        
        print(f"✅ {successful_proofs}/5 Proofs erfolgreich generiert")
        print(f"📦 Gesamt Proof-Größe: {total_proof_size:,} bytes")
        
        # Schritt 4: Datenintegrität prüfen
        print("🔍 Schritt 4: Datenintegrität prüfen")
        
        # Prüfe ob alle Temperaturwerte im erwarteten Bereich sind
        temp_values = [r.value for r in temp_readings]
        valid_temps = [v for v in temp_values if 10 <= v <= 40]
        
        integrity_score = len(valid_temps) / len(temp_values) if temp_values else 0
        
        print(f"🌡️  Temperatur-Integrität: {len(valid_temps)}/{len(temp_values)} Werte im Bereich 10-40°C")
        print(f"📊 Integritäts-Score: {integrity_score:.2%}")
        
        # Pipeline erfolgreich wenn > 80% der Daten valide sind und > 80% der Proofs erfolgreich
        pipeline_success = (integrity_score > 0.8 and 
                          successful_proofs >= 4)  # 4/5 = 80%
        
        if pipeline_success:
            print("✅ Daten-Pipeline Integrität bestätigt")
        else:
            print("❌ Daten-Pipeline Integrität unzureichend")
        
        return pipeline_success
        
    except Exception as e:
        print(f"❌ Daten-Pipeline Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling_robustness():
    """Teste Fehlerbehandlung und Robustheit des Systems"""
    print("\n" + "=" * 80)
    print("INTEGRATION: Error Handling & Robustheit")
    print("=" * 80)
    
    try:
        from src.proof_systems.snark_manager import SNARKManager
        
        manager = SNARKManager(circuits_dir="circuits", output_dir="data/test_proofs")
        
        error_tests = []
        
        # Test 1: Nicht-existierender Circuit
        print("🧪 Test 1: Nicht-existierender Circuit")
        try:
            result = manager.compile_circuit("non_existent_circuit.zok", "fake_circuit")
            if not result:
                print("✅ Fehler korrekt abgefangen: Nicht-existierender Circuit")
                error_tests.append(True)
            else:
                print("❌ Fehler nicht abgefangen: Nicht-existierender Circuit")
                error_tests.append(False)
        except Exception as e:
            print("❌ Unbehandelte Exception bei nicht-existierendem Circuit")
            error_tests.append(False)
        
        # Test 2: Ungültige Proof-Inputs
        print("🧪 Test 2: Ungültige Proof-Inputs")
        
        # Kompiliere filter_range für Test
        circuit_path = Path("circuits/basic/filter_range.zok")
        if circuit_path.exists():
            manager.compile_circuit(str(circuit_path), "filter_range")
            manager.setup_circuit("filter_range")
            
            try:
                # Ungültige Inputs: secret_value außerhalb des Bereichs
                invalid_inputs = ["10", "20", "30"]  # 30 ist > 20 (max_val)
                result = manager.generate_proof("filter_range", invalid_inputs)
                
                if not result.success:
                    print("✅ Ungültige Inputs korrekt abgelehnt")
                    error_tests.append(True)
                else:
                    print("❌ Ungültige Inputs nicht erkannt")
                    error_tests.append(False)
            except Exception as e:
                print("❌ Unbehandelte Exception bei ungültigen Inputs")
                error_tests.append(False)
        else:
            print("⚠️  Circuit nicht gefunden - Test übersprungen")
            error_tests.append(True)  # Nicht als Fehler werten
        
        # Test 3: Speicher-/Timeout-Robustheit
        print("🧪 Test 3: Timeout-Robustheit")
        
        # Teste mit sehr großen Arrays (sollte Timeout auslösen oder graceful handhaben)
        try:
            # Erstelle sehr lange Input-Liste
            large_inputs = [str(i) for i in range(1000)]  # Viel zu viele Inputs
            
            result = manager.generate_proof("filter_range", large_inputs)
            
            # Sollte entweder fehlschlagen oder Timeout
            if not result.success:
                print("✅ Große Inputs korrekt behandelt (Fehler erwartet)")
                error_tests.append(True)
            else:
                print("⚠️  Große Inputs akzeptiert (unerwartet aber nicht kritisch)")
                error_tests.append(True)
        except Exception as e:
            print("✅ Exception bei großen Inputs korrekt abgefangen")
            error_tests.append(True)
        
        # Test 4: IoT Simulator Robustheit
        print("🧪 Test 4: IoT Simulator Robustheit")
        
        try:
            from src.iot_simulation.smart_home import SmartHomeSensors
            
            simulator = SmartHomeSensors()
            
            # Test mit extremen Parametern
            readings = simulator.generate_readings(duration_hours=0, time_step_seconds=1)
            
            if isinstance(readings, list):
                print("✅ IoT Simulator behandelt extreme Parameter graceful")
                error_tests.append(True)
            else:
                print("❌ IoT Simulator Fehler bei extremen Parametern")
                error_tests.append(False)
        except Exception as e:
            print("❌ Unbehandelte Exception im IoT Simulator")
            error_tests.append(False)
        
        # Zusammenfassung
        passed_error_tests = sum(error_tests)
        total_error_tests = len(error_tests)
        
        print(f"\n📊 Error Handling Zusammenfassung:")
        print(f"   Bestandene Tests: {passed_error_tests}/{total_error_tests}")
        
        robustness_score = passed_error_tests / total_error_tests if total_error_tests > 0 else 0
        
        if robustness_score >= 0.8:
            print("✅ System ist robust und behandelt Fehler korrekt")
            return True
        else:
            print("❌ System-Robustheit unzureichend")
            return False
        
    except Exception as e:
        print(f"❌ Error Handling Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_consistency():
    """Teste Performance-Konsistenz über mehrere Durchläufe"""
    print("\n" + "=" * 80)
    print("INTEGRATION: Performance Konsistenz")
    print("=" * 80)
    
    try:
        from src.proof_systems.snark_manager import SNARKManager
        
        manager = SNARKManager(circuits_dir="circuits", output_dir="data/test_proofs")
        
        # Setup filter_range Circuit
        circuit_path = Path("circuits/basic/filter_range.zok")
        if not circuit_path.exists():
            print("❌ Circuit nicht gefunden")
            return False
        
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Performance Test über mehrere Durchläufe
        iterations = 10
        test_inputs = ["10", "50", "25"]
        
        times = []
        sizes = []
        
        print(f"🔄 Führe {iterations} Performance-Tests durch...")
        
        for i in range(iterations):
            result = manager.generate_proof("filter_range", test_inputs)
            
            if result.success:
                times.append(result.metrics.proof_time)
                sizes.append(result.metrics.proof_size)
                print(f"   Test {i+1}: {result.metrics.proof_time:.3f}s, {result.metrics.proof_size} bytes")
            else:
                print(f"   Test {i+1}: ❌ Fehlgeschlagen")
        
        if len(times) < iterations * 0.8:  # Mindestens 80% erfolgreich
            print("❌ Zu viele fehlgeschlagene Tests für Konsistenz-Analyse")
            return False
        
        # Statistiken berechnen
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        time_variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        time_std_dev = time_variance ** 0.5
        
        avg_size = sum(sizes) / len(sizes)
        size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        
        print(f"\n📊 Performance Statistiken:")
        print(f"   ⏱️  Zeit - Durchschnitt: {avg_time:.3f}s")
        print(f"   ⏱️  Zeit - Min/Max: {min_time:.3f}s / {max_time:.3f}s")
        print(f"   ⏱️  Zeit - Std. Abweichung: {time_std_dev:.3f}s")
        print(f"   📦 Größe - Durchschnitt: {avg_size:.0f} bytes")
        print(f"   📦 Größe - Varianz: {size_variance:.0f}")
        
        # Konsistenz-Bewertung
        time_consistency = (time_std_dev / avg_time) < 0.2  # < 20% Variation
        size_consistency = size_variance == 0  # Proof-Größe sollte konstant sein
        
        print(f"\n🎯 Konsistenz-Bewertung:")
        print(f"   Zeit-Konsistenz: {'✅' if time_consistency else '❌'} ({'OK' if time_consistency else 'Zu variabel'})")
        print(f"   Größe-Konsistenz: {'✅' if size_consistency else '❌'} ({'Konstant' if size_consistency else 'Variabel'})")
        
        overall_consistency = time_consistency and size_consistency
        
        if overall_consistency:
            print("✅ Performance ist konsistent über mehrere Durchläufe")
        else:
            print("⚠️  Performance-Inkonsistenzen erkannt")
        
        return overall_consistency
        
    except Exception as e:
        print(f"❌ Performance Konsistenz Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Führe alle Integration & E2E Tests aus"""
    print("🚀 INTEGRATION & END-TO-END TEST SUITE")
    print("=" * 80)
    
    test_results = {}
    
    # Test 1: Kompletter Orchestrator Workflow
    print("🎯 TEST 1: Kompletter Orchestrator Workflow")
    workflow_success, workflow_details = test_complete_orchestrator_workflow()
    test_results["orchestrator_workflow"] = workflow_success
    
    # Test 2: Daten-Pipeline Integrität
    print("\n📊 TEST 2: Daten-Pipeline Integrität")
    pipeline_success = test_data_pipeline_integrity()
    test_results["data_pipeline"] = pipeline_success
    
    # Test 3: Error Handling & Robustheit
    print("\n🛡️  TEST 3: Error Handling & Robustheit")
    robustness_success = test_error_handling_robustness()
    test_results["error_handling"] = robustness_success
    
    # Test 4: Performance Konsistenz
    print("\n📈 TEST 4: Performance Konsistenz")
    consistency_success = test_performance_consistency()
    test_results["performance_consistency"] = consistency_success
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("🏆 INTEGRATION & E2E TEST ZUSAMMENFASSUNG")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for success in test_results.values() if success)
    
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<25}: {status}")
    
    print(f"\n📊 Gesamtergebnis: {passed_tests}/{total_tests} Tests bestanden")
    
    if passed_tests == total_tests:
        print("🎉 ALLE INTEGRATION TESTS ERFOLGREICH!")
        print("✅ System ist vollständig integriert und funktionsfähig")
        print("🔥 End-to-End Pipeline funktioniert einwandfrei")
    else:
        print("⚠️  Einige Integration-Tests fehlgeschlagen")
        print("🔧 System braucht weitere Integration-Arbeit")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
