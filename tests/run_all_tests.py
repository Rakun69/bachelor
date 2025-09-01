#!/usr/bin/env python3
"""
Test Runner - Führt alle Tests systematisch aus
Erstellt einen umfassenden Test-Report für die Bachelorarbeit
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_test_file(test_file: Path, timeout: int = 300):
    """Führe eine Test-Datei aus und sammle Ergebnisse"""
    print(f"\n🧪 FÜHRE TEST AUS: {test_file.name}")
    print("=" * 60)
    
    try:
        start_time = time.time()
        
        # Führe Test aus
        result = subprocess.run([
            sys.executable, str(test_file)
        ], 
        capture_output=True, 
        text=True, 
        timeout=timeout,
        cwd=test_file.parent.parent  # Führe aus dem Projekt-Root aus
        )
        
        execution_time = time.time() - start_time
        
        # Analysiere Ergebnis
        success = result.returncode == 0
        
        print(f"⏱️  Ausführungszeit: {execution_time:.2f}s")
        print(f"📤 Return Code: {result.returncode}")
        
        if success:
            print("✅ TEST ERFOLGREICH")
        else:
            print("❌ TEST FEHLGESCHLAGEN")
        
        # Zeige Output (letzte 20 Zeilen)
        if result.stdout:
            stdout_lines = result.stdout.split('\n')
            print(f"\n📄 Output (letzte 20 Zeilen):")
            for line in stdout_lines[-20:]:
                if line.strip():
                    print(f"   {line}")
        
        if result.stderr and not success:
            print(f"\n🚨 Fehler-Output:")
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        
        return {
            "test_file": test_file.name,
            "success": success,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout_lines": len(result.stdout.split('\n')) if result.stdout else 0,
            "stderr_lines": len(result.stderr.split('\n')) if result.stderr else 0,
            "timeout": False
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TEST TIMEOUT nach {timeout}s")
        return {
            "test_file": test_file.name,
            "success": False,
            "execution_time": timeout,
            "return_code": -1,
            "timeout": True,
            "error": f"Timeout nach {timeout}s"
        }
    except Exception as e:
        print(f"❌ TEST FEHLER: {e}")
        return {
            "test_file": test_file.name,
            "success": False,
            "execution_time": 0,
            "error": str(e)
        }

def create_test_report(test_results: list, total_time: float):
    """Erstelle umfassenden Test-Report"""
    
    # Berechne Statistiken
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results if r["success"])
    failed_tests = total_tests - successful_tests
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    avg_execution_time = sum(r["execution_time"] for r in test_results) / total_tests if total_tests > 0 else 0
    
    # Erstelle Report
    report = {
        "test_run_info": {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "python_version": sys.version,
            "working_directory": str(Path.cwd())
        },
        "summary": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time
        },
        "test_results": test_results,
        "recommendations": []
    }
    
    # Füge Empfehlungen hinzu
    if success_rate == 100:
        report["recommendations"].append("🎉 Alle Tests erfolgreich! System ist produktionsreif.")
    elif success_rate >= 80:
        report["recommendations"].append("✅ Meiste Tests erfolgreich. Kleinere Fixes nötig.")
    elif success_rate >= 60:
        report["recommendations"].append("⚠️  Moderate Erfolgsrate. Debugging empfohlen.")
    else:
        report["recommendations"].append("❌ Niedrige Erfolgsrate. Umfassendes Debugging nötig.")
    
    # Identifiziere langsame Tests
    slow_tests = [r for r in test_results if r["execution_time"] > 60]
    if slow_tests:
        report["recommendations"].append(f"🐌 {len(slow_tests)} langsame Tests identifiziert (>60s)")
    
    # Identifiziere Timeout-Tests
    timeout_tests = [r for r in test_results if r.get("timeout", False)]
    if timeout_tests:
        report["recommendations"].append(f"⏰ {len(timeout_tests)} Tests mit Timeout - Optimierung nötig")
    
    return report

def print_final_summary(report: dict):
    """Drucke finale Zusammenfassung"""
    
    print("\n" + "=" * 80)
    print("🏆 FINALER TEST-REPORT")
    print("=" * 80)
    
    summary = report["summary"]
    
    print(f"📊 TEST STATISTIKEN:")
    print(f"   Gesamt Tests: {summary['total_tests']}")
    print(f"   Erfolgreich: {summary['successful_tests']}")
    print(f"   Fehlgeschlagen: {summary['failed_tests']}")
    print(f"   Erfolgsrate: {summary['success_rate']:.1f}%")
    print(f"   Durchschnitt Zeit: {summary['average_execution_time']:.2f}s")
    print(f"   Gesamt Zeit: {report['test_run_info']['total_execution_time']:.2f}s")
    
    print(f"\n📋 DETAILLIERTE ERGEBNISSE:")
    for result in report["test_results"]:
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['execution_time']:.1f}s"
        print(f"   {status} {result['test_file']:<30} ({time_str})")
    
    print(f"\n💡 EMPFEHLUNGEN:")
    for rec in report["recommendations"]:
        print(f"   {rec}")
    
    # Finale Bewertung
    success_rate = summary['success_rate']
    
    if success_rate == 100:
        print(f"\n🎉 PERFEKT! Alle Tests bestanden!")
        print(f"✅ System ist 100% funktionsfähig")
        print(f"🔥 Bachelorarbeit kann mit vollem Vertrauen geschrieben werden!")
    elif success_rate >= 90:
        print(f"\n🌟 EXZELLENT! {success_rate:.1f}% Tests bestanden!")
        print(f"✅ System ist hochgradig funktionsfähig")
        print(f"🚀 Minimale Nacharbeiten nötig")
    elif success_rate >= 80:
        print(f"\n👍 GUT! {success_rate:.1f}% Tests bestanden!")
        print(f"✅ System ist größtenteils funktionsfähig")
        print(f"🔧 Einige Verbesserungen empfohlen")
    elif success_rate >= 60:
        print(f"\n⚠️  AKZEPTABEL. {success_rate:.1f}% Tests bestanden.")
        print(f"🔧 System braucht Debugging und Fixes")
    else:
        print(f"\n❌ KRITISCH! Nur {success_rate:.1f}% Tests bestanden.")
        print(f"🚨 System braucht umfassende Überarbeitung")

def main():
    """Führe alle Tests aus und erstelle Report"""
    
    print("🚀 COMPREHENSIVE TEST SUITE")
    print("Führt alle Tests systematisch aus und erstellt Report")
    print("=" * 80)
    
    # Finde alle Test-Dateien
    tests_dir = Path(__file__).parent
    test_files = []
    
    # Definiere Test-Reihenfolge (wichtigste zuerst)
    test_order = [
        "test_basic_functionality.py",
        "test_small_benchmark.py", 
        "test_nova_functionality.py",
        "test_recursive_comparison.py",
        "test_integration_e2e.py",
        "test_large_scale_evaluation.py"
    ]
    
    # Sammle Tests in der definierten Reihenfolge
    for test_name in test_order:
        test_file = tests_dir / test_name
        if test_file.exists():
            test_files.append(test_file)
        else:
            print(f"⚠️  Test nicht gefunden: {test_name}")
    
    # Füge weitere Tests hinzu (falls vorhanden)
    for test_file in tests_dir.glob("test_*.py"):
        if test_file not in test_files and test_file.name != "run_all_tests.py":
            test_files.append(test_file)
    
    print(f"📋 Gefundene Tests: {len(test_files)}")
    for i, test_file in enumerate(test_files, 1):
        print(f"   {i}. {test_file.name}")
    
    if not test_files:
        print("❌ Keine Test-Dateien gefunden!")
        return False
    
    # Führe Tests aus
    print(f"\n🏃 STARTE TEST-AUSFÜHRUNG")
    print("=" * 80)
    
    start_time = time.time()
    test_results = []
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n📍 TEST {i}/{len(test_files)}")
        
        # Bestimme Timeout basierend auf Test-Typ
        if "large_scale" in test_file.name:
            timeout = 600  # 10 Minuten für große Tests
        elif "integration" in test_file.name:
            timeout = 300  # 5 Minuten für Integration
        else:
            timeout = 180  # 3 Minuten für normale Tests
        
        result = run_test_file(test_file, timeout)
        test_results.append(result)
        
        # Kurze Pause zwischen Tests
        if i < len(test_files):
            print(f"⏸️  Kurze Pause...")
            time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Erstelle und speichere Report
    print(f"\n📊 ERSTELLE TEST-REPORT...")
    
    report = create_test_report(test_results, total_time)
    
    # Speichere Report
    report_file = Path("data/test_reports/comprehensive_test_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Report gespeichert: {report_file}")
    
    # Drucke finale Zusammenfassung
    print_final_summary(report)
    
    # Return True wenn mindestens 80% der Tests erfolgreich
    success_rate = report["summary"]["success_rate"]
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
