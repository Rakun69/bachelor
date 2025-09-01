#!/usr/bin/env python3
"""
Nova Debug Script - Findet heraus warum Nova nicht funktioniert
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_nova_manual():
    """Teste Nova manuell Schritt für Schritt"""
    print("=" * 60)
    print("DEBUG: Nova Manual Test")
    print("=" * 60)
    
    nova_dir = Path("circuits/nova")
    
    if not nova_dir.exists():
        print(f"❌ Nova Verzeichnis nicht gefunden: {nova_dir}")
        return False
    
    print(f"📁 Nova Verzeichnis: {nova_dir}")
    
    # Wechsle in Nova Verzeichnis
    original_cwd = os.getcwd()
    os.chdir(nova_dir)
    
    try:
        # 1. Prüfe existierende Dateien
        print("\n📋 Existierende Nova Dateien:")
        for file in nova_dir.iterdir():
            if file.is_file():
                size = file.stat().st_size
                print(f"   - {file.name}: {size:,} bytes")
        
        # 2. Teste Nova Commands direkt
        print("\n🔧 Teste Nova Commands:")
        
        # Nova Help
        result = subprocess.run(['zokrates', 'nova', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Nova Help verfügbar")
        else:
            print(f"❌ Nova Help fehlgeschlagen: {result.stderr}")
            return False
        
        # 3. Prüfe init.json und steps.json
        print("\n📄 Prüfe Input Dateien:")
        
        if Path("init.json").exists():
            with open("init.json", "r") as f:
                init_data = json.load(f)
            print(f"✅ init.json: {init_data}")
        else:
            print("❌ init.json fehlt")
        
        if Path("steps.json").exists():
            with open("steps.json", "r") as f:
                steps_data = json.load(f)
            print(f"✅ steps.json: {len(steps_data)} steps")
            print(f"   Erste 2 steps: {steps_data[:2]}")
        else:
            print("❌ steps.json fehlt")
        
        # 4. Teste Nova Prove mit existierenden Daten
        print("\n🔐 Teste Nova Prove:")
        
        prove_result = subprocess.run(['zokrates', 'nova', 'prove'], 
                                    capture_output=True, text=True, timeout=60)
        
        print(f"Nova Prove Return Code: {prove_result.returncode}")
        print(f"Nova Prove STDOUT: {prove_result.stdout}")
        print(f"Nova Prove STDERR: {prove_result.stderr}")
        
        if prove_result.returncode == 0:
            print("✅ Nova Prove erfolgreich!")
        else:
            print(f"❌ Nova Prove fehlgeschlagen")
            print(f"   Fehler: {prove_result.stderr}")
        
        # 5. Teste Nova Compress
        print("\n📦 Teste Nova Compress:")
        
        compress_result = subprocess.run(['zokrates', 'nova', 'compress'], 
                                       capture_output=True, text=True, timeout=30)
        
        print(f"Nova Compress Return Code: {compress_result.returncode}")
        print(f"Nova Compress STDOUT: {compress_result.stdout}")
        print(f"Nova Compress STDERR: {compress_result.stderr}")
        
        # 6. Teste Nova Verify
        print("\n✅ Teste Nova Verify:")
        
        verify_result = subprocess.run(['zokrates', 'nova', 'verify'], 
                                     capture_output=True, text=True, timeout=30)
        
        print(f"Nova Verify Return Code: {verify_result.returncode}")
        print(f"Nova Verify STDOUT: {verify_result.stdout}")
        print(f"Nova Verify STDERR: {verify_result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Nova Debug fehlgeschlagen: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def debug_nova_python_integration():
    """Debug Nova Python Integration"""
    print("\n" + "=" * 60)
    print("DEBUG: Nova Python Integration")
    print("=" * 60)
    
    try:
        from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
        
        # Erstelle Nova Manager
        nova_manager = ZoKratesNovaManager(
            circuit_path="circuits/nova/iot_recursive.zok",
            batch_size=3
        )
        
        print(f"📁 Circuit Path: {nova_manager.circuit_path}")
        print(f"📁 Working Dir: {nova_manager.working_dir}")
        print(f"🔧 Batch Size: {nova_manager.batch_size}")
        
        # Prüfe Nova Support
        nova_support = nova_manager.check_zokrates_nova_support()
        print(f"🔍 Nova Support: {nova_support}")
        
        if not nova_support:
            print("❌ Nova Support nicht verfügbar!")
            return False
        
        # Teste Setup
        print("\n🔨 Teste Nova Setup...")
        setup_result = nova_manager.setup()
        print(f"Setup Result: {setup_result}")
        
        if not setup_result:
            print("❌ Nova Setup fehlgeschlagen!")
            return False
        
        # Erstelle Test Daten
        test_batches = [
            [
                {"value": 22.5, "sensor_type": "temperature"},
                {"value": 45.0, "sensor_type": "humidity"}, 
                {"value": 1.0, "sensor_type": "motion"}
            ]
        ]
        
        print(f"\n🔐 Teste Nova Proof mit {len(test_batches)} Batch(es)...")
        
        # Teste Proof Generation
        result = nova_manager.prove_recursive_batch(test_batches)
        
        print(f"Proof Success: {result.success}")
        if result.success:
            print(f"✅ Nova Proof erfolgreich!")
            print(f"   Steps: {result.step_count}")
            print(f"   Zeit: {result.total_time:.3f}s")
            print(f"   Proof Size: {result.proof_size} bytes")
        else:
            print(f"❌ Nova Proof fehlgeschlagen: {result.error_message}")
            
            # Debug: Schaue in die _execute_nova_proof Methode
            print("\n🔍 Debug Nova Execution:")
            initial_state = nova_manager.prepare_initial_state()
            steps = []
            for batch in test_batches:
                step_input = nova_manager.prepare_step_input(batch)
                steps.append(step_input)
            
            print(f"Initial State: {initial_state}")
            print(f"Steps: {steps}")
            
            # Teste direkt
            debug_result = nova_manager._execute_nova_proof(initial_state, steps)
            print(f"Direct Execution Result: {debug_result}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Nova Python Integration fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Debug Nova komplett"""
    print("🔍 NOVA DEBUG SESSION")
    print("=" * 80)
    
    # 1. Manual Nova Test
    manual_success = debug_nova_manual()
    
    # 2. Python Integration Test
    python_success = debug_nova_python_integration()
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("🔍 NOVA DEBUG ZUSAMMENFASSUNG")
    print("=" * 80)
    
    if manual_success:
        print("✅ Nova CLI Commands funktionieren")
    else:
        print("❌ Nova CLI Commands haben Probleme")
    
    if python_success:
        print("✅ Nova Python Integration funktioniert")
        print("🎉 RECURSIVE SNARKs sind verwendbar!")
    else:
        print("❌ Nova Python Integration hat Probleme")
        print("🔧 Braucht Debugging/Fixes")
    
    return manual_success and python_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
