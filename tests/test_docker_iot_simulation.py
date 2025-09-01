#!/usr/bin/env python3
"""
🐳 DOCKER IOT SIMULATION TEST
Simuliert IoT-Device-Limitationen mit Docker Resource Constraints
"""

import sys
import time
import subprocess
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_docker_test_script():
    """Erstellt ein Docker-Test-Script"""
    docker_script = project_root / "docker_zk_test.py"
    
    script_content = '''#!/usr/bin/env python3
import sys
import time
import psutil
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def get_system_info():
    """Zeigt verfügbare Ressourcen"""
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_mb": psutil.virtual_memory().total / (1024*1024),
        "memory_available_mb": psutil.virtual_memory().available / (1024*1024)
    }

def test_constrained_standard_snark(num_items=20):
    """Test Standard SNARK unter Resource-Constraints"""
    print(f"📊 Standard SNARK Test: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        manager = SNARKManager()
        
        # System Info
        sys_info = get_system_info()
        print(f"   💻 CPU Cores: {sys_info['cpu_count']}")
        print(f"   💾 Memory: {sys_info['memory_mb']:.0f} MB available")
        
        # Setup
        circuit_path = Path("circuits/basic/filter_range.zok")
        manager.compile_circuit(str(circuit_path), "filter_range")
        manager.setup_circuit("filter_range")
        
        # Generate data
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Measure performance
        start_time = time.time()
        successful = 0
        
        for reading in temp_readings:
            secret_value = max(10, min(50, int(reading.value)))
            inputs = ["10", "50", str(secret_value)]
            
            result = manager.generate_proof("filter_range", inputs)
            if result.success:
                successful += 1
        
        total_time = time.time() - start_time
        
        return {
            "type": "standard",
            "num_items": num_items,
            "successful": successful,
            "total_time": total_time,
            "throughput": successful / total_time,
            "system_info": sys_info
        }
        
    except Exception as e:
        return {"error": str(e)}

def test_constrained_recursive_snark(num_items=20):
    """Test Recursive SNARK unter Resource-Constraints"""
    print(f"🚀 Recursive SNARK Test: {num_items} Items")
    
    try:
        sensors = SmartHomeSensors()
        nova_manager = FixedZoKratesNovaManager()
        
        # System Info
        sys_info = get_system_info()
        
        # Setup
        if not nova_manager.setup():
            return {"error": "Nova setup failed"}
        
        # Generate data
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        # Prepare batches
        batches = []
        for i in range(0, len(temp_readings), 3):
            batch_readings = temp_readings[i:i+3]
            while len(batch_readings) < 3:
                batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
            
            batch_dicts = [{'value': r.value} for r in batch_readings]
            batches.append(batch_dicts)
        
        # Measure performance
        start_time = time.time()
        result = nova_manager.prove_recursive_batch(batches)
        total_time = time.time() - start_time
        
        if result.success:
            return {
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "throughput": num_items / total_time,
                "proof_size_kb": result.proof_size / 1024,
                "system_info": sys_info
            }
        else:
            return {"error": "Recursive proof failed"}
            
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main function for Docker test"""
    print("🐳 DOCKER IOT CONSTRAINT TEST")
    print("=" * 50)
    
    # Test both approaches
    std_result = test_constrained_standard_snark(20)
    rec_result = test_constrained_recursive_snark(20)
    
    results = {
        "standard": std_result,
        "recursive": rec_result,
        "timestamp": time.time()
    }
    
    # Output results as JSON for parent process
    print("\\n📊 RESULTS:")
    print(json.dumps(results, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(docker_script, 'w') as f:
        f.write(script_content)
    
    return docker_script

def run_docker_iot_test(device_profile: dict) -> dict:
    """Führt IoT-Test in Docker mit Resource-Constraints aus"""
    print(f"🐳 Docker Test: {device_profile['name']}")
    
    try:
        # Erstelle Test-Script
        docker_script = create_docker_test_script()
        
        # Docker Command mit Resource-Limits
        docker_cmd = [
            "docker", "run", "--rm",
            f"--cpus={device_profile['cpu_limit']}",
            f"--memory={device_profile['memory_limit']}",
            "-v", f"{project_root}:/workspace",
            "-w", "/workspace",
            "python:3.9-slim",
            "bash", "-c",
            f"pip install psutil > /dev/null 2>&1 && python {docker_script.name}"
        ]
        
        print(f"   🔧 CPU Limit: {device_profile['cpu_limit']}")
        print(f"   💾 Memory Limit: {device_profile['memory_limit']}")
        
        # Führe Docker-Test aus
        start_time = time.time()
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 Minuten Timeout
        )
        docker_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse JSON output
            output_lines = result.stdout.strip().split('\n')
            json_start = -1
            for i, line in enumerate(output_lines):
                if line.strip() == "📊 RESULTS:":
                    json_start = i + 1
                    break
            
            if json_start > 0:
                json_output = '\n'.join(output_lines[json_start:])
                test_results = json.loads(json_output)
                test_results["docker_execution_time"] = docker_time
                test_results["device_profile"] = device_profile
                
                print(f"   ✅ Erfolg in {docker_time:.1f}s")
                return test_results
            else:
                print(f"   ❌ Keine JSON-Ausgabe gefunden")
                return {"error": "No JSON output found", "stdout": result.stdout}
        else:
            print(f"   ❌ Docker-Fehler: {result.stderr}")
            return {"error": result.stderr, "returncode": result.returncode}
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout nach 5 Minuten")
        return {"error": "Docker test timeout"}
    except Exception as e:
        print(f"   💥 Fehler: {e}")
        return {"error": str(e)}

def main():
    """Hauptfunktion"""
    print("🐳 DOCKER IOT SIMULATION TEST")
    print("Simuliert verschiedene IoT-Device-Limitationen mit Docker")
    print("=" * 70)
    
    # Verschiedene IoT-Device-Profile
    device_profiles = [
        {
            "name": "High-End IoT (Raspberry Pi 4)",
            "cpu_limit": "1.0",      # 1 CPU Core
            "memory_limit": "512m"   # 512 MB RAM
        },
        {
            "name": "Mid-Range IoT (Raspberry Pi Zero)",
            "cpu_limit": "0.5",      # 0.5 CPU Core
            "memory_limit": "256m"   # 256 MB RAM
        },
        {
            "name": "Low-End IoT (ESP32-like)",
            "cpu_limit": "0.1",      # 0.1 CPU Core
            "memory_limit": "64m"    # 64 MB RAM
        }
    ]
    
    all_results = []
    
    # Prüfe Docker-Verfügbarkeit
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker verfügbar")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker nicht verfügbar - Test wird übersprungen")
        return False
    
    # Führe Tests für alle Device-Profile aus
    for profile in device_profiles:
        print(f"\n🔬 TESTE: {profile['name']}")
        print("-" * 50)
        
        result = run_docker_iot_test(profile)
        all_results.append(result)
        
        time.sleep(2)  # Kurze Pause zwischen Tests
    
    # Analyse der Ergebnisse
    print("\n" + "=" * 70)
    print("📊 DOCKER IOT SIMULATION ANALYSE")
    print("=" * 70)
    
    print(f"{'Device':<25} {'Type':<10} {'Time(s)':<8} {'Throughput':<12} {'Status':<10}")
    print("-" * 70)
    
    for result in all_results:
        if "error" not in result:
            device_name = result["device_profile"]["name"][:24]
            
            # Standard SNARK Ergebnisse
            if "standard" in result and "error" not in result["standard"]:
                std = result["standard"]
                print(f"{device_name:<25} {'Standard':<10} {std['total_time']:<8.2f} "
                      f"{std['throughput']:<12.1f} {'✅ OK':<10}")
            
            # Recursive SNARK Ergebnisse
            if "recursive" in result and "error" not in result["recursive"]:
                rec = result["recursive"]
                print(f"{'':<25} {'Recursive':<10} {rec['total_time']:<8.2f} "
                      f"{rec['throughput']:<12.1f} {'✅ OK':<10}")
        else:
            device_name = result.get("device_profile", {}).get("name", "Unknown")[:24]
            print(f"{device_name:<25} {'Both':<10} {'N/A':<8} {'N/A':<12} {'❌ FAIL':<10}")
    
    # Speichere Ergebnisse
    results_dir = project_root / "data" / "docker_iot_simulation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "docker_iot_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Ergebnisse gespeichert: {results_file}")
    
    # Cleanup
    docker_script = project_root / "docker_zk_test.py"
    if docker_script.exists():
        docker_script.unlink()
    
    return len([r for r in all_results if "error" not in r]) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 DOCKER IOT SIMULATION ABGESCHLOSSEN!' if success else '❌ DOCKER IOT SIMULATION FEHLGESCHLAGEN!'}")
    sys.exit(0 if success else 1)
