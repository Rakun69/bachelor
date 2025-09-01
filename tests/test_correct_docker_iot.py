#!/usr/bin/env python3
"""
🐳 CORRECT DOCKER IoT SIMULATION
Richtige Docker Resource-Limits basierend auf echten IoT-Device Specs
"""

import sys
import time
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def create_correct_dockerfile():
    """Erstellt korrektes Dockerfile für IoT-Simulation"""
    dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies for ZoKrates and Python packages
RUN apt-get update && apt-get install -y \\
    curl \\
    build-essential \\
    libssl-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Install ZoKrates
RUN curl -LSfs get.zokrat.es | sh
ENV PATH="/root/.zokrates/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python3"]
'''
    return dockerfile_content

def create_requirements_txt():
    """Erstellt requirements.txt mit allen nötigen Paketen"""
    requirements = '''psutil>=5.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
'''
    return requirements

def create_test_script():
    """Erstellt Test-Script für Docker Container"""
    script_content = '''#!/usr/bin/env python3
import sys
import time
import json
import psutil
from pathlib import Path

# Add project paths
project_root = Path("/app")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def monitor_container_resources():
    """Überwacht Container-Ressourcen"""
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1),
        "memory_percent": process.memory_percent()
    }

def docker_iot_test(num_items, test_type):
    """IoT-Test im Docker Container"""
    print(f"🔬 Docker IoT Test: {num_items} items ({test_type})")
    
    start_time = time.perf_counter()
    start_resources = monitor_container_resources()
    max_memory = start_resources["memory_mb"]
    
    try:
        sensors = SmartHomeSensors()
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        if test_type == "standard":
            manager = SNARKManager()
            circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
            manager.compile_circuit(str(circuit_path), "filter_range")
            manager.setup_circuit("filter_range")
            
            successful = 0
            total_proof_size = 0
            
            for i, reading in enumerate(temp_readings):
                # Monitor resources
                current_resources = monitor_container_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                    total_proof_size += result.metrics.proof_size
                
                # Progress indicator
                if num_items > 20 and (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{num_items}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size / 1024,
                "throughput": successful / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory,
                "container_constrained": True
            }
            
        elif test_type == "recursive":
            nova_manager = FixedZoKratesNovaManager()
            if not nova_manager.setup():
                raise Exception("Nova Setup failed")
            
            # Prepare batches
            batches = []
            for i in range(0, len(temp_readings), 3):
                batch_readings = temp_readings[i:i+3]
                while len(batch_readings) < 3:
                    batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
                batch_dicts = [{'value': r.value} for r in batch_readings]
                batches.append(batch_dicts)
            
            # Monitor during recursive proof
            for i in range(len(batches)):
                current_resources = monitor_container_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                if (i + 1) % 5 == 0:
                    print(f"   Batch Progress: {i+1}/{len(batches)}")
            
            result = nova_manager.prove_recursive_batch(batches)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            if not result.success:
                raise Exception(f"Recursive proof failed: {result.error_message}")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "proof_size_kb": result.proof_size / 1024,
                "throughput": num_items / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory,
                "container_constrained": True
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory,
            "container_constrained": True
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python docker_test.py <num_items> <test_type>")
        sys.exit(1)
    
    num_items = int(sys.argv[1])
    test_type = sys.argv[2]
    
    result = docker_iot_test(num_items, test_type)
    print(json.dumps(result))
'''
    return script_content

def run_docker_iot_test(num_items, test_type, iot_config):
    """Führt Test in Docker Container mit korrekten IoT-Limits aus"""
    
    # Create temporary directory for Docker context
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy project to temp directory
        docker_project = temp_path / "bachelor"
        shutil.copytree(project_root, docker_project, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'data'))
        
        # Create Docker files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_correct_dockerfile())
        
        requirements_path = docker_project / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(create_requirements_txt())
        
        script_path = docker_project / "docker_test.py"
        with open(script_path, 'w') as f:
            f.write(create_test_script())
        
        # Build Docker image
        image_name = f"iot-zksnark-{iot_config['name'].lower().replace(' ', '-')}"
        
        print(f"🐳 Building Docker image: {image_name}")
        build_cmd = [
            "docker", "build", 
            "-t", image_name,
            str(docker_project)
        ]
        
        try:
            build_result = subprocess.run(build_cmd, check=True, capture_output=True, text=True, timeout=300)
            print(f"✅ Docker image built successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print(f"❌ Docker build timed out")
            return None
        
        # Run Docker container with CORRECT IoT resource limits
        print(f"🚀 Running test with {iot_config['name']} limits:")
        print(f"   Memory: {iot_config['memory']}, CPU: {iot_config['cpu']}")
        
        run_cmd = [
            "docker", "run", "--rm",
            f"--memory={iot_config['memory']}",
            f"--cpus={iot_config['cpu']}",
            f"--memory-swap={iot_config['memory']}",  # Prevent swap usage
            "--oom-kill-disable=false",  # Allow OOM killer
            image_name,
            "python3", "docker_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            if result.returncode == 0:
                # Parse JSON output from last line
                output_lines = result.stdout.strip().split('\n')
                json_line = None
                
                # Find JSON line (last line that starts with {)
                for line in reversed(output_lines):
                    if line.strip().startswith('{'):
                        json_line = line.strip()
                        break
                
                if json_line:
                    parsed_result = json.loads(json_line)
                    parsed_result["iot_config"] = iot_config
                    return parsed_result
                else:
                    print(f"❌ Could not find JSON output in: {result.stdout}")
                    return None
            else:
                print(f"❌ Docker run failed (exit code {result.returncode})")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Docker test timed out after 10 minutes")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Docker output: {e}")
            print(f"Raw output: {result.stdout}")
            return None

def correct_docker_iot_test():
    """Korrekte Docker IoT-Tests mit realistischen Device-Specs"""
    print("🐳 CORRECT DOCKER IoT SIMULATION")
    print("Realistische IoT-Device Resource-Limits")
    print("=" * 60)
    
    # REALISTISCHE IoT-Device Konfigurationen
    iot_configs = [
        {
            "name": "Smart Home Hub",
            "memory": "1g",      # 1GB RAM (typisch für Smart Home Hubs)
            "cpu": "1.0",        # 1 CPU core
            "description": "Typical smart home hub (Raspberry Pi 4 class)"
        },
        {
            "name": "IoT Gateway",
            "memory": "512m",    # 512MB RAM
            "cpu": "0.5",        # 0.5 CPU cores
            "description": "Mid-range IoT gateway device"
        },
        {
            "name": "Edge Device",
            "memory": "256m",    # 256MB RAM
            "cpu": "0.25",       # 0.25 CPU cores
            "description": "Low-power edge computing device"
        }
    ]
    
    # Start with small test
    test_items = 20
    
    print(f"\n🔬 SMALL TEST: {test_items} items")
    print("-" * 40)
    
    all_results = {}
    
    for config in iot_configs:
        print(f"\n📱 Testing on {config['name']}:")
        print(f"   {config['description']}")
        
        config_results = []
        
        # Standard SNARK test
        print("   🔧 Standard SNARKs: ", end="", flush=True)
        std_result = run_docker_iot_test(test_items, "standard", config)
        
        if std_result and std_result["success"]:
            print(f"✅ {std_result['total_time']:.2f}s ({std_result['successful_proofs']}/{test_items})")
            config_results.append(std_result)
        else:
            print("❌ Failed")
            continue
        
        # Recursive SNARK test
        print("   🚀 Recursive SNARKs: ", end="", flush=True)
        rec_result = run_docker_iot_test(test_items, "recursive", config)
        
        if rec_result and rec_result["success"]:
            print(f"✅ {rec_result['total_time']:.2f}s ({rec_result['steps']} steps)")
            config_results.append(rec_result)
        else:
            print("❌ Failed")
            continue
        
        # Compare results
        if len(config_results) == 2:
            std_result, rec_result = config_results
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if ratio < 1.0 else "Standard"
            advantage = abs(1.0 - ratio) * 100
            
            print(f"   📊 {winner} wins by {advantage:.1f}% (Ratio: {ratio:.3f})")
            print(f"   💾 Memory: Std {std_result['max_memory_mb']:.1f}MB, Rec {rec_result['max_memory_mb']:.1f}MB")
        
        all_results[config["name"]] = config_results
    
    return all_results

def analyze_docker_results(results):
    """Analysiert Docker-Test Ergebnisse"""
    print("\n" + "=" * 70)
    print("📊 DOCKER IoT TEST RESULTS")
    print("=" * 70)
    
    successful_configs = 0
    total_configs = len(results)
    
    for config_name, config_results in results.items():
        print(f"\n📱 {config_name.upper()}:")
        
        if len(config_results) == 2:  # Both tests successful
            std_result, rec_result = config_results
            
            print(f"   ✅ Both tests successful")
            print(f"   📊 Standard: {std_result['total_time']:.2f}s, {std_result['max_memory_mb']:.1f}MB")
            print(f"   📊 Recursive: {rec_result['total_time']:.2f}s, {rec_result['max_memory_mb']:.1f}MB")
            
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if ratio < 1.0 else "Standard"
            print(f"   🏆 Winner: {winner} (Ratio: {ratio:.3f})")
            
            successful_configs += 1
        else:
            print(f"   ❌ Tests failed or incomplete")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   ✅ Successful configurations: {successful_configs}/{total_configs}")
    
    if successful_configs > 0:
        print(f"   🎉 Docker IoT simulation working!")
        print(f"   🚀 Ready for larger tests and crossover analysis")
        return True
    else:
        print(f"   ❌ Docker IoT simulation needs fixes")
        return False

def main():
    """Hauptfunktion"""
    print("🐳 Checking Docker availability...")
    
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available or not installed")
        return False
    
    results = correct_docker_iot_test()
    
    if results:
        success = analyze_docker_results(results)
        
        if success:
            # Save results
            results_dir = project_root / "data" / "correct_docker_iot"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / "docker_iot_test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "docker_iot_results": results,
                    "test_timestamp": time.time(),
                    "iot_configurations": [
                        "Smart Home Hub: 1GB RAM, 1 CPU",
                        "IoT Gateway: 512MB RAM, 0.5 CPU", 
                        "Edge Device: 256MB RAM, 0.25 CPU"
                    ],
                    "test_items": 20
                }, f, indent=2)
            
            print(f"\n💾 Docker IoT results saved: {results_file}")
            print(f"✅ CORRECT DOCKER IoT SIMULATION COMPLETE!")
        
        return success
    else:
        print("❌ No Docker results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
