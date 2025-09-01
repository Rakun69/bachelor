#!/usr/bin/env python3
"""
🐳 DOCKER IoT RESOURCE CONSTRAINTS TEST
Simuliert echte IoT-Device-Limitierungen
"""

import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def create_docker_test_script():
    """Erstellt Test-Script für Docker Container"""
    script_content = '''#!/usr/bin/env python3
import sys
import time
import json
from pathlib import Path

# Add project paths
project_root = Path("/app")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def iot_constrained_test(num_items, test_type):
    """Test unter IoT-Constraints"""
    print(f"🔬 IoT-Constrained Test: {num_items} items ({test_type})")
    
    start_time = time.perf_counter()
    
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
            
            for reading in temp_readings:
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                if result.success:
                    successful += 1
                    total_proof_size += result.metrics.proof_size
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size / 1024,
                "throughput": successful / total_time if total_time > 0 else 0
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
                "throughput": num_items / total_time if total_time > 0 else 0
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time
        }

if __name__ == "__main__":
    import sys
    num_items = int(sys.argv[1])
    test_type = sys.argv[2]
    
    result = iot_constrained_test(num_items, test_type)
    print(json.dumps(result))
'''
    return script_content

def create_dockerfile():
    """Erstellt Dockerfile für IoT-Constraints"""
    dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install ZoKrates
RUN curl -LSfs get.zokrat.es | sh
ENV PATH="/root/.zokrates/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \\
    psutil \\
    numpy \\
    pandas \\
    matplotlib \\
    seaborn \\
    scipy \\
    scikit-learn

# Set memory and CPU limits via environment
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "docker_test.py"]
'''
    return dockerfile_content

def run_docker_iot_test(num_items, test_type, memory_limit="512m", cpu_limit="0.5"):
    """Führt Test in Docker Container mit IoT-Constraints aus"""
    
    # Create temporary directory for Docker context
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy project to temp directory
        subprocess.run([
            "cp", "-r", str(project_root), str(temp_path / "bachelor")
        ], check=True)
        
        docker_project = temp_path / "bachelor"
        
        # Create Docker files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_dockerfile())
        
        script_path = docker_project / "docker_test.py"
        with open(script_path, 'w') as f:
            f.write(create_docker_test_script())
        
        # Build Docker image (only once per session)
        image_name = f"iot-zksnark-test-{memory_limit}-{cpu_limit}".replace(".", "")
        
        # Check if image already exists
        check_cmd = ["docker", "images", "-q", image_name]
        existing = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if not existing.stdout.strip():
            print(f"🐳 Building Docker image with IoT constraints...")
            build_cmd = [
                "docker", "build", 
                "-t", image_name,
                str(docker_project)
            ]
            
            try:
                result = subprocess.run(build_cmd, check=True, capture_output=True, text=True, timeout=600)
                print(f"✅ Docker image built successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Docker build failed: {e.stderr}")
                return None
            except subprocess.TimeoutExpired:
                print(f"❌ Docker build timed out")
                return None
        else:
            print(f"✅ Using existing Docker image: {image_name}")
        
        # Run Docker container with resource limits
        print(f"🚀 Running test in IoT-constrained container...")
        print(f"   Memory: {memory_limit}, CPU: {cpu_limit}")
        
        run_cmd = [
            "docker", "run", "--rm",
            f"--memory={memory_limit}",
            f"--cpus={cpu_limit}",
            image_name,
            "python3", "docker_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse JSON output
                output_lines = result.stdout.strip().split('\n')
                json_line = output_lines[-1]  # Last line should be JSON
                return json.loads(json_line)
            else:
                print(f"❌ Docker run failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Docker test timed out after 5 minutes")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Docker output: {e}")
            print(f"Raw output: {result.stdout}")
            return None

def docker_crossover_analysis():
    """Crossover-Analyse unter Docker IoT-Constraints"""
    print("🐳 DOCKER IoT RESOURCE CONSTRAINTS TEST")
    print("Simuliert echte IoT-Device-Limitierungen")
    print("=" * 60)
    
    # IoT Device Constraints
    iot_configs = [
        {"name": "Low-End IoT", "memory": "256m", "cpu": "0.25"},
        {"name": "Mid-Range IoT", "memory": "512m", "cpu": "0.5"},
        {"name": "High-End IoT", "memory": "1g", "cpu": "1.0"}
    ]
    
    test_items = [85, 90, 95, 100]
    all_results = {}
    
    for config in iot_configs:
        print(f"\n🔬 TESTING: {config['name']}")
        print(f"   Memory: {config['memory']}, CPU: {config['cpu']}")
        print("-" * 40)
        
        config_results = []
        
        for num_items in test_items:
            print(f"\n📊 Testing {num_items} items...")
            
            # Standard SNARK test
            print("   🔧 Standard SNARKs: ", end="", flush=True)
            std_result = run_docker_iot_test(
                num_items, "standard", 
                config["memory"], config["cpu"]
            )
            
            if std_result and std_result["success"]:
                print(f"{std_result['total_time']:.2f}s")
            else:
                print("❌ Failed")
                continue
            
            # Recursive SNARK test  
            print("   🚀 Recursive SNARKs: ", end="", flush=True)
            rec_result = run_docker_iot_test(
                num_items, "recursive",
                config["memory"], config["cpu"]
            )
            
            if rec_result and rec_result["success"]:
                print(f"{rec_result['total_time']:.2f}s")
            else:
                print("❌ Failed")
                continue
            
            # Compare results
            ratio = rec_result["total_time"] / std_result["total_time"]
            winner = "Recursive" if ratio < 1.0 else "Standard"
            advantage = abs(1.0 - ratio) * 100
            
            result_entry = {
                "num_items": num_items,
                "standard_time": std_result["total_time"],
                "recursive_time": rec_result["total_time"],
                "ratio": ratio,
                "winner": winner,
                "advantage_percent": advantage,
                "standard_throughput": std_result["throughput"],
                "recursive_throughput": rec_result["throughput"],
                "iot_config": config
            }
            
            config_results.append(result_entry)
            print(f"   📊 {winner} wins by {advantage:.1f}%")
        
        all_results[config["name"]] = config_results
    
    return all_results

def analyze_docker_results(all_results):
    """Analysiert Docker IoT-Constraint Ergebnisse"""
    print("\n" + "=" * 70)
    print("📊 DOCKER IoT CONSTRAINTS ANALYSIS")
    print("=" * 70)
    
    for config_name, results in all_results.items():
        if not results:
            continue
            
        print(f"\n🔬 {config_name.upper()}:")
        print(f"{'Items':<6} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Winner':<10} {'Advantage':<10}")
        print("-" * 60)
        
        crossover_point = None
        
        for result in results:
            items = result["num_items"]
            std_time = result["standard_time"]
            rec_time = result["recursive_time"]
            ratio = result["ratio"]
            winner = result["winner"]
            advantage = result["advantage_percent"]
            
            print(f"{items:<6} {std_time:<8.2f} {rec_time:<8.2f} {ratio:<8.3f} {winner:<10} {advantage:<10.1f}%")
            
            if winner == "Recursive" and crossover_point is None:
                crossover_point = items
        
        if crossover_point:
            print(f"   🎯 Crossover Point: {crossover_point} items")
        else:
            print(f"   ⚠️  No crossover found in tested range")

def main():
    """Hauptfunktion"""
    print("🐳 Checking Docker availability...")
    
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available or not installed")
        print("Please install Docker to run IoT constraint tests")
        return False
    
    results = docker_crossover_analysis()
    
    if results:
        analyze_docker_results(results)
        
        # Save results
        results_dir = project_root / "data" / "docker_iot_constraints"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "docker_iot_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "docker_iot_results": results,
                "test_timestamp": time.time(),
                "constraints_tested": ["256m/0.25cpu", "512m/0.5cpu", "1g/1.0cpu"]
            }, f, indent=2)
        
        print(f"\n💾 Docker IoT results saved: {results_file}")
        print(f"✅ IoT RESOURCE CONSTRAINTS ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No Docker results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
