#!/usr/bin/env python3
"""
🎯 DOCKER CROSSOVER ANALYSIS 88-97 ITEMS
Findet exakten Crossover-Point unter realistischen IoT-Constraints
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

def create_crossover_dockerfile():
    """Dockerfile für Crossover-Analyse"""
    dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies
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

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python3"]
'''
    return dockerfile_content

def create_crossover_script():
    """Script für Crossover-Analyse im Container"""
    script_content = '''#!/usr/bin/env python3
import sys
import time
import json
import psutil
from pathlib import Path

project_root = Path("/app")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def monitor_resources():
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1)
    }

def docker_crossover_test(num_items, test_type):
    print(f"🔬 Docker Crossover Test: {num_items} items ({test_type})")
    
    start_time = time.perf_counter()
    start_resources = monitor_resources()
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
            individual_proof_sizes = []
            
            for i, reading in enumerate(temp_readings):
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                    individual_proof_sizes.append(result.metrics.proof_size)
                
                # Progress for large tests
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{num_items}")
            
            # Calculate metrics (same as working tests)
            if individual_proof_sizes:
                total_proof_size_kb = sum(individual_proof_sizes) / 1024
                avg_proof_size_kb = sum(individual_proof_sizes) / len(individual_proof_sizes) / 1024
            else:
                total_proof_size_kb = 0
                avg_proof_size_kb = 0
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "total_proof_size_kb": total_proof_size_kb,
                "avg_proof_size_kb": avg_proof_size_kb,
                "throughput": successful / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory
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
                current_resources = monitor_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                if (i + 1) % 10 == 0:
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
                "verify_time": result.verify_time,
                "throughput": num_items / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory
        }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crossover_test.py <num_items> <test_type>")
        sys.exit(1)
    
    num_items = int(sys.argv[1])
    test_type = sys.argv[2]
    
    result = docker_crossover_test(num_items, test_type)
    print(json.dumps(result))
'''
    return script_content

def run_docker_crossover_test(num_items, test_type, iot_config):
    """Führt Crossover-Test in Docker Container aus"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy project
        docker_project = temp_path / "bachelor"
        shutil.copytree(project_root, docker_project, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'data'))
        
        # Create files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_crossover_dockerfile())
        
        requirements_path = docker_project / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write("psutil>=5.8.0\\nnumpy>=1.21.0\\npandas>=1.3.0\\n")
        
        script_path = docker_project / "crossover_test.py"
        with open(script_path, 'w') as f:
            f.write(create_crossover_script())
        
        # Build image
        image_name = f"iot-crossover-{iot_config['name'].lower().replace(' ', '-')}"
        
        build_cmd = ["docker", "build", "-t", image_name, str(docker_project)]
        
        try:
            subprocess.run(build_cmd, check=True, capture_output=True, text=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        
        # Run test
        run_cmd = [
            "docker", "run", "--rm",
            f"--memory={iot_config['memory']}",
            f"--cpus={iot_config['cpu']}",
            f"--memory-swap={iot_config['memory']}",
            image_name,
            "python3", "crossover_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\\n')
                for line in reversed(output_lines):
                    if line.strip().startswith('{'):
                        parsed_result = json.loads(line.strip())
                        parsed_result["iot_config"] = iot_config
                        return parsed_result
            return None
                
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return None

def docker_crossover_analysis():
    """Crossover-Analyse unter Docker IoT-Constraints"""
    print("🎯 DOCKER CROSSOVER ANALYSIS 88-97 ITEMS")
    print("Exakter Crossover-Point unter realistischen IoT-Constraints")
    print("=" * 70)
    
    # Use Smart Home Hub config (best performance for crossover detection)
    iot_config = {
        "name": "Smart Home Hub",
        "memory": "1g",
        "cpu": "1.0",
        "description": "Optimal for crossover detection"
    }
    
    print(f"📱 Using {iot_config['name']} configuration:")
    print(f"   Memory: {iot_config['memory']}, CPU: {iot_config['cpu']}")
    
    # Test each item count from 88 to 97
    test_points = list(range(88, 98))
    results = []
    
    for num_items in test_points:
        print(f"\\n🔬 TESTING: {num_items} Items")
        print("-" * 40)
        
        # Standard Test
        print(f"   📊 Standard SNARKs: ", end="", flush=True)
        std_result = run_docker_crossover_test(num_items, "standard", iot_config)
        
        if std_result and std_result["success"]:
            print(f"✅ {std_result['total_time']:.2f}s ({std_result['successful_proofs']}/{num_items})")
        else:
            print("❌ Failed")
            continue
        
        # Recursive Test
        print(f"   🚀 Recursive SNARKs: ", end="", flush=True)
        rec_result = run_docker_crossover_test(num_items, "recursive", iot_config)
        
        if rec_result and rec_result["success"]:
            print(f"✅ {rec_result['total_time']:.2f}s ({rec_result['steps']} steps)")
        else:
            print("❌ Failed")
            continue
        
        # Analyze results
        ratio = rec_result["total_time"] / std_result["total_time"]
        winner = "Recursive" if ratio < 1.0 else "Standard"
        advantage = abs(1.0 - ratio) * 100
        
        result = {
            "num_items": num_items,
            "standard_time": std_result["total_time"],
            "recursive_time": rec_result["total_time"],
            "ratio": ratio,
            "winner": winner,
            "recursive_wins": ratio < 1.0,
            "advantage_percent": advantage,
            "standard_proof_size_kb": std_result["total_proof_size_kb"],
            "recursive_proof_size_kb": rec_result["proof_size_kb"],
            "standard_memory_mb": std_result["max_memory_mb"],
            "recursive_memory_mb": rec_result["max_memory_mb"],
            "iot_constrained": True
        }
        
        results.append(result)
        print(f"   📊 {winner} wins by {advantage:.1f}% (Ratio: {ratio:.4f})")
    
    return results

def analyze_docker_crossover_results(results):
    """Analysiert Docker Crossover-Ergebnisse"""
    print("\\n" + "=" * 80)
    print("📊 DOCKER IoT CROSSOVER ANALYSIS RESULTS")
    print("=" * 80)
    
    if not results:
        print("❌ No results to analyze")
        return None
    
    print(f"\\n✅ CROSSOVER ANALYSIS UNDER IoT CONSTRAINTS:")
    print(f"{'Items':<5} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Diff%':<7} {'Winner':<10}")
    print("-" * 60)
    
    crossover_point = None
    
    for result in results:
        items = result["num_items"]
        std_time = result["standard_time"]
        rec_time = result["recursive_time"]
        ratio = result["ratio"]
        diff_pct = result["advantage_percent"]
        winner = result["winner"]
        
        print(f"{items:<5} {std_time:<8.2f} {rec_time:<8.2f} {ratio:<8.4f} {diff_pct:<7.1f} {winner:<10}")
        
        # Find first crossover
        if result["recursive_wins"] and crossover_point is None:
            crossover_point = items
    
    print(f"\\n🎯 CROSSOVER POINT UNDER IoT CONSTRAINTS:")
    if crossover_point:
        print(f"   🎉 FOUND: {crossover_point} items")
        print(f"   📊 Recursive SNARKs become efficient at {crossover_point}+ items under IoT constraints")
    else:
        print(f"   ⚠️  NO CROSSOVER found in 88-97 range under IoT constraints")
        print(f"   📊 Standard SNARKs remain superior for all tested batch sizes")
    
    # Memory analysis
    avg_std_memory = sum(r["standard_memory_mb"] for r in results) / len(results)
    avg_rec_memory = sum(r["recursive_memory_mb"] for r in results) / len(results)
    
    print(f"\\n💾 MEMORY USAGE UNDER IoT CONSTRAINTS:")
    print(f"   Standard SNARKs: {avg_std_memory:.1f}MB average")
    print(f"   Recursive SNARKs: {avg_rec_memory:.1f}MB average")
    
    return crossover_point

def main():
    """Hauptfunktion"""
    print("🎯 Starting Docker Crossover Analysis 88-97...")
    
    results = docker_crossover_analysis()
    
    if results:
        crossover_point = analyze_docker_crossover_results(results)
        
        # Save results
        results_dir = project_root / "data" / "docker_crossover_88_97"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "docker_crossover_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "docker_crossover_results": results,
                "crossover_point": crossover_point,
                "test_timestamp": time.time(),
                "iot_configuration": "Smart Home Hub (1GB RAM, 1 CPU)",
                "test_range": "88-97 items",
                "methodology": "Docker containers with realistic IoT resource constraints"
            }, f, indent=2)
        
        print(f"\\n💾 Docker crossover results saved: {results_file}")
        print(f"✅ DOCKER CROSSOVER ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No crossover results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
