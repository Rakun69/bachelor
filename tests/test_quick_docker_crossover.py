#!/usr/bin/env python3
"""
⚡ QUICK DOCKER CROSSOVER 88-97
Verwendet den funktionierenden Docker-Ansatz vom kleinen Test
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

def create_minimal_dockerfile():
    """Minimales Dockerfile - genau wie beim funktionierenden Test"""
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

def create_minimal_requirements():
    """Minimale requirements.txt"""
    requirements = '''psutil>=5.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
'''
    return requirements

def create_crossover_test_script():
    """Crossover-Test Script - genau wie der funktionierende Test"""
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

def docker_crossover_test(num_items, test_type):
    """Crossover-Test im Docker Container"""
    print(f"🔬 Docker Crossover Test: {num_items} items ({test_type})")
    
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
            individual_proof_sizes = []
            
            for i, reading in enumerate(temp_readings):
                # Monitor resources
                current_resources = monitor_container_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                result = manager.generate_proof("filter_range", inputs)
                
                if result.success:
                    successful += 1
                    individual_proof_sizes.append(result.metrics.proof_size)
                
                # Progress indicator
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
    
    result = docker_crossover_test(num_items, test_type)
    print(json.dumps(result))
'''
    return script_content

def copy_essential_files_only(source_dir, target_dir):
    """Kopiert nur die essentiellen Dateien - vermeidet copytree Probleme"""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Essential directories to copy
    essential_dirs = ["src", "circuits"]
    
    for dir_name in essential_dirs:
        source_path = source_dir / dir_name
        target_path = target_dir / dir_name
        
        if source_path.exists():
            shutil.copytree(source_path, target_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

def run_quick_docker_crossover_test(num_items, test_type, iot_config):
    """Führt Crossover-Test aus - verwendet funktionierenden Ansatz"""
    
    # Create temporary directory for Docker context
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy only essential files (avoid copytree problems)
        docker_project = temp_path / "bachelor"
        copy_essential_files_only(project_root, docker_project)
        
        # Create Docker files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_minimal_dockerfile())
        
        requirements_path = docker_project / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(create_minimal_requirements())
        
        script_path = docker_project / "docker_test.py"
        with open(script_path, 'w') as f:
            f.write(create_crossover_test_script())
        
        # Build Docker image (same as working test)
        image_name = f"iot-crossover-{iot_config['name'].lower().replace(' ', '-')}"
        
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
        
        # Run Docker container (same as working test)
        print(f"🚀 Running test with {iot_config['name']} limits:")
        print(f"   Memory: {iot_config['memory']}, CPU: {iot_config['cpu']}")
        
        run_cmd = [
            "docker", "run", "--rm",
            f"--memory={iot_config['memory']}",
            f"--cpus={iot_config['cpu']}",
            f"--memory-swap={iot_config['memory']}",  # Same as working test
            "--oom-kill-disable=false",
            image_name,
            "python3", "docker_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
            
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
                    print(f"❌ Could not find JSON output")
                    return None
            else:
                print(f"❌ Docker run failed (exit code {result.returncode})")
                print(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Docker test timed out after 15 minutes")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Docker output: {e}")
            return None

def quick_docker_crossover_analysis():
    """Quick Crossover-Analyse - verwendet funktionierenden Ansatz"""
    print("⚡ QUICK DOCKER CROSSOVER ANALYSIS 88-97 ITEMS")
    print("Verwendet funktionierenden Docker-Ansatz vom kleinen Test")
    print("=" * 70)
    
    # Use Smart Home Hub config (same as working test)
    iot_config = {
        "name": "Smart Home Hub",
        "memory": "1g",
        "cpu": "1.0",
        "description": "Same config as working small test"
    }
    
    print(f"📱 Using {iot_config['name']} configuration:")
    print(f"   Memory: {iot_config['memory']}, CPU: {iot_config['cpu']}")
    
    # Test key points around crossover (smaller range for speed)
    test_points = [88, 90, 95, 97]  # Reduced for speed
    results = []
    
    for num_items in test_points:
        print(f"\n🔬 TESTING: {num_items} Items")
        print("-" * 40)
        
        # Standard Test
        print(f"   📊 Standard SNARKs: ", end="", flush=True)
        std_result = run_quick_docker_crossover_test(num_items, "standard", iot_config)
        
        if std_result and std_result["success"]:
            print(f"✅ {std_result['total_time']:.2f}s ({std_result['successful_proofs']}/{num_items})")
        else:
            print("❌ Failed")
            continue
        
        # Recursive Test
        print(f"   🚀 Recursive SNARKs: ", end="", flush=True)
        rec_result = run_quick_docker_crossover_test(num_items, "recursive", iot_config)
        
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

def analyze_quick_crossover_results(results):
    """Analysiert Quick Crossover-Ergebnisse"""
    print("\n" + "=" * 80)
    print("📊 QUICK DOCKER CROSSOVER RESULTS")
    print("=" * 80)
    
    if not results:
        print("❌ No results to analyze")
        return None
    
    print(f"\n✅ CROSSOVER ANALYSIS UNDER IoT CONSTRAINTS:")
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
    
    print(f"\n🎯 CROSSOVER POINT UNDER IoT CONSTRAINTS:")
    if crossover_point:
        print(f"   🎉 FOUND: {crossover_point} items")
        print(f"   📊 Recursive SNARKs become efficient at {crossover_point}+ items")
    else:
        print(f"   ⚠️  NO CROSSOVER found in tested range under IoT constraints")
        print(f"   📊 Standard SNARKs remain superior for all tested batch sizes")
        
        # Check trend
        if len(results) >= 2:
            last_ratio = results[-1]["ratio"]
            first_ratio = results[0]["ratio"]
            if last_ratio < first_ratio:
                print(f"   📈 Trend: Recursive getting better with more items (Ratio: {first_ratio:.3f} → {last_ratio:.3f})")
                estimated_crossover = int(results[-1]["num_items"] * (1 / last_ratio))
                print(f"   🔮 Estimated crossover: ~{estimated_crossover} items")
    
    return crossover_point

def main():
    """Hauptfunktion"""
    print("⚡ Starting Quick Docker Crossover Analysis...")
    
    results = quick_docker_crossover_analysis()
    
    if results:
        crossover_point = analyze_quick_crossover_results(results)
        
        # Save results
        results_dir = project_root / "data" / "quick_docker_crossover"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "quick_crossover_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "quick_crossover_results": results,
                "crossover_point": crossover_point,
                "test_timestamp": time.time(),
                "iot_configuration": "Smart Home Hub (1GB RAM, 1 CPU)",
                "test_points": [88, 90, 95, 97],
                "methodology": "Quick Docker test using working approach"
            }, f, indent=2)
        
        print(f"\n💾 Quick crossover results saved: {results_file}")
        print(f"✅ QUICK DOCKER CROSSOVER ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No crossover results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
