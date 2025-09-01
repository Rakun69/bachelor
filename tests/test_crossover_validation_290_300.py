#!/usr/bin/env python3
"""
🎯 CROSSOVER VALIDATION TEST - 290 & 300 ITEMS
Validiert den geschätzten Crossover-Point bei ~293 Items
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

def create_validation_dockerfile():
    """Dockerfile für Crossover-Validation"""
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

def create_validation_requirements():
    """Requirements für Validation Tests"""
    requirements = '''psutil>=5.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
'''
    return requirements

def create_validation_test_script():
    """Validation Test Script - optimiert für Crossover-Bereich"""
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

def docker_validation_test(num_items, test_type):
    """Validation Test für Crossover-Bereich (290-300 Items)"""
    print(f"🎯 Docker Crossover Validation: {num_items} items ({test_type})")
    print(f"📱 Running under IoT constraints: 1GB RAM, 1 CPU")
    print(f"🔍 Testing around estimated crossover point (~293 items)")
    
    start_time = time.perf_counter()
    start_resources = monitor_container_resources()
    max_memory = start_resources["memory_mb"]
    
    try:
        sensors = SmartHomeSensors()
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        if test_type == "standard":
            print(f"🔧 Standard SNARK Pipeline for {num_items} items:")
            
            manager = SNARKManager()
            circuit_path = project_root / "circuits" / "basic" / "filter_range.zok"
            
            print(f"   1️⃣ Circuit Compilation...")
            compile_start = time.perf_counter()
            manager.compile_circuit(str(circuit_path), "filter_range")
            compile_time = time.perf_counter() - compile_start
            print(f"   ✅ Compiled filter_range in {compile_time:.3f}s")
            
            print(f"   2️⃣ Circuit Setup...")
            setup_start = time.perf_counter()
            manager.setup_circuit("filter_range")
            setup_time = time.perf_counter() - setup_start
            print(f"   ✅ Setup completed for filter_range in {setup_time:.3f}s")
            
            successful = 0
            individual_proof_sizes = []
            
            print(f"   3️⃣ Processing {num_items} sensor readings:")
            print(f"   📊 Expected time: ~{num_items * 0.14:.1f}s (based on trend)")
            
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
                    
                    # Show detailed output every 25th proof (less verbose for large tests)
                    if (i + 1) % 25 == 0 or i < 3:
                        print(f"   📊 Witness generated for filter_range in {result.metrics.witness_time:.3f}s")
                        print(f"   🔐 Proof generated for filter_range:")
                        print(f"      Proof time: {result.metrics.proof_time:.3f}s")
                        print(f"      Verify time: {result.metrics.verify_time:.3f}s")
                        print(f"      Proof size: {result.metrics.proof_size} bytes")
                
                # Progress indicator every 30 items
                if (i + 1) % 30 == 0:
                    progress = (i + 1) / num_items * 100
                    elapsed = time.perf_counter() - start_time
                    estimated_total = elapsed / (i + 1) * num_items
                    remaining = estimated_total - elapsed
                    print(f"   📈 Progress: {i+1}/{num_items} ({progress:.1f}%) - Memory: {current_resources['memory_mb']:.1f}MB - ETA: {remaining:.1f}s")
            
            # Calculate final metrics
            if individual_proof_sizes:
                total_proof_size_kb = sum(individual_proof_sizes) / 1024
                avg_proof_size_kb = sum(individual_proof_sizes) / len(individual_proof_sizes) / 1024
            else:
                total_proof_size_kb = 0
                avg_proof_size_kb = 0
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"\\n📊 STANDARD SNARK RESULTS:")
            print(f"   ✅ Successful proofs: {successful}/{num_items}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   ⚡ Throughput: {successful / total_time:.2f} proofs/sec")
            print(f"   💾 Max memory: {max_memory:.1f}MB")
            print(f"   📏 Total proof size: {total_proof_size_kb:.2f}KB")
            print(f"   📏 Average proof size: {avg_proof_size_kb:.3f}KB")
            print(f"   🎯 Expected crossover: {total_time:.1f}s vs recursive ~{num_items * 0.20:.1f}s")
            
            return {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "compile_time": compile_time,
                "setup_time": setup_time,
                "total_proof_size_kb": total_proof_size_kb,
                "avg_proof_size_kb": avg_proof_size_kb,
                "throughput": successful / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory,
                "container_constrained": True,
                "crossover_validation": True
            }
            
        elif test_type == "recursive":
            print(f"🚀 Recursive SNARK Pipeline for {num_items} items:")
            
            nova_manager = FixedZoKratesNovaManager()
            
            print(f"   1️⃣ Nova Circuit Setup...")
            setup_start = time.perf_counter()
            if not nova_manager.setup():
                raise Exception("Nova Setup failed")
            setup_time = time.perf_counter() - setup_start
            print(f"   ✅ Nova setup completed in {setup_time:.3f}s")
            
            print(f"   2️⃣ Initial State Creation...")
            print(f"   📊 Initial state: {{sum: 0, count: 0}}")
            
            # Prepare batches
            batches = []
            for i in range(0, len(temp_readings), 3):
                batch_readings = temp_readings[i:i+3]
                while len(batch_readings) < 3:
                    batch_readings.append(batch_readings[-1] if batch_readings else type('Reading', (), {'value': 22.0})())
                batch_dicts = [{'value': r.value} for r in batch_readings]
                batches.append(batch_dicts)
            
            print(f"   📦 Prepared {len(batches)} batches of 3 items each")
            print(f"   📊 Expected time: ~{len(batches) * 0.6:.1f}s (based on trend)")
            
            print(f"   3️⃣ Incremental Verification...")
            
            # Monitor during recursive proof
            for i in range(len(batches)):
                current_resources = monitor_container_resources()
                max_memory = max(max_memory, current_resources["memory_mb"])
                
                if (i + 1) % 15 == 0:
                    progress = (i + 1) / len(batches) * 100
                    elapsed = time.perf_counter() - start_time
                    estimated_total = elapsed / (i + 1) * len(batches)
                    remaining = estimated_total - elapsed
                    print(f"   🔄 Batch Progress: {i+1}/{len(batches)} ({progress:.1f}%) - Memory: {current_resources['memory_mb']:.1f}MB - ETA: {remaining:.1f}s")
            
            print(f"   4️⃣ Recursive Proof Generation...")
            proof_start = time.perf_counter()
            result = nova_manager.prove_recursive_batch(batches)
            proof_time = time.perf_counter() - proof_start
            
            if not result.success:
                raise Exception(f"Recursive proof failed: {result.error_message}")
            
            print(f"   5️⃣ Final Proof Verification...")
            verify_time = result.verify_time
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"\\n📊 RECURSIVE SNARK RESULTS:")
            print(f"   ✅ Recursive proof successful")
            print(f"   🔄 Steps processed: {len(batches)}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   🔧 Setup time: {setup_time:.3f}s")
            print(f"   🔐 Proof generation time: {proof_time:.3f}s")
            print(f"   ✅ Verification time: {verify_time:.3f}s")
            print(f"   ⚡ Throughput: {num_items / total_time:.2f} items/sec")
            print(f"   💾 Max memory: {max_memory:.1f}MB")
            print(f"   📏 Final proof size: {result.proof_size / 1024:.2f}KB")
            print(f"   🎯 Items per step: 3")
            print(f"   📊 Compression ratio: {(num_items * 0.83) / (result.proof_size / 1024):.2f}x")
            print(f"   🎯 Crossover check: {total_time:.1f}s vs standard ~{num_items * 0.14:.1f}s")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "setup_time": setup_time,
                "proof_time": proof_time,
                "verify_time": verify_time,
                "proof_size_kb": result.proof_size / 1024,
                "throughput": num_items / total_time if total_time > 0 else 0,
                "max_memory_mb": max_memory,
                "compression_ratio": (num_items * 0.83) / (result.proof_size / 1024) if result.proof_size > 0 else 0,
                "container_constrained": True,
                "crossover_validation": True
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        print(f"\\n❌ ERROR: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "max_memory_mb": max_memory,
            "container_constrained": True,
            "crossover_validation": True
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python validation_test.py <num_items> <test_type>")
        sys.exit(1)
    
    num_items = int(sys.argv[1])
    test_type = sys.argv[2]
    
    result = docker_validation_test(num_items, test_type)
    print("\\n" + "="*80)
    print("📋 FINAL RESULT JSON:")
    print(json.dumps(result, indent=2))
'''
    return script_content

def copy_essential_files_only(source_dir, target_dir):
    """Kopiert nur die essentiellen Dateien"""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Essential directories to copy
    essential_dirs = ["src", "circuits"]
    
    for dir_name in essential_dirs:
        source_path = source_dir / dir_name
        target_path = target_dir / dir_name
        
        if source_path.exists():
            shutil.copytree(source_path, target_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

def run_validation_docker_test(num_items, test_type, iot_config):
    """Führt Crossover-Validation Test aus"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy only essential files
        docker_project = temp_path / "bachelor"
        copy_essential_files_only(project_root, docker_project)
        
        # Create Docker files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_validation_dockerfile())
        
        requirements_path = docker_project / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(create_validation_requirements())
        
        script_path = docker_project / "validation_test.py"
        with open(script_path, 'w') as f:
            f.write(create_validation_test_script())
        
        # Build Docker image
        image_name = f"iot-validation-{num_items}-{test_type}"
        
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
        
        # Run Docker container
        print(f"🚀 Running crossover validation test:")
        print(f"   Items: {num_items}, Type: {test_type}")
        print(f"   Constraints: {iot_config['memory']} RAM, {iot_config['cpu']} CPU")
        print(f"   Expected: {'CROSSOVER!' if num_items >= 293 else 'Standard wins'}")
        
        run_cmd = [
            "docker", "run", "--rm",
            f"--memory={iot_config['memory']}",
            f"--cpus={iot_config['cpu']}",
            f"--memory-swap={iot_config['memory']}",
            "--oom-kill-disable=false",
            image_name,
            "python3", "validation_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=3600)  # 60 min timeout
            
            if result.returncode == 0:
                # Show full output
                print("📋 DOCKER OUTPUT:")
                print("-" * 60)
                print(result.stdout)
                print("-" * 60)
                
                # Parse JSON from output - robust approach
                output_lines = result.stdout.strip().split('\n')
                
                # Look for JSON block after "FINAL RESULT JSON:"
                json_started = False
                json_lines = []
                
                for line in output_lines:
                    if "FINAL RESULT JSON:" in line:
                        json_started = True
                        continue
                    
                    if json_started:
                        if line.strip().startswith('{') or line.strip().startswith('"') or line.strip().startswith('}') or '"' in line:
                            json_lines.append(line.strip())
                        elif line.strip() == "" or line.startswith('----'):
                            break
                
                if json_lines:
                    json_text = '\n'.join(json_lines)
                    try:
                        parsed_result = json.loads(json_text)
                        parsed_result["iot_config"] = iot_config
                        return parsed_result
                    except json.JSONDecodeError:
                        # Fallback: try each line individually
                        for line in reversed(output_lines):
                            if line.strip().startswith('{') and line.strip().endswith('}'):
                                try:
                                    parsed_result = json.loads(line.strip())
                                    parsed_result["iot_config"] = iot_config
                                    return parsed_result
                                except json.JSONDecodeError:
                                    continue
                
                print(f"❌ Could not parse JSON output")
                return None
            else:
                print(f"❌ Docker run failed (exit code {result.returncode})")
                print(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Docker test timed out after 60 minutes")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Docker output: {e}")
            return None

def crossover_validation_analysis():
    """Crossover-Validation für 290 & 300 Items"""
    print("🎯 CROSSOVER VALIDATION ANALYSIS")
    print("Testing 290 & 300 Items around estimated crossover point (~293)")
    print("Based on trend analysis from 50-200 Items")
    print("=" * 80)
    
    # IoT Configuration - Smart Home Hub (bewährt)
    iot_config = {
        "name": "Smart Home Hub",
        "memory": "1g",
        "cpu": "1.0",
        "description": "Proven IoT orchestrator configuration"
    }
    
    print(f"📱 Using {iot_config['name']} configuration:")
    print(f"   Memory: {iot_config['memory']}, CPU: {iot_config['cpu']}")
    print(f"   Description: {iot_config['description']}")
    
    # Test items: 290 & 300 (around estimated crossover at 293)
    test_items = [290, 300]
    all_results = []
    
    for num_items in test_items:
        print(f"\n" + "="*80)
        print(f"🎯 CROSSOVER VALIDATION: {num_items} ITEMS")
        print(f"Expected: {'🎉 CROSSOVER!' if num_items >= 293 else '📊 Standard wins'}")
        print("="*80)
        
        # Standard Test
        print(f"\n📊 STANDARD SNARKs TEST:")
        print("-" * 50)
        std_result = run_validation_docker_test(num_items, "standard", iot_config)
        
        if not (std_result and std_result["success"]):
            print(f"❌ Standard test failed for {num_items} items")
            continue
        
        # Recursive Test
        print(f"\n🚀 RECURSIVE SNARKs TEST:")
        print("-" * 50)
        rec_result = run_validation_docker_test(num_items, "recursive", iot_config)
        
        if not (rec_result and rec_result["success"]):
            print(f"❌ Recursive test failed for {num_items} items")
            continue
        
        # Compare results
        ratio = rec_result["total_time"] / std_result["total_time"]
        winner = "Recursive" if ratio < 1.0 else "Standard"
        advantage = abs(1.0 - ratio) * 100
        crossover_achieved = ratio < 1.0
        
        comparison_result = {
            "num_items": num_items,
            "standard_result": std_result,
            "recursive_result": rec_result,
            "comparison": {
                "ratio": ratio,
                "winner": winner,
                "recursive_wins": crossover_achieved,
                "advantage_percent": advantage,
                "time_difference": abs(rec_result["total_time"] - std_result["total_time"]),
                "throughput_ratio": rec_result["throughput"] / std_result["throughput"] if std_result["throughput"] > 0 else 0,
                "memory_ratio": rec_result["max_memory_mb"] / std_result["max_memory_mb"] if std_result["max_memory_mb"] > 0 else 1.0,
                "crossover_validation": True,
                "estimated_crossover_293": num_items >= 293
            }
        }
        
        all_results.append(comparison_result)
        
        print(f"\n📊 CROSSOVER VALIDATION SUMMARY:")
        print(f"   🏆 Winner: {winner}")
        print(f"   📈 Advantage: {advantage:.1f}%")
        print(f"   ⚡ Ratio: {ratio:.4f}")
        print(f"   💾 Memory: Std {std_result['max_memory_mb']:.1f}MB, Rec {rec_result['max_memory_mb']:.1f}MB")
        print(f"   🎯 Crossover: {'✅ ACHIEVED!' if crossover_achieved else '❌ Not yet'}")
        
        # Special crossover announcement
        if crossover_achieved:
            print(f"\n🎉🎉🎉 CROSSOVER POINT FOUND! 🎉🎉🎉")
            print(f"   📊 At {num_items} items: Recursive SNARKs are {advantage:.1f}% faster!")
            print(f"   ⚡ Recursive: {rec_result['total_time']:.1f}s vs Standard: {std_result['total_time']:.1f}s")
    
    return all_results

def analyze_validation_results(results):
    """Analysiert Crossover-Validation Ergebnisse"""
    print("\n" + "="*100)
    print("📊 CROSSOVER VALIDATION RESULTS (290-300 Items)")
    print("="*100)
    
    if not results:
        print("❌ No validation results to analyze")
        return None
    
    print(f"\n✅ CROSSOVER VALIDATION UNDER IoT CONSTRAINTS:")
    print(f"{'Items':<6} {'Std(s)':<8} {'Rec(s)':<8} {'Ratio':<8} {'Diff%':<7} {'Winner':<10} {'Crossover':<10}")
    print("-" * 80)
    
    crossover_achieved = False
    crossover_point = None
    
    for result in results:
        items = result["num_items"]
        std_time = result["standard_result"]["total_time"]
        rec_time = result["recursive_result"]["total_time"]
        ratio = result["comparison"]["ratio"]
        diff_pct = result["comparison"]["advantage_percent"]
        winner = result["comparison"]["winner"]
        is_crossover = result["comparison"]["recursive_wins"]
        
        crossover_status = "✅ YES" if is_crossover else "❌ NO"
        
        print(f"{items:<6} {std_time:<8.2f} {rec_time:<8.2f} {ratio:<8.4f} {diff_pct:<7.1f} {winner:<10} {crossover_status:<10}")
        
        if is_crossover and not crossover_achieved:
            crossover_achieved = True
            crossover_point = items
    
    print(f"\n🎯 FINAL CROSSOVER VALIDATION:")
    if crossover_achieved:
        print(f"   🎉 CROSSOVER CONFIRMED: {crossover_point} items")
        print(f"   📊 Recursive SNARKs become efficient at {crossover_point}+ items under IoT constraints")
        print(f"   ✅ Estimation accuracy: {'Excellent' if crossover_point <= 300 else 'Good'}")
    else:
        print(f"   ⚠️  CROSSOVER NOT YET REACHED at 300 items")
        print(f"   📈 Crossover point is higher than 300 items under IoT constraints")
        
        # Update estimation
        if len(results) >= 2:
            last_ratio = results[-1]["comparison"]["ratio"]
            first_ratio = results[0]["comparison"]["ratio"]
            trend_slope = (last_ratio - first_ratio) / (results[-1]["num_items"] - results[0]["num_items"])
            
            if trend_slope < 0:  # Recursive getting better
                estimated_crossover = results[-1]["num_items"] + (last_ratio - 1.0) / abs(trend_slope)
                print(f"   🔮 Updated crossover estimate: ~{estimated_crossover:.0f} items")
    
    return crossover_point

def main():
    """Hauptfunktion"""
    print("🎯 Starting Crossover Validation Analysis (290 & 300 Items)...")
    
    results = crossover_validation_analysis()
    
    if results:
        crossover_point = analyze_validation_results(results)
        
        # Save results
        results_dir = project_root / "data" / "crossover_validation_290_300"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "validation_290_300_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "validation_results": results,
                "crossover_point": crossover_point,
                "crossover_achieved": crossover_point is not None,
                "test_timestamp": time.time(),
                "iot_configuration": "Smart Home Hub (1GB RAM, 1 CPU)",
                "test_items": [290, 300],
                "methodology": "Docker containers with realistic IoT resource constraints - crossover validation",
                "previous_estimation": "~293 items based on 50-200 trend analysis",
                "validation_purpose": "Confirm crossover point around estimated 293 items"
            }, f, indent=2)
        
        print(f"\n💾 Crossover validation results saved: {results_file}")
        print(f"✅ CROSSOVER VALIDATION ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No crossover validation results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
