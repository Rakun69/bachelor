#!/usr/bin/env python3
"""
🔧 FAIR COMPARISON DEBUG TEST
Debuggt warum Standard SNARKs bei 270 Items abbrechen
Testet mit gleicher Item-Anzahl für fairen Vergleich
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

def create_debug_dockerfile():
    """Dockerfile für Debug-Tests mit mehr Ressourcen"""
    dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    build-essential \\
    libssl-dev \\
    pkg-config \\
    htop \\
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

def create_debug_requirements():
    """Requirements für Debug Tests"""
    requirements = '''psutil>=5.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
'''
    return requirements

def create_debug_test_script():
    """Debug Test Script - findet warum Standard bei 270 abbricht"""
    script_content = '''#!/usr/bin/env python3
import sys
import time
import json
import psutil
import traceback
from pathlib import Path

# Add project paths
project_root = Path("/app")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

def debug_standard_snark_limit(num_items, test_type):
    """Debug warum Standard SNARKs bei 270 Items abbrechen"""
    print(f"🔧 DEBUG TEST: {num_items} items ({test_type})")
    print(f"📱 Debugging Standard SNARK limit issue")
    
    start_time = time.perf_counter()
    
    try:
        sensors = SmartHomeSensors()
        readings = sensors.generate_readings(duration_hours=1, time_step_seconds=60)
        temp_readings = [r for r in readings if r.sensor_type == "temperature"][:num_items]
        
        print(f"📊 Generated {len(temp_readings)} temperature readings")
        
        if test_type == "standard":
            print(f"🔧 Standard SNARK Pipeline DEBUG for {num_items} items:")
            
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
            failed = 0
            individual_proof_sizes = []
            
            print(f"   3️⃣ Processing {num_items} sensor readings with DEBUG:")
            
            for i, reading in enumerate(temp_readings):
                try:
                    # Monitor resources before each proof
                    memory_info = psutil.virtual_memory()
                    memory_mb = memory_info.used / 1024 / 1024
                    memory_available_mb = memory_info.available / 1024 / 1024
                    
                    if i % 50 == 0:
                        print(f"   📊 Item {i}: Memory {memory_mb:.1f}MB used, {memory_available_mb:.1f}MB available")
                    
                    # Check if we're running out of memory
                    if memory_available_mb < 100:  # Less than 100MB available
                        print(f"   ⚠️  LOW MEMORY WARNING at item {i}: Only {memory_available_mb:.1f}MB available")
                    
                    secret_value = max(10, min(50, int(reading.value)))
                    inputs = ["10", "50", str(secret_value)]
                    
                    result = manager.generate_proof("filter_range", inputs)
                    
                    if result.success:
                        successful += 1
                        individual_proof_sizes.append(result.metrics.proof_size)
                        
                        if (i + 1) % 100 == 0:
                            print(f"   ✅ Successfully completed {i+1} proofs")
                    else:
                        failed += 1
                        print(f"   ❌ Proof {i+1} failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                        
                        # Stop on first failure to debug
                        if failed >= 5:
                            print(f"   🛑 Stopping after {failed} failures for debugging")
                            break
                
                except Exception as e:
                    failed += 1
                    print(f"   💥 EXCEPTION at item {i+1}: {str(e)}")
                    print(f"   📊 Memory at exception: {memory_mb:.1f}MB used")
                    traceback.print_exc()
                    
                    # Stop on exception to debug
                    break
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Calculate final metrics
            if individual_proof_sizes:
                total_proof_size_kb = sum(individual_proof_sizes) / 1024
                avg_proof_size_kb = sum(individual_proof_sizes) / len(individual_proof_sizes) / 1024
            else:
                total_proof_size_kb = 0
                avg_proof_size_kb = 0
            
            print(f"\\n📊 DEBUG STANDARD SNARK RESULTS:")
            print(f"   ✅ Successful proofs: {successful}/{num_items}")
            print(f"   ❌ Failed proofs: {failed}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   📏 Total proof size: {total_proof_size_kb:.2f}KB")
            print(f"   📏 Average proof size: {avg_proof_size_kb:.3f}KB")
            
            # Extrapolate what full run would look like
            if successful > 0:
                time_per_proof = total_time / successful
                estimated_full_time = time_per_proof * num_items
                estimated_full_size = avg_proof_size_kb * num_items
                
                print(f"\\n🔮 EXTRAPOLATED FULL RESULTS:")
                print(f"   ⏱️  Estimated full time: {estimated_full_time:.3f}s")
                print(f"   📏 Estimated full size: {estimated_full_size:.2f}KB")
                print(f"   ⚡ Estimated throughput: {num_items / estimated_full_time:.2f} proofs/sec")
            
            return {
                "success": successful > 0,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "failed_proofs": failed,
                "total_time": total_time,
                "compile_time": compile_time,
                "setup_time": setup_time,
                "total_proof_size_kb": total_proof_size_kb,
                "avg_proof_size_kb": avg_proof_size_kb,
                "throughput": successful / total_time if total_time > 0 else 0,
                "debug_mode": True,
                "completion_rate": successful / num_items if num_items > 0 else 0,
                "estimated_full_time": time_per_proof * num_items if successful > 0 else 0,
                "estimated_full_size_kb": avg_proof_size_kb * num_items if avg_proof_size_kb > 0 else 0
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
            
            print(f"   3️⃣ Recursive Proof Generation...")
            proof_start = time.perf_counter()
            result = nova_manager.prove_recursive_batch(batches)
            proof_time = time.perf_counter() - proof_start
            
            if not result.success:
                raise Exception(f"Recursive proof failed: {result.error_message}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"\\n📊 RECURSIVE SNARK RESULTS:")
            print(f"   ✅ Recursive proof successful")
            print(f"   🔄 Steps processed: {len(batches)}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   🔧 Setup time: {setup_time:.3f}s")
            print(f"   🔐 Proof generation time: {proof_time:.3f}s")
            print(f"   ✅ Verification time: {result.verify_time:.3f}s")
            print(f"   ⚡ Throughput: {num_items / total_time:.2f} items/sec")
            print(f"   📏 Final proof size: {result.proof_size / 1024:.2f}KB")
            print(f"   🎯 Items per step: 3")
            
            return {
                "success": True,
                "type": "recursive",
                "num_items": num_items,
                "steps": len(batches),
                "total_time": total_time,
                "setup_time": setup_time,
                "proof_time": proof_time,
                "verify_time": result.verify_time,
                "proof_size_kb": result.proof_size / 1024,
                "throughput": num_items / total_time if total_time > 0 else 0,
                "debug_mode": True
            }
            
    except Exception as e:
        end_time = time.perf_counter()
        print(f"\\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "debug_mode": True
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python debug_test.py <num_items> <test_type>")
        sys.exit(1)
    
    num_items = int(sys.argv[1])
    test_type = sys.argv[2]
    
    result = debug_standard_snark_limit(num_items, test_type)
    print("\\n" + "="*80)
    print("📋 FINAL DEBUG RESULT JSON:")
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

def run_debug_docker_test(num_items, test_type):
    """Führt Debug Test mit mehr Ressourcen aus"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy only essential files
        docker_project = temp_path / "bachelor"
        copy_essential_files_only(project_root, docker_project)
        
        # Create Docker files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_debug_dockerfile())
        
        requirements_path = docker_project / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(create_debug_requirements())
        
        script_path = docker_project / "debug_test.py"
        with open(script_path, 'w') as f:
            f.write(create_debug_test_script())
        
        # Build Docker image
        image_name = f"iot-debug-{num_items}-{test_type}"
        
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
        
        # Run Docker container with MORE resources for debugging
        print(f"🚀 Running debug test with increased resources:")
        print(f"   Items: {num_items}, Type: {test_type}")
        print(f"   Resources: 2GB RAM, 2.0 CPU (increased for debugging)")
        
        run_cmd = [
            "docker", "run", "--rm",
            "--memory=2g",  # Increased memory
            "--cpus=2.0",   # Increased CPU
            "--memory-swap=2g",
            "--oom-kill-disable=false",
            image_name,
            "python3", "debug_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=3600)  # 60 min timeout
            
            if result.returncode == 0:
                # Show full output
                print("📋 DOCKER DEBUG OUTPUT:")
                print("-" * 60)
                print(result.stdout)
                print("-" * 60)
                
                # Parse JSON from output
                output_lines = result.stdout.strip().split('\n')
                
                # Look for JSON block after "FINAL DEBUG RESULT JSON:"
                json_started = False
                json_lines = []
                
                for line in output_lines:
                    if "FINAL DEBUG RESULT JSON:" in line:
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
                        return parsed_result
                    except json.JSONDecodeError:
                        # Fallback: try each line individually
                        for line in reversed(output_lines):
                            if line.strip().startswith('{') and line.strip().endswith('}'):
                                try:
                                    parsed_result = json.loads(line.strip())
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

def main():
    """Debug warum Standard SNARKs bei 270 abbrechen"""
    print("🔧 DEBUG: Why do Standard SNARKs stop at 270 items?")
    print("Testing with increased Docker resources to isolate the issue")
    print("=" * 80)
    
    # Test both 270 (where it stops) and 450 (target)
    test_cases = [
        (270, "standard"),  # Where it currently stops
        (450, "standard"),  # Where we want it to go
        (450, "recursive")  # For comparison
    ]
    
    results = []
    
    for num_items, test_type in test_cases:
        print(f"\n" + "="*80)
        print(f"🔧 DEBUG TEST: {num_items} ITEMS ({test_type.upper()})")
        print("="*80)
        
        result = run_debug_docker_test(num_items, test_type)
        
        if result:
            results.append(result)
            
            if result.get("success"):
                print(f"\n📊 DEBUG SUMMARY:")
                print(f"   Items: {num_items}")
                print(f"   Type: {test_type}")
                print(f"   Success: {result.get('success')}")
                
                if test_type == "standard":
                    successful = result.get("successful_proofs", 0)
                    failed = result.get("failed_proofs", 0)
                    completion_rate = result.get("completion_rate", 0)
                    
                    print(f"   Successful: {successful}/{num_items} ({completion_rate*100:.1f}%)")
                    print(f"   Failed: {failed}")
                    
                    if "estimated_full_time" in result:
                        print(f"   Estimated full time: {result['estimated_full_time']:.1f}s")
                        print(f"   Estimated full size: {result.get('estimated_full_size_kb', 0):.1f}KB")
                
                else:  # recursive
                    print(f"   Steps: {result.get('steps', 0)}")
                    print(f"   Time: {result.get('total_time', 0):.1f}s")
                    print(f"   Proof size: {result.get('proof_size_kb', 0):.1f}KB")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ No result obtained")
    
    # Save debug results
    if results:
        results_dir = project_root / "data" / "debug_fair_comparison"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "debug_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "debug_results": results,
                "test_timestamp": time.time(),
                "purpose": "Debug why Standard SNARKs stop at 270 items",
                "increased_resources": "2GB RAM, 2.0 CPU for debugging",
                "findings": "Investigate memory limits and container constraints"
            }, f, indent=2)
        
        print(f"\n💾 Debug results saved: {results_file}")
    
    print(f"\n✅ DEBUG ANALYSIS COMPLETE!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
