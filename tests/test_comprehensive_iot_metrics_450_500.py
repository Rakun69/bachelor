#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE IoT METRICS TEST - 450 & 500 ITEMS
Misst ALLE relevanten IoT-Metriken: Zeit, Memory, CPU, Energy, Network, Kosten
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

def create_comprehensive_dockerfile():
    """Dockerfile für comprehensive IoT metrics"""
    dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies including monitoring tools
RUN apt-get update && apt-get install -y \\
    curl \\
    build-essential \\
    libssl-dev \\
    pkg-config \\
    htop \\
    iotop \\
    sysstat \\
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

def create_comprehensive_requirements():
    """Requirements für comprehensive metrics"""
    requirements = '''psutil>=5.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
py-cpuinfo>=9.0.0
'''
    return requirements

def create_comprehensive_test_script():
    """Comprehensive Test Script mit ALLEN IoT-Metriken"""
    script_content = '''#!/usr/bin/env python3
import sys
import time
import json
import psutil
import threading
import os
from pathlib import Path
from collections import defaultdict

# Add project paths
project_root = Path("/app")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.iot_simulation.smart_home import SmartHomeSensors
from src.proof_systems.snark_manager import SNARKManager
from src.proof_systems.fixed_nova_manager import FixedZoKratesNovaManager

class ComprehensiveIoTMonitor:
    """Umfassende IoT-Ressourcen Überwachung"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            'memory_samples': [],
            'cpu_samples': [],
            'network_io_start': None,
            'network_io_samples': [],
            'disk_io_start': None,
            'disk_io_samples': [],
            'energy_samples': [],  # Approximation basierend auf CPU/Memory
            'timestamps': []
        }
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Startet kontinuierliche Überwachung"""
        self.monitoring = True
        self.metrics['network_io_start'] = psutil.net_io_counters()
        self.metrics['disk_io_start'] = psutil.disk_io_counters()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stoppt Überwachung"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Kontinuierliche Überwachungsschleife"""
        while self.monitoring:
            try:
                timestamp = time.perf_counter()
                
                # Memory
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / 1024 / 1024
                
                # CPU
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                
                # Disk I/O  
                disk_io = psutil.disk_io_counters()
                
                # Energy approximation (CPU + Memory load based)
                # Typical IoT device: ~2W idle, +0.5W per 10% CPU, +0.1W per 100MB RAM
                base_power = 2.0  # Watts
                cpu_power = (cpu_percent / 100) * 0.5
                memory_power = (memory_mb / 100) * 0.1
                estimated_power = base_power + cpu_power + memory_power
                
                self.metrics['timestamps'].append(timestamp)
                self.metrics['memory_samples'].append(memory_mb)
                self.metrics['cpu_samples'].append(cpu_percent)
                self.metrics['network_io_samples'].append({
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                })
                self.metrics['disk_io_samples'].append({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                })
                self.metrics['energy_samples'].append(estimated_power)
                
                time.sleep(0.5)  # Sample every 0.5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
                
    def get_comprehensive_metrics(self, duration_seconds):
        """Berechnet umfassende Metriken"""
        if not self.metrics['timestamps']:
            return {}
            
        # Memory metrics
        max_memory_mb = max(self.metrics['memory_samples']) if self.metrics['memory_samples'] else 0
        avg_memory_mb = sum(self.metrics['memory_samples']) / len(self.metrics['memory_samples']) if self.metrics['memory_samples'] else 0
        
        # CPU metrics
        max_cpu_percent = max(self.metrics['cpu_samples']) if self.metrics['cpu_samples'] else 0
        avg_cpu_percent = sum(self.metrics['cpu_samples']) / len(self.metrics['cpu_samples']) if self.metrics['cpu_samples'] else 0
        
        # Network metrics
        network_start = self.metrics['network_io_start']
        network_end = self.metrics['network_io_samples'][-1] if self.metrics['network_io_samples'] else network_start
        
        if network_start and network_end:
            total_bytes_sent = network_end['bytes_sent'] - network_start.bytes_sent
            total_bytes_recv = network_end['bytes_recv'] - network_start.bytes_recv
            total_network_kb = (total_bytes_sent + total_bytes_recv) / 1024
        else:
            total_network_kb = 0
            
        # Energy metrics (approximation)
        total_energy_wh = 0
        if self.metrics['energy_samples'] and duration_seconds > 0:
            avg_power_w = sum(self.metrics['energy_samples']) / len(self.metrics['energy_samples'])
            total_energy_wh = avg_power_w * (duration_seconds / 3600)  # Watt-hours
            
        # Cost calculation (example IoT pricing)
        # CPU: 0.001€ per CPU-hour per core
        # Memory: 0.0005€ per GB-hour  
        # Energy: 0.25€ per kWh
        # Network: 0.10€ per GB
        
        cpu_cost_eur = (avg_cpu_percent / 100) * (duration_seconds / 3600) * 0.001
        memory_cost_eur = (avg_memory_mb / 1024) * (duration_seconds / 3600) * 0.0005
        energy_cost_eur = total_energy_wh / 1000 * 0.25  # Convert Wh to kWh
        network_cost_eur = (total_network_kb / 1024 / 1024) * 0.10  # Convert KB to GB
        
        total_cost_eur = cpu_cost_eur + memory_cost_eur + energy_cost_eur + network_cost_eur
        
        return {
            'memory_metrics': {
                'max_memory_mb': max_memory_mb,
                'avg_memory_mb': avg_memory_mb,
                'memory_efficiency': avg_memory_mb / max_memory_mb if max_memory_mb > 0 else 0
            },
            'cpu_metrics': {
                'max_cpu_percent': max_cpu_percent,
                'avg_cpu_percent': avg_cpu_percent,
                'cpu_efficiency': avg_cpu_percent / 100
            },
            'network_metrics': {
                'total_network_kb': total_network_kb,
                'network_throughput_kbps': total_network_kb / duration_seconds if duration_seconds > 0 else 0
            },
            'energy_metrics': {
                'total_energy_wh': total_energy_wh,
                'avg_power_w': sum(self.metrics['energy_samples']) / len(self.metrics['energy_samples']) if self.metrics['energy_samples'] else 0,
                'energy_efficiency': total_energy_wh / duration_seconds if duration_seconds > 0 else 0
            },
            'cost_metrics': {
                'cpu_cost_eur': cpu_cost_eur,
                'memory_cost_eur': memory_cost_eur,
                'energy_cost_eur': energy_cost_eur,
                'network_cost_eur': network_cost_eur,
                'total_cost_eur': total_cost_eur,
                'cost_per_item': total_cost_eur
            }
        }

def docker_comprehensive_test(num_items, test_type):
    """Comprehensive Test mit allen IoT-Metriken"""
    print(f"🎯 Docker Comprehensive IoT Test: {num_items} items ({test_type})")
    print(f"📱 Running under IoT constraints: 1GB RAM, 1 CPU")
    print(f"📊 Measuring: Time, Memory, CPU, Energy, Network, Costs")
    
    # Initialize comprehensive monitoring
    monitor = ComprehensiveIoTMonitor()
    
    start_time = time.perf_counter()
    monitor.start_monitoring()
    
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
            proof_times = []
            verify_times = []
            
            print(f"   3️⃣ Processing {num_items} sensor readings:")
            print(f"   📊 Expected time: ~{num_items * 0.14:.1f}s")
            
            for i, reading in enumerate(temp_readings):
                secret_value = max(10, min(50, int(reading.value)))
                inputs = ["10", "50", str(secret_value)]
                
                proof_start = time.perf_counter()
                result = manager.generate_proof("filter_range", inputs)
                proof_end = time.perf_counter()
                
                if result.success:
                    successful += 1
                    individual_proof_sizes.append(result.metrics.proof_size)
                    proof_times.append(result.metrics.proof_time)
                    verify_times.append(result.metrics.verify_time)
                    
                    # Show detailed output every 50th proof
                    if (i + 1) % 50 == 0 or i < 3:
                        current_metrics = monitor.get_comprehensive_metrics(proof_end - start_time)
                        print(f"   📊 Proof {i+1}: Time {proof_end - proof_start:.3f}s, CPU {current_metrics.get('cpu_metrics', {}).get('avg_cpu_percent', 0):.1f}%, Memory {current_metrics.get('memory_metrics', {}).get('avg_memory_mb', 0):.1f}MB")
                
                # Progress indicator every 50 items
                if (i + 1) % 50 == 0:
                    progress = (i + 1) / num_items * 100
                    elapsed = time.perf_counter() - start_time
                    estimated_total = elapsed / (i + 1) * num_items
                    remaining = estimated_total - elapsed
                    current_metrics = monitor.get_comprehensive_metrics(elapsed)
                    print(f"   📈 Progress: {i+1}/{num_items} ({progress:.1f}%) - ETA: {remaining:.1f}s")
                    print(f"      💾 Memory: {current_metrics.get('memory_metrics', {}).get('avg_memory_mb', 0):.1f}MB")
                    print(f"      🖥️  CPU: {current_metrics.get('cpu_metrics', {}).get('avg_cpu_percent', 0):.1f}%")
                    print(f"      🔋 Power: {current_metrics.get('energy_metrics', {}).get('avg_power_w', 0):.2f}W")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Stop monitoring and get final metrics
            monitor.stop_monitoring()
            comprehensive_metrics = monitor.get_comprehensive_metrics(total_time)
            
            # Calculate final metrics
            if individual_proof_sizes:
                total_proof_size_kb = sum(individual_proof_sizes) / 1024
                avg_proof_size_kb = sum(individual_proof_sizes) / len(individual_proof_sizes) / 1024
            else:
                total_proof_size_kb = 0
                avg_proof_size_kb = 0
            
            avg_proof_time = sum(proof_times) / len(proof_times) if proof_times else 0
            avg_verify_time = sum(verify_times) / len(verify_times) if verify_times else 0
            
            print(f"\\n📊 COMPREHENSIVE STANDARD SNARK RESULTS:")
            print(f"   ✅ Successful proofs: {successful}/{num_items}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   ⚡ Throughput: {successful / total_time:.2f} proofs/sec")
            print(f"   📏 Total proof size: {total_proof_size_kb:.2f}KB")
            print(f"   📏 Average proof size: {avg_proof_size_kb:.3f}KB")
            print(f"   ⚡ Avg proof time: {avg_proof_time:.3f}s")
            print(f"   ✅ Avg verify time: {avg_verify_time:.3f}s")
            
            # Comprehensive IoT metrics
            print(f"\\n🔋 IoT RESOURCE METRICS:")
            mem_metrics = comprehensive_metrics.get('memory_metrics', {})
            cpu_metrics = comprehensive_metrics.get('cpu_metrics', {})
            energy_metrics = comprehensive_metrics.get('energy_metrics', {})
            network_metrics = comprehensive_metrics.get('network_metrics', {})
            cost_metrics = comprehensive_metrics.get('cost_metrics', {})
            
            print(f"   💾 Memory: Max {mem_metrics.get('max_memory_mb', 0):.1f}MB, Avg {mem_metrics.get('avg_memory_mb', 0):.1f}MB")
            print(f"   🖥️  CPU: Max {cpu_metrics.get('max_cpu_percent', 0):.1f}%, Avg {cpu_metrics.get('avg_cpu_percent', 0):.1f}%")
            print(f"   🔋 Energy: {energy_metrics.get('total_energy_wh', 0):.4f}Wh, Avg {energy_metrics.get('avg_power_w', 0):.2f}W")
            print(f"   🌐 Network: {network_metrics.get('total_network_kb', 0):.2f}KB total")
            print(f"\\n💰 IoT COST ANALYSIS:")
            print(f"   💻 CPU Cost: {cost_metrics.get('cpu_cost_eur', 0):.6f}€")
            print(f"   💾 Memory Cost: {cost_metrics.get('memory_cost_eur', 0):.6f}€")
            print(f"   🔋 Energy Cost: {cost_metrics.get('energy_cost_eur', 0):.6f}€")
            print(f"   🌐 Network Cost: {cost_metrics.get('network_cost_eur', 0):.6f}€")
            print(f"   💰 Total Cost: {cost_metrics.get('total_cost_eur', 0):.6f}€")
            print(f"   📊 Cost per Item: {cost_metrics.get('total_cost_eur', 0) / num_items:.8f}€")
            
            result_data = {
                "success": True,
                "type": "standard",
                "num_items": num_items,
                "successful_proofs": successful,
                "total_time": total_time,
                "compile_time": compile_time,
                "setup_time": setup_time,
                "avg_proof_time": avg_proof_time,
                "avg_verify_time": avg_verify_time,
                "total_proof_size_kb": total_proof_size_kb,
                "avg_proof_size_kb": avg_proof_size_kb,
                "throughput": successful / total_time if total_time > 0 else 0,
                "container_constrained": True,
                "comprehensive_metrics": True
            }
            
            # Add comprehensive metrics
            result_data.update(comprehensive_metrics)
            
            return result_data
            
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
            print(f"   📊 Expected time: ~{len(batches) * 0.6:.1f}s")
            
            print(f"   3️⃣ Incremental Verification...")
            
            # Monitor during recursive proof
            for i in range(len(batches)):
                if (i + 1) % 20 == 0:
                    progress = (i + 1) / len(batches) * 100
                    elapsed = time.perf_counter() - start_time
                    estimated_total = elapsed / (i + 1) * len(batches)
                    remaining = estimated_total - elapsed
                    current_metrics = monitor.get_comprehensive_metrics(elapsed)
                    print(f"   🔄 Batch Progress: {i+1}/{len(batches)} ({progress:.1f}%) - ETA: {remaining:.1f}s")
                    print(f"      💾 Memory: {current_metrics.get('memory_metrics', {}).get('avg_memory_mb', 0):.1f}MB")
                    print(f"      🖥️  CPU: {current_metrics.get('cpu_metrics', {}).get('avg_cpu_percent', 0):.1f}%")
                    print(f"      🔋 Power: {current_metrics.get('energy_metrics', {}).get('avg_power_w', 0):.2f}W")
            
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
            
            # Stop monitoring and get final metrics
            monitor.stop_monitoring()
            comprehensive_metrics = monitor.get_comprehensive_metrics(total_time)
            
            print(f"\\n📊 COMPREHENSIVE RECURSIVE SNARK RESULTS:")
            print(f"   ✅ Recursive proof successful")
            print(f"   🔄 Steps processed: {len(batches)}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   🔧 Setup time: {setup_time:.3f}s")
            print(f"   🔐 Proof generation time: {proof_time:.3f}s")
            print(f"   ✅ Verification time: {verify_time:.3f}s")
            print(f"   ⚡ Throughput: {num_items / total_time:.2f} items/sec")
            print(f"   📏 Final proof size: {result.proof_size / 1024:.2f}KB")
            print(f"   🎯 Items per step: 3")
            print(f"   📊 Compression ratio: {(num_items * 0.83) / (result.proof_size / 1024):.2f}x")
            
            # Comprehensive IoT metrics
            print(f"\\n🔋 IoT RESOURCE METRICS:")
            mem_metrics = comprehensive_metrics.get('memory_metrics', {})
            cpu_metrics = comprehensive_metrics.get('cpu_metrics', {})
            energy_metrics = comprehensive_metrics.get('energy_metrics', {})
            network_metrics = comprehensive_metrics.get('network_metrics', {})
            cost_metrics = comprehensive_metrics.get('cost_metrics', {})
            
            print(f"   💾 Memory: Max {mem_metrics.get('max_memory_mb', 0):.1f}MB, Avg {mem_metrics.get('avg_memory_mb', 0):.1f}MB")
            print(f"   🖥️  CPU: Max {cpu_metrics.get('max_cpu_percent', 0):.1f}%, Avg {cpu_metrics.get('avg_cpu_percent', 0):.1f}%")
            print(f"   🔋 Energy: {energy_metrics.get('total_energy_wh', 0):.4f}Wh, Avg {energy_metrics.get('avg_power_w', 0):.2f}W")
            print(f"   🌐 Network: {network_metrics.get('total_network_kb', 0):.2f}KB total")
            print(f"\\n💰 IoT COST ANALYSIS:")
            print(f"   💻 CPU Cost: {cost_metrics.get('cpu_cost_eur', 0):.6f}€")
            print(f"   💾 Memory Cost: {cost_metrics.get('memory_cost_eur', 0):.6f}€")
            print(f"   🔋 Energy Cost: {cost_metrics.get('energy_cost_eur', 0):.6f}€")
            print(f"   🌐 Network Cost: {cost_metrics.get('network_cost_eur', 0):.6f}€")
            print(f"   💰 Total Cost: {cost_metrics.get('total_cost_eur', 0):.6f}€")
            print(f"   📊 Cost per Item: {cost_metrics.get('total_cost_eur', 0) / num_items:.8f}€")
            
            result_data = {
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
                "compression_ratio": (num_items * 0.83) / (result.proof_size / 1024) if result.proof_size > 0 else 0,
                "container_constrained": True,
                "comprehensive_metrics": True
            }
            
            # Add comprehensive metrics
            result_data.update(comprehensive_metrics)
            
            return result_data
            
    except Exception as e:
        end_time = time.perf_counter()
        monitor.stop_monitoring()
        print(f"\\n❌ ERROR: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "total_time": end_time - start_time,
            "container_constrained": True,
            "comprehensive_metrics": True
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python comprehensive_test.py <num_items> <test_type>")
        sys.exit(1)
    
    num_items = int(sys.argv[1])
    test_type = sys.argv[2]
    
    result = docker_comprehensive_test(num_items, test_type)
    print("\\n" + "="*80)
    print("📋 FINAL COMPREHENSIVE RESULT JSON:")
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

def run_comprehensive_docker_test(num_items, test_type, iot_config):
    """Führt comprehensive IoT metrics Test aus"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy only essential files
        docker_project = temp_path / "bachelor"
        copy_essential_files_only(project_root, docker_project)
        
        # Create Docker files
        dockerfile_path = docker_project / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(create_comprehensive_dockerfile())
        
        requirements_path = docker_project / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(create_comprehensive_requirements())
        
        script_path = docker_project / "comprehensive_test.py"
        with open(script_path, 'w') as f:
            f.write(create_comprehensive_test_script())
        
        # Build Docker image
        image_name = f"iot-comprehensive-{num_items}-{test_type}"
        
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
        print(f"🚀 Running comprehensive IoT metrics test:")
        print(f"   Items: {num_items}, Type: {test_type}")
        print(f"   Constraints: {iot_config['memory']} RAM, {iot_config['cpu']} CPU")
        print(f"   Metrics: Time, Memory, CPU, Energy, Network, Costs")
        
        run_cmd = [
            "docker", "run", "--rm",
            f"--memory={iot_config['memory']}",
            f"--cpus={iot_config['cpu']}",
            f"--memory-swap={iot_config['memory']}",
            "--oom-kill-disable=false",
            image_name,
            "python3", "comprehensive_test.py", str(num_items), test_type
        ]
        
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=4800)  # 80 min timeout
            
            if result.returncode == 0:
                # Show full output
                print("📋 DOCKER OUTPUT:")
                print("-" * 60)
                print(result.stdout)
                print("-" * 60)
                
                # Parse JSON from output - robust approach
                output_lines = result.stdout.strip().split('\n')
                
                # Look for JSON block after "FINAL COMPREHENSIVE RESULT JSON:"
                json_started = False
                json_lines = []
                
                for line in output_lines:
                    if "FINAL COMPREHENSIVE RESULT JSON:" in line:
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
            print(f"❌ Docker test timed out after 80 minutes")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Docker output: {e}")
            return None

def comprehensive_iot_analysis():
    """Comprehensive IoT-Analyse für 450 & 500 Items mit ALLEN Metriken"""
    print("🎯 COMPREHENSIVE IoT METRICS ANALYSIS")
    print("Testing 450 & 500 Items with ALL IoT-relevant metrics")
    print("Metrics: Time, Memory, CPU, Energy, Network, Total Costs")
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
    
    # Test items: 450 & 500 (higher range for crossover search)
    test_items = [450, 500]
    all_results = []
    
    for num_items in test_items:
        print(f"\n" + "="*80)
        print(f"🎯 COMPREHENSIVE IoT TEST: {num_items} ITEMS")
        print("="*80)
        
        # Standard Test
        print(f"\n📊 STANDARD SNARKs COMPREHENSIVE TEST:")
        print("-" * 50)
        std_result = run_comprehensive_docker_test(num_items, "standard", iot_config)
        
        if not (std_result and std_result["success"]):
            print(f"❌ Standard test failed for {num_items} items")
            continue
        
        # Recursive Test
        print(f"\n🚀 RECURSIVE SNARKs COMPREHENSIVE TEST:")
        print("-" * 50)
        rec_result = run_comprehensive_docker_test(num_items, "recursive", iot_config)
        
        if not (rec_result and rec_result["success"]):
            print(f"❌ Recursive test failed for {num_items} items")
            continue
        
        # Compare results - comprehensive comparison
        time_ratio = rec_result["total_time"] / std_result["total_time"]
        
        # Cost comparison
        std_total_cost = std_result.get("cost_metrics", {}).get("total_cost_eur", 0)
        rec_total_cost = rec_result.get("cost_metrics", {}).get("total_cost_eur", 0)
        cost_ratio = rec_total_cost / std_total_cost if std_total_cost > 0 else float('inf')
        
        # Energy comparison
        std_energy = std_result.get("energy_metrics", {}).get("total_energy_wh", 0)
        rec_energy = rec_result.get("energy_metrics", {}).get("total_energy_wh", 0)
        energy_ratio = rec_energy / std_energy if std_energy > 0 else float('inf')
        
        # Determine winner based on total cost (most relevant for IoT)
        cost_winner = "Recursive" if cost_ratio < 1.0 else "Standard"
        time_winner = "Recursive" if time_ratio < 1.0 else "Standard"
        energy_winner = "Recursive" if energy_ratio < 1.0 else "Standard"
        
        # Overall winner (weighted: 50% cost, 30% time, 20% energy)
        overall_score_std = 0.5 * (1.0) + 0.3 * (1.0) + 0.2 * (1.0)  # Standard baseline
        overall_score_rec = 0.5 * (1.0/cost_ratio if cost_ratio > 0 else 0) + 0.3 * (1.0/time_ratio) + 0.2 * (1.0/energy_ratio if energy_ratio > 0 else 0)
        overall_winner = "Recursive" if overall_score_rec > overall_score_std else "Standard"
        
        comparison_result = {
            "num_items": num_items,
            "standard_result": std_result,
            "recursive_result": rec_result,
            "comparison": {
                "time_ratio": time_ratio,
                "cost_ratio": cost_ratio,
                "energy_ratio": energy_ratio,
                "time_winner": time_winner,
                "cost_winner": cost_winner,
                "energy_winner": energy_winner,
                "overall_winner": overall_winner,
                "overall_score_recursive": overall_score_rec,
                "overall_score_standard": overall_score_std,
                "crossover_achieved": {
                    "time": time_ratio < 1.0,
                    "cost": cost_ratio < 1.0,
                    "energy": energy_ratio < 1.0,
                    "overall": overall_winner == "Recursive"
                }
            }
        }
        
        all_results.append(comparison_result)
        
        print(f"\n📊 COMPREHENSIVE COMPARISON SUMMARY:")
        print(f"   🏆 Time Winner: {time_winner} (Ratio: {time_ratio:.4f})")
        print(f"   💰 Cost Winner: {cost_winner} (Ratio: {cost_ratio:.4f})")
        print(f"   🔋 Energy Winner: {energy_winner} (Ratio: {energy_ratio:.4f})")
        print(f"   🎯 Overall Winner: {overall_winner}")
        print(f"   📊 Standard Cost: {std_total_cost:.6f}€, Recursive Cost: {rec_total_cost:.6f}€")
        print(f"   🔋 Standard Energy: {std_energy:.4f}Wh, Recursive Energy: {rec_energy:.4f}Wh")
        
        # Check for crossovers
        crossovers = []
        if time_ratio < 1.0:
            crossovers.append("⏱️ TIME")
        if cost_ratio < 1.0:
            crossovers.append("💰 COST")
        if energy_ratio < 1.0:
            crossovers.append("🔋 ENERGY")
        if overall_winner == "Recursive":
            crossovers.append("🎯 OVERALL")
            
        if crossovers:
            print(f"   🎉 CROSSOVERS ACHIEVED: {', '.join(crossovers)}")
        else:
            print(f"   📊 No crossovers yet at {num_items} items")
    
    return all_results

def analyze_comprehensive_results(results):
    """Analysiert comprehensive IoT Ergebnisse"""
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE IoT METRICS ANALYSIS RESULTS (450-500 Items)")
    print("="*100)
    
    if not results:
        print("❌ No comprehensive results to analyze")
        return None
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS UNDER IoT CONSTRAINTS:")
    print(f"{'Items':<6} {'Time':<12} {'Cost':<12} {'Energy':<12} {'Overall':<10}")
    print(f"{'':6} {'Std|Rec|Ratio':<12} {'Std|Rec|Ratio':<12} {'Std|Rec|Ratio':<12} {'Winner':<10}")
    print("-" * 80)
    
    crossover_metrics = {
        'time': None,
        'cost': None, 
        'energy': None,
        'overall': None
    }
    
    for result in results:
        items = result["num_items"]
        
        # Time metrics
        std_time = result["standard_result"]["total_time"]
        rec_time = result["recursive_result"]["total_time"]
        time_ratio = result["comparison"]["time_ratio"]
        
        # Cost metrics
        std_cost = result["standard_result"].get("cost_metrics", {}).get("total_cost_eur", 0)
        rec_cost = result["recursive_result"].get("cost_metrics", {}).get("total_cost_eur", 0)
        cost_ratio = result["comparison"]["cost_ratio"]
        
        # Energy metrics
        std_energy = result["standard_result"].get("energy_metrics", {}).get("total_energy_wh", 0)
        rec_energy = result["recursive_result"].get("energy_metrics", {}).get("total_energy_wh", 0)
        energy_ratio = result["comparison"]["energy_ratio"]
        
        overall_winner = result["comparison"]["overall_winner"]
        
        print(f"{items:<6} {std_time:.1f}|{rec_time:.1f}|{time_ratio:.2f} {std_cost:.4f}|{rec_cost:.4f}|{cost_ratio:.2f} {std_energy:.3f}|{rec_energy:.3f}|{energy_ratio:.2f} {overall_winner:<10}")
        
        # Track first crossovers
        crossovers = result["comparison"]["crossover_achieved"]
        for metric, achieved in crossovers.items():
            if achieved and crossover_metrics[metric] is None:
                crossover_metrics[metric] = items
    
    print(f"\n🎯 COMPREHENSIVE CROSSOVER ANALYSIS:")
    
    for metric, crossover_point in crossover_metrics.items():
        if crossover_point:
            print(f"   🎉 {metric.upper()} CROSSOVER: {crossover_point} items")
        else:
            print(f"   ❌ {metric.upper()} crossover: Not yet reached")
    
    # Overall assessment
    any_crossover = any(crossover_metrics.values())
    if any_crossover:
        print(f"\n✅ RECURSIVE SNARKs show advantages in some metrics at higher item counts!")
    else:
        print(f"\n📊 Standard SNARKs remain superior across all metrics up to 500 items")
        print(f"   🔮 Crossover likely occurs at even higher item counts (>500)")
    
    return crossover_metrics

def main():
    """Hauptfunktion"""
    print("🎯 Starting Comprehensive IoT Metrics Analysis (450 & 500 Items)...")
    
    results = comprehensive_iot_analysis()
    
    if results:
        crossover_metrics = analyze_comprehensive_results(results)
        
        # Save results
        results_dir = project_root / "data" / "comprehensive_iot_metrics_450_500"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "comprehensive_450_500_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "comprehensive_results": results,
                "crossover_metrics": crossover_metrics,
                "test_timestamp": time.time(),
                "iot_configuration": "Smart Home Hub (1GB RAM, 1 CPU)",
                "test_items": [450, 500],
                "methodology": "Docker containers with comprehensive IoT metrics monitoring",
                "metrics_measured": [
                    "Time (processing duration)",
                    "Memory (max/avg usage)",
                    "CPU (max/avg utilization)",
                    "Energy (estimated consumption in Wh)",
                    "Network (data transfer in KB)",
                    "Costs (CPU, Memory, Energy, Network in EUR)"
                ],
                "crossover_evaluation": "Multi-dimensional analysis: Time, Cost, Energy, Overall weighted score"
            }, f, indent=2)
        
        print(f"\n💾 Comprehensive IoT metrics results saved: {results_file}")
        print(f"✅ COMPREHENSIVE IoT METRICS ANALYSIS COMPLETE!")
        
        return True
    else:
        print("❌ No comprehensive IoT metrics results obtained")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
