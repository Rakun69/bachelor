# Getting Started - IoT ZK-SNARK Evaluation System

## Quick Start Guide

### Prerequisites

1. **Linux/WSL Environment** (tested on Ubuntu 20.04+)
2. **Python 3.8+** with pip
3. **curl** for downloading ZoKrates
4. **Git** (if cloning repository)

### Installation

```bash
# Navigate to project directory
cd /home/ramon/bachelor

# Run automated setup
./run_evaluation.sh --setup
```

This will:
- Create Python virtual environment
- Install all dependencies
- Download and install ZoKrates
- Setup project directories

### Quick Evaluation Run

```bash
# Run complete evaluation (takes 10-30 minutes)
./run_evaluation.sh

# Or run specific phases
./run_evaluation.sh --phase data      # Generate IoT data
./run_evaluation.sh --phase compile   # Compile circuits  
./run_evaluation.sh --phase benchmark # Run benchmarks
```

### Understanding the Output

After completion, you'll find:

```
data/
├── raw/                   # IoT simulation data
├── proofs/               # Generated ZK proofs
├── benchmarks/           # Performance results
└── final_report.json     # Complete analysis
```

## Project Structure

```
bachelor/
├── src/
│   ├── iot_simulation/          # Smart home IoT simulator
│   │   └── smart_home.py
│   ├── circuits/                # Circuit management
│   ├── proof_systems/           # SNARK & Recursive SNARK
│   │   └── snark_manager.py
│   ├── evaluation/              # Benchmarking framework
│   │   └── benchmark_framework.py
│   └── orchestrator.py          # Main coordinator
├── circuits/                    # ZoKrates .zok files
│   ├── basic/                   # Simple circuits
│   ├── advanced/               # Complex circuits
│   └── recursive/              # Recursive SNARK circuits
├── data/                       # Generated data and results
├── configs/                    # Configuration files
└── docs/                      # Documentation
```

## Core Components

### 1. IoT Simulation (`src/iot_simulation/smart_home.py`)

Simulates a realistic smart home with:
- Temperature, humidity, motion sensors
- Multiple rooms (living room, bedroom, kitchen, bathroom)
- Realistic daily patterns and noise
- Privacy-sensitive data generation

```python
from src.iot_simulation.smart_home import SmartHomeSensors

simulator = SmartHomeSensors()
readings = simulator.generate_readings(duration_hours=24)
```

### 2. ZK Circuits (`circuits/`)

Implements various privacy-preserving computations:

- **`filter_range.zok`**: Range validation without revealing exact values
- **`min_max.zok`**: Min/max computation with privacy
- **`median.zok`**: Median calculation 
- **`aggregation.zok`**: Multi-sensor statistical analysis
- **`batch_processor.zok`**: Recursive proof composition

### 3. Proof Systems (`src/proof_systems/snark_manager.py`)

Manages both standard and recursive SNARKs:

```python
from src.proof_systems.snark_manager import SNARKManager

manager = SNARKManager()
result = manager.generate_proof("filter_range", ["10", "30", "25"])
recursive_proof = manager.create_recursive_proof(individual_proofs)
```

### 4. Evaluation Framework (`src/evaluation/benchmark_framework.py`)

Comprehensive benchmarking system:
- Performance metrics (time, memory, proof size)
- Privacy analysis (information leakage, anonymity)
- Scalability testing (data size, batch size effects)
- Comparison between proof systems

## Research Questions Addressed

1. **Threshold Analysis**: When do recursive SNARKs become more efficient?
2. **Privacy Evaluation**: What privacy guarantees do different circuits provide?
3. **Scalability**: How do systems scale with data size and complexity?
4. **Use Case Optimization**: Which proof system for which scenario?

## Configuration

The system uses `configs/default_config.json` for configuration:

```json
{
  "iot_simulation": {
    "duration_hours": 24,
    "time_step_seconds": 60
  },
  "evaluation_parameters": {
    "data_sizes": [10, 50, 100, 500],
    "batch_sizes": [5, 10, 20, 50],
    "iterations": 5
  }
}
```

## Understanding Results

### Performance Comparison

The system generates comparisons between:
- **Standard SNARKs**: Individual proofs for each computation
- **Recursive SNARKs**: Batch processing with proof composition

Key metrics:
- Proof generation time
- Verification time  
- Proof size
- Memory usage
- Throughput (proofs/second)

### Privacy Analysis

Evaluates:
- Information leakage (entropy-based)
- Anonymity set size
- Re-identification risk
- Differential privacy metrics

### Scalability Results

Determines:
- Linear vs. quadratic scaling behavior
- Memory scaling patterns
- Optimal batch sizes
- Performance thresholds

## Troubleshooting

### Common Issues

1. **ZoKrates not found**
   ```bash
   # Manual installation
   curl -LSfs get.zokrat.es | sh
   export PATH="$HOME/.zokrates/bin:$PATH"
   ```

2. **Python dependencies**
   ```bash
   # Reinstall in virtual environment
   source iot_zk_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Circuit compilation errors**
   - Check ZoKrates version compatibility
   - Verify circuit syntax in `.zok` files
   - Ensure sufficient memory for compilation

### Debug Mode

Run with detailed logging:
```bash
# Set debug logging
export LOG_LEVEL=DEBUG
./run_evaluation.sh --phase all
```

### Partial Runs

Test individual components:
```bash
# Test IoT simulation only
python src/iot_simulation/smart_home.py

# Test SNARK manager
python src/proof_systems/snark_manager.py

# Test specific circuit
zokrates compile -i circuits/basic/filter_range.zok
```

## Results Interpretation

### Performance Thresholds

The evaluation determines thresholds where recursive SNARKs become beneficial:
- **Data size threshold**: Typically around 100-500 data points
- **Batch size threshold**: Usually 20-50 items per batch
- **Complexity threshold**: Depends on circuit complexity

### Privacy-Performance Trade-offs

Different circuits offer different privacy levels:
- **High Privacy**: `filter_range` (minimal information leakage)
- **Medium Privacy**: `min_max`, `median` (aggregate statistics)
- **Lower Privacy**: `aggregation` (multiple statistics revealed)

### Recommendations

The system generates specific recommendations:
- When to use standard vs. recursive SNARKs
- Optimal batch sizes for different scenarios
- Privacy considerations for different applications
- Implementation guidelines for production systems

## Next Steps

1. **Customize for your use case**: Modify circuits and evaluation parameters
2. **Extend the simulation**: Add more sensor types or scenarios
3. **Implement additional circuits**: Create domain-specific computations
4. **Deploy in production**: Use findings to implement real IoT systems
5. **Research extensions**: Investigate quantum-resistant alternatives

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `logs/`
3. Examine intermediate results in `data/`
4. Consult the generated final report

---

**Note**: This is a research prototype for bachelor thesis evaluation. For production use, additional security reviews and optimizations may be required.