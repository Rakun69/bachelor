# Verifiable Data Transformations in IoT Environments using Recursive zk-SNARKs

## Bachelorarbeit - Smart Home Privacy-Preserving Data Processing

Dieses Projekt implementiert ein System zur verifizierbaren und privacy-preserving Verarbeitung von IoT-Sensordaten in Smart Home Umgebungen unter Verwendung von zk-SNARKs und Recursive SNARKs.

## 🎯 Projektübersicht

### Ziele
- **Simulation** eines Smart Home IoT-Netzwerks mit verschiedenen Sensoren
- **Implementierung** verschiedener Zero-Knowledge Proof Systeme
- **Vergleich** zwischen Standard zk-SNARKs und Recursive SNARKs  
- **Evaluation** von Performance, Privacy und Skalierbarkeit
- **Bestimmung** optimaler Einsatzbedingungen für Recursive SNARKs

### Forschungsfragen
1. Unter welchen Bedingungen lohnt sich der Einsatz von Recursive SNARKs?
2. Welchen Mehrwert bieten Recursive SNARKs im Bereich Privacy?
3. Ab welcher Datenmenge/Komplexität sind Recursive SNARKs effizienter?
4. Wie verhalten sich verschiedene ZKP-Systeme bei IoT-Datenverarbeitung?

## 🏗️ Architektur

```
IoT Devices → Data Ingestion → Batch Processing → ZK Circuits → Proof Generation
     ↓              ↓              ↓              ↓              ↓
Sensordaten → Filterung/Agg. → Batching → SNARK/Rec.SNARK → Verification
```

## 📁 Projektstruktur

```
bachelor/
├── src/
│   ├── iot_simulation/          # Smart Home IoT Simulation
│   ├── circuits/                # ZoKrates Circuits  
│   ├── proof_systems/           # SNARK & Recursive SNARK Implementation
│   ├── data_processing/         # Data Ingestion & Transformation
│   ├── evaluation/              # Performance & Privacy Evaluation
│   └── utils/                   # Helper Functions
├── data/                        # Datasets & Outputs
├── circuits/                    # ZoKrates .zok files
├── docs/                        # Documentation
├── tests/                       # Unit Tests
└── configs/                     # Configuration Files
```

## 🚀 Implemented Features

### IoT Simulation
- **Temperatur-, Luftfeuchtigkeits-, Bewegungssensoren**
- **Realistische Datenpattern** mit Tages-/Nachtzyklen
- **Configurable Sensor Network** Topologie
- **Privacy-sensitive Data Generation**

### ZK Circuits (ZoKrates)
- **Basic Aggregation**: Sum, Count, Average
- **Statistical Functions**: Min, Max, Median, Standard Deviation
- **Privacy Filters**: Range checks, Outlier detection
- **Composite Functions**: Multi-sensor correlation

### Proof Systems
- **Standard zk-SNARKs**: Single-step proofs
- **Recursive SNARKs**: Batch processing with proof composition
- **Performance Metrics**: Proof generation time, verification time, proof size
- **Privacy Analysis**: Information leakage assessment

### Evaluation Framework
- **Automated Benchmarking** verschiedener Proof-Systeme
- **Scalability Tests** mit verschiedenen Datenmengen
- **Privacy Metrics** Assessment
- **Cost-Benefit Analysis** für Recursive SNARKs

## 📊 Evaluation Metrics

### Performance
- Proof Generation Time
- Verification Time  
- Proof Size
- Memory Usage
- Circuit Compilation Time

### Privacy
- Information Leakage Analysis
- Differential Privacy Metrics
- Anonymity Set Size
- Re-identification Risk

### Scalability  
- Throughput (Proofs/second)
- Latency vs. Batch Size
- Resource Utilization
- Network Overhead

## 🛠️ Technologies

- **ZoKrates**: zk-SNARK Circuit Development
- **Python**: Simulation & Data Processing
- **Numpy/Pandas**: Data Analysis
- **Matplotlib/Seaborn**: Visualization
- **JSON**: Data Exchange Format
- **GitHub Actions**: CI/CD

## 📈 Expected Results

1. **Threshold Analysis**: Bestimmung ab welcher Datenmenge Recursive SNARKs effizienter sind
2. **Privacy-Performance Trade-offs**: Quantifizierung des Privacy-Mehrwerts
3. **Use Case Recommendations**: Guidelines für optimale ZKP-System Auswahl
4. **Benchmark Suite**: Standardisierte Tests für IoT-ZKP Systeme

## 🔧 Getting Started

### Prerequisites
```bash
# Install ZoKrates
curl -LSfs get.zokrat.es | sh

# Install Python dependencies  
pip install -r requirements.txt
```

### Quick Start
```bash
# 1) Create & activate venv, install deps
python3 -m venv iot_zk_env
source iot_zk_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Verify ZoKrates (install if missing)
zokrates --version || (curl -LSfs get.zokrat.es | sh && export PATH="$HOME/.zokrates/bin:$PATH")

# 3) Run demo and basic visualizations
./run_demo.sh demo

# 4) Full evaluation workflow
./run_evaluation.sh --setup
./run_evaluation.sh --phase data
./run_evaluation.sh --phase compile
./run_evaluation.sh --phase benchmark
./run_evaluation.sh --phase analyze
./run_evaluation.sh --phase visualize

# Or run everything in one go
./run_evaluation.sh
```

## 📚 Documentation

- Getting Started: `docs/GETTING_STARTED.md`
- Generated docs and reports are written to `data/` and `docs/generated/`

## 👨‍🎓 Bachelor Thesis Context

This project serves as the practical implementation for the bachelor thesis:
**"Verifiable Data Transformations in IoT Environments using Recursive zk-SNARKs"**

Supervisor recommendations incorporated:
- ✅ Focus on evaluation and comparison
- ✅ Multiple ZKP systems beyond just zk-SNARKs  
- ✅ Privacy-preserving aspects
- ✅ Threshold analysis for Recursive SNARK adoption
- ✅ Comprehensive metrics framework
- ✅ Realistic use case formulation

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---
*Developed as part of Bachelor Thesis at [University Name] - [Year]*