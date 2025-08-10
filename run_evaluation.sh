#!/bin/bash

# IoT ZK-SNARK Evaluation System - Main Execution Script
# This script runs the complete evaluation workflow

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_ROOT="/home/ramon/bachelor"
VENV_NAME="iot_zk_env"
PYTHON_VERSION="3.8"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Check if Python 3.8+ is available
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_NAME" ]; then
        print_status "Creating virtual environment: $VENV_NAME"
        python3 -m venv "$VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found, skipping dependency installation"
    fi
}

# Function to check ZoKrates installation
check_zokrates() {
    print_status "Checking ZoKrates installation..."
    
    if command_exists zokrates; then
        ZOKRATES_VERSION=$(zokrates --version 2>&1 | head -n1)
        print_success "ZoKrates found: $ZOKRATES_VERSION"
    else
        print_warning "ZoKrates not found. Installing..."
        
        # Install ZoKrates
        curl -LSfs get.zokrat.es | sh
        
        # Add to PATH for current session
        export PATH="$HOME/.zokrates/bin:$PATH"
        
        if command_exists zokrates; then
            print_success "ZoKrates installed successfully"
        else
            print_error "Failed to install ZoKrates"
            print_error "Please install ZoKrates manually: https://zokrates.github.io/gettingstarted.html"
            exit 1
        fi
    fi
}

# Function to setup project directories
setup_directories() {
    print_status "Setting up project directories..."
    
    # Create necessary directories
    mkdir -p data/{raw,processed,proofs,benchmarks}
    mkdir -p logs
    mkdir -p docs/generated
    mkdir -p tests/outputs
    
    print_success "Project directories created"
}

# Function to run IoT data generation
run_data_generation() {
    print_status "Generating IoT simulation data..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    python src/iot_simulation/smart_home.py
    
    if [ $? -eq 0 ]; then
        print_success "IoT data generation completed"
    else
        print_error "IoT data generation failed"
        return 1
    fi
}

# Function to compile ZK circuits
compile_circuits() {
    print_status "Compiling ZK circuits..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    # Run circuit compilation through SNARK manager
    python -c "
from pathlib import Path
from src.proof_systems.snark_manager import SNARKManager

project_root = Path('$PROJECT_ROOT')
manager = SNARKManager(circuits_dir=str(project_root / 'circuits'), output_dir=str(project_root / 'data' / 'proofs'))

circuits = [
    (str(project_root / 'circuits/basic/filter_range.zok'), 'filter_range'),
    (str(project_root / 'circuits/basic/min_max.zok'), 'min_max'),
    (str(project_root / 'circuits/basic/median.zok'), 'median'),
    (str(project_root / 'circuits/advanced/aggregation.zok'), 'aggregation'),
    (str(project_root / 'circuits/recursive/batch_processor.zok'), 'batch_processor')
]

for circuit_path, name in circuits:
    try:
        if manager.compile_circuit(circuit_path, name):
            manager.setup_circuit(name)
            print(f'Successfully compiled and setup: {name}')
        else:
            print(f'Failed to compile: {name}')
    except Exception as e:
        print(f'Error with {name}: {e}')
"
    
    if [ $? -eq 0 ]; then
        print_success "Circuit compilation completed"
    else
        print_warning "Some circuits may have failed to compile"
    fi
}

# Function to run benchmarks
run_benchmarks() {
    print_status "Running comprehensive benchmarks..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    # Run the full evaluation
    python src/orchestrator.py --phase benchmark
    
    if [ $? -eq 0 ]; then
        print_success "Benchmarks completed"
    else
        print_error "Benchmark execution failed"
        return 1
    fi
}

# Function to run complete evaluation
run_complete_evaluation() {
    print_status "Running complete evaluation workflow..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    # Run the orchestrator
    python src/orchestrator.py --phase all --config configs/default_config.json
    
    if [ $? -eq 0 ]; then
        print_success "Complete evaluation finished successfully"
        
        # Show summary
        echo ""
        echo "=== EVALUATION SUMMARY ==="
        echo "Results saved to: $PROJECT_ROOT/data/"
        echo "Final report: $PROJECT_ROOT/data/final_report.json"
        echo "Benchmarks: $PROJECT_ROOT/data/benchmarks/"
        echo "IoT data: $PROJECT_ROOT/data/raw/"
        echo "Proofs: $PROJECT_ROOT/data/proofs/"
        echo "Visualizations: $PROJECT_ROOT/data/visualizations/"
        
    else
        print_error "Complete evaluation failed"
        return 1
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    # Run pytest if available
    if command_exists pytest; then
        pytest tests/ -v --cov=src --cov-report=html
        print_success "Tests completed. Coverage report in htmlcov/"
    else
        print_warning "pytest not available, skipping tests"
    fi
}

# Function to generate documentation
generate_docs() {
    print_status "Generating documentation..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    # Create simple documentation
    cat > docs/generated/README.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>IoT ZK-SNARK Evaluation Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1, h2 { color: #2c3e50; }
        code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
        pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>IoT ZK-SNARK Evaluation System</h1>
    <h2>Project Overview</h2>
    <p>This system evaluates the performance, privacy, and scalability of standard zk-SNARKs versus recursive SNARKs for IoT data processing.</p>
    
    <h2>Architecture</h2>
    <ul>
        <li><strong>IoT Simulation:</strong> Smart home sensor data generation</li>
        <li><strong>ZK Circuits:</strong> Various privacy-preserving computation circuits</li>
        <li><strong>Proof Systems:</strong> Standard and recursive SNARK implementations</li>
        <li><strong>Evaluation Framework:</strong> Comprehensive benchmarking and analysis</li>
    </ul>
    
    <h2>Usage</h2>
    <pre><code># Run complete evaluation
./run_evaluation.sh

# Run specific phases
./run_evaluation.sh --phase data
./run_evaluation.sh --phase compile  
./run_evaluation.sh --phase benchmark</code></pre>
    
    <h2>Results</h2>
    <p>Evaluation results are saved in the <code>data/</code> directory with comprehensive reports and visualizations.</p>
</body>
</html>
EOF
    
    print_success "Documentation generated in docs/generated/"
}

# Function to show help
show_help() {
    echo "IoT ZK-SNARK Evaluation System"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --setup             Setup environment and dependencies"
    echo "  --phase <phase>     Run specific phase: data|compile|benchmark|analyze|visualize|all"
    echo "  --test              Run tests"
    echo "  --docs              Generate documentation"
    echo "  --clean             Clean generated files"
    echo ""
    echo "Phases:"
    echo "  data               Generate IoT simulation data"
    echo "  compile            Compile ZK circuits"
    echo "  benchmark          Run performance benchmarks"
    echo "  analyze            Analyze results"
    echo "  visualize          Generate household activity visualizations"
    echo "  all                Run complete evaluation (default)"
    echo ""
    echo "Examples:"
    echo "  $0                 Run complete evaluation"
    echo "  $0 --setup         Setup environment"
    echo "  $0 --phase data    Generate data only"
    echo "  $0 --phase visualize  Generate visualizations"
    echo "  $0 --test          Run tests"
}

# Function to clean generated files
clean_files() {
    print_status "Cleaning generated files..."
    
    rm -rf data/processed/* 2>/dev/null || true
    rm -rf data/proofs/* 2>/dev/null || true
    rm -rf data/benchmarks/* 2>/dev/null || true
    rm -rf data/visualizations/* 2>/dev/null || true
    rm -rf logs/* 2>/dev/null || true
    rm -rf docs/generated/* 2>/dev/null || true
    
    print_success "Cleaned generated files"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    # Parse command line arguments
    PHASE="all"
    SETUP_ONLY=false
    TEST_ONLY=false
    DOCS_ONLY=false
    CLEAN_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            --setup)
                SETUP_ONLY=true
                shift
                ;;
            --phase)
                PHASE="$2"
                shift 2
                ;;
            --test)
                TEST_ONLY=true
                shift
                ;;
            --docs)
                DOCS_ONLY=true
                shift
                ;;
            --clean)
                CLEAN_ONLY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Print header
    echo "======================================"
    echo "  IoT ZK-SNARK Evaluation System"
    echo "======================================"
    echo ""
    
    # Handle specific actions
    if [ "$CLEAN_ONLY" = true ]; then
        clean_files
        exit 0
    fi
    
    if [ "$DOCS_ONLY" = true ]; then
        generate_docs
        exit 0
    fi
    
    if [ "$TEST_ONLY" = true ]; then
        setup_python_env
        run_tests
        exit 0
    fi
    
    # Setup phase (always run unless specific phase requested)
    if [ "$SETUP_ONLY" = true ] || [ "$PHASE" = "all" ]; then
        setup_python_env
        check_zokrates
        setup_directories
        
        if [ "$SETUP_ONLY" = true ]; then
            print_success "Setup completed successfully"
            exit 0
        fi
    fi
    
    # Execute requested phase
    case $PHASE in
        "data")
            run_data_generation
            ;;
        "compile")
            compile_circuits
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "analyze")
            print_status "Running analysis phase..."
            python src/orchestrator.py --phase analyze
            ;;
        "visualize")
            print_status "Running visualization phase..."
            python demo.py
            ;;
        "all")
            run_complete_evaluation
            ;;
        *)
            print_error "Unknown phase: $PHASE"
            show_help
            exit 1
            ;;
    esac
    
    print_success "Execution completed successfully!"
}

# Run main function
main "$@"