#!/bin/bash

# Wrapper script to run demos with proper environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to activate virtual environment
activate_venv() {
    if [ -d "iot_zk_env" ]; then
        source iot_zk_env/bin/activate
        print_status "Activated virtual environment: iot_zk_env"
        return 0
    else
        print_error "Virtual environment not found: iot_zk_env"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running ZoKrates Nova integration tests..."
    
    if activate_venv; then
        python3 test_zokrates_nova_integration.py
        return $?
    else
        return 1
    fi
}

# Function to run demo
run_demo() {
    print_status "Running IoT ZK-SNARK demo..."
    
    if activate_venv; then
        python3 demo.py
        return $?
    else
        return 1
    fi
}

# Function to run orchestrator
run_orchestrator() {
    print_status "Running full orchestrator evaluation..."
    
    if activate_venv; then
        python3 src/orchestrator.py --phase all
        return $?
    else
        return 1
    fi
}

# Main function
main() {
    echo "🚀 IoT ZK-SNARK with ZoKrates Nova - Demo Runner"
    echo "=================================================="
    
    case "${1:-demo}" in
        "test"|"tests")
            run_integration_tests
            ;;
        "demo")
            run_demo
            ;;
        "orchestrator"|"full")
            run_orchestrator
            ;;
        "all")
            print_status "Running complete demo suite..."
            echo ""
            
            print_status "1. Integration Tests"
            run_integration_tests
            echo ""
            
            print_status "2. Demo"
            run_demo
            echo ""
            
            print_status "3. Full Orchestrator"
            run_orchestrator
            ;;
        "--help"|"-h")
            echo "Usage: $0 [test|demo|orchestrator|all]"
            echo ""
            echo "Commands:"
            echo "  test        - Run integration tests"
            echo "  demo        - Run IoT demo"
            echo "  orchestrator - Run full evaluation"
            echo "  all         - Run everything"
            echo ""
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"
