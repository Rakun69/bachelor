#!/bin/bash

# Build script for ZoKrates Nova Integration
# This script sets up ZoKrates with Nova support and tests the integration

set -e  # Exit on any error

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites for ZoKrates Nova..."
    
    # Check ZoKrates
    if ! command_exists zokrates; then
        print_error "ZoKrates is required but not installed"
        print_status "Install ZoKrates from: https://zokrates.github.io/introduction.html"
        print_status "Run: curl -LSfs get.zokrat.es | sh"
        exit 1
    fi
    
    # Check ZoKrates version and Nova support
    print_status "Checking ZoKrates Nova support..."
    if ! zokrates nova --help >/dev/null 2>&1; then
        print_error "ZoKrates Nova support not found"
        print_status "Please update ZoKrates to the latest version with Nova support"
        print_status "Current ZoKrates version: $(zokrates --version)"
        exit 1
    fi
    
    # Check Python 3
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
    print_status "ZoKrates version: $(zokrates --version)"
}

# Setup Nova circuits directory
setup_nova_circuits() {
    print_status "Setting up Nova circuits directory..."
    
    # Create nova circuits directory
    mkdir -p circuits/nova
    
    # Ensure the IoT recursive circuit exists
    if [ ! -f "circuits/nova/iot_recursive.zok" ]; then
        print_error "Nova circuit file not found: circuits/nova/iot_recursive.zok"
        print_status "The circuit should have been created by the integration script"
        exit 1
    fi
    
    print_success "Nova circuits directory ready"
}

# Test ZoKrates Nova compilation
test_nova_compilation() {
    print_status "Testing ZoKrates Nova circuit compilation..."
    
    # Create working directory
    mkdir -p nova_test_workspace
    cd nova_test_workspace
    
    # Copy circuit to working directory
    cp ../circuits/nova/iot_recursive.zok .
    
    # Test compilation with Pallas curve (required for Nova)
    print_status "Compiling with Pallas curve (required for Nova)..."
    if zokrates compile -i iot_recursive.zok --curve pallas; then
        print_success "Nova circuit compiled successfully"
        compilation_success=true
    else
        print_error "Nova circuit compilation failed"
        compilation_success=false
    fi
    
    # Return to original directory
    cd ..
    
    if [ "$compilation_success" = true ]; then
        print_success "ZoKrates Nova compilation test passed"
        return 0
    else
        print_error "ZoKrates Nova compilation test failed"
        return 1
    fi
}

# Test Nova proof generation with simple example
test_nova_proof() {
    print_status "Testing Nova proof generation..."
    
    cd nova_test_workspace
    
    # Create simple test state and steps
    echo '{"sum_temperature": "0", "sum_humidity": "0", "sum_motion": "0", "count_readings": "0", "min_temp": "0", "max_temp": "0", "privacy_violations": "0", "last_batch_hash": "0"}' > init_state.json
    
    # Create test step with simple IoT data
    cat > test_steps.json << 'EOF'
[{
    "sensor_readings": ["2000", "3000", "0", "0", "0", "0", "0", "0", "0", "0"],
    "sensor_types": ["1", "2", "0", "0", "0", "0", "0", "0", "0", "0"],
    "privacy_levels": ["1", "1", "0", "0", "0", "0", "0", "0", "0", "0"],
    "room_ids": ["1", "1", "0", "0", "0", "0", "0", "0", "0", "0"],
    "batch_id": "1"
}]
EOF
    
    # Try to generate Nova proof
    print_status "Generating Nova recursive proof..."
    if timeout 60 zokrates nova prove; then
        print_success "Nova proof generation successful"
        
        # Test compression if available
        if zokrates nova compress; then
            print_success "Nova proof compression successful"
        else
            print_warning "Nova proof compression failed (not critical)"
        fi
        
        # Test verification
        if zokrates nova verify; then
            print_success "Nova proof verification successful"
            proof_success=true
        else
            print_error "Nova proof verification failed"
            proof_success=false
        fi
    else
        print_error "Nova proof generation failed or timed out"
        proof_success=false
    fi
    
    cd ..
    
    if [ "$proof_success" = true ]; then
        print_success "Nova proof test passed"
        return 0
    else
        print_warning "Nova proof test failed (circuit may need adjustment)"
        return 1
    fi
}

# Test Python integration
test_python_integration() {
    print_status "Testing Python integration with ZoKrates Nova..."
    
    # Activate virtual environment if available
    if [ -d "iot_zk_env" ]; then
        source iot_zk_env/bin/activate
        print_status "Activated virtual environment: iot_zk_env"
    fi
    
    # Test Python import and basic functionality
    python3 -c "
import sys
sys.path.append('src')

try:
    from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
    print('✅ ZoKrates Nova Manager imported successfully')
    
    # Test manager creation
    manager = ZoKratesNovaManager('circuits/nova/iot_recursive.zok', batch_size=10)
    print('✅ Nova manager created successfully')
    
    # Test Nova support check
    nova_available = manager.check_zokrates_nova_support()
    if nova_available:
        print('✅ ZoKrates Nova support detected')
    else:
        print('⚠️  ZoKrates Nova support not detected')
        exit(1)
    
    print('🎉 Python integration test passed!')
    
except ImportError as e:
    print(f'❌ Failed to import Nova manager: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Python integration test failed: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Python integration test passed"
        return 0
    else
        print_error "Python integration test failed"
        return 1
    fi
}

# Run Nova demo with real IoT data
run_nova_demo() {
    print_status "Running ZoKrates Nova demonstration..."
    
    # Activate virtual environment if available
    if [ -d "iot_zk_env" ]; then
        source iot_zk_env/bin/activate
    fi
    
    # Run integrated demo
    python3 -c "
import sys
sys.path.append('src')

try:
    from proof_systems.zokrates_nova_manager import ZoKratesNovaManager
    import json
    import time
    
    print('🔥 ZoKrates Nova Recursive SNARK Demo')
    print('=' * 50)
    
    # Create manager
    manager = ZoKratesNovaManager('circuits/nova/iot_recursive.zok', batch_size=5)
    
    # Setup (compile circuit)
    print('📋 Setting up ZoKrates Nova...')
    setup_success = manager.setup()
    
    if setup_success:
        print('✅ Setup successful - circuit compiled')
        
        # Create sample IoT data
        test_data = []
        for i in range(15):  # 3 batches of 5 readings each
            test_data.append({
                'sensor_id': f'demo_sensor_{i}',
                'sensor_type': 'temperature' if i % 3 == 0 else ('humidity' if i % 3 == 1 else 'motion'),
                'room': f'room_{(i % 3) + 1}',
                'value': 20.0 + i * 0.5,
                'privacy_level': (i % 3) + 1,
                'timestamp': int(time.time()) + i * 60
            })
        
        print(f'📊 Generated {len(test_data)} test IoT readings')
        
        # Split into batches
        batches = []
        for i in range(0, len(test_data), 5):
            batch = test_data[i:i+5]
            batches.append(batch)
        
        print(f'📦 Split into {len(batches)} batches')
        
        # Generate recursive proof
        print('🔄 Generating recursive Nova proof...')
        result = manager.prove_recursive_batch(batches)
        
        if result.success:
            print('🎉 ZoKrates Nova proof successful!')
            print(f'   📈 Steps: {result.step_count}')
            print(f'   ⏱️  Total time: {result.total_time:.3f}s')
            print(f'   ✅ Verify time: {result.verify_time:.3f}s')
            print(f'   📦 Proof size: {result.proof_size} bytes')
            print('🏆 Demo completed successfully!')
        else:
            print(f'❌ Nova proof failed: {result.error_message}')
    else:
        print('❌ Setup failed - check ZoKrates Nova support')
    
except Exception as e:
    print(f'❌ Demo failed: {e}')
    import traceback
    traceback.print_exc()
"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    if [ -d "nova_test_workspace" ]; then
        rm -rf nova_test_workspace
        print_status "Removed test workspace"
    fi
}

# Main execution
main() {
    echo "🚀 ZoKrates Nova Integration Setup & Test"
    echo "=========================================="
    
    check_prerequisites
    setup_nova_circuits
    
    # Track test results
    compilation_ok=false
    proof_ok=false  
    python_ok=false
    
    if test_nova_compilation; then
        compilation_ok=true
        
        if test_nova_proof; then
            proof_ok=true
        fi
    fi
    
    if test_python_integration; then
        python_ok=true
    fi
    
    echo ""
    echo "=========================================="
    print_status "Test Summary:"
    
    if [ "$compilation_ok" = true ]; then
        print_success "✅ Circuit compilation working"
    else
        print_error "❌ Circuit compilation failed"
    fi
    
    if [ "$proof_ok" = true ]; then
        print_success "✅ Nova proof generation working"
    else
        print_warning "⚠️  Nova proof generation issues"
    fi
    
    if [ "$python_ok" = true ]; then
        print_success "✅ Python integration working"
    else
        print_error "❌ Python integration failed"
    fi
    
    if [ "$compilation_ok" = true ] && [ "$python_ok" = true ]; then
        print_success "🎉 ZoKrates Nova integration ready!"
        echo ""
        print_status "Next steps:"
        echo "  1. Run: python3 demo.py (includes Nova demo)"
        echo "  2. Run: python3 src/orchestrator.py (full evaluation)"
        echo "  3. Check: data/benchmarks/nova_vs_zokrates_comparison.json"
        
        # Run demo if everything works
        echo ""
        run_nova_demo
    else
        print_warning "⚠️  Some tests failed - check configuration"
    fi
    
    cleanup
}

# Help function
show_help() {
    echo "ZoKrates Nova Integration Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h      Show this help message"
    echo "  --test-only     Only run tests, don't setup"
    echo "  --demo-only     Only run demo"
    echo "  --cleanup       Clean up test artifacts"
    echo ""
    echo "This script sets up and tests ZoKrates Nova recursive SNARKs"
    echo "for IoT data processing."
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --test-only)
        check_prerequisites
        test_nova_compilation && test_nova_proof
        test_python_integration
        exit $?
        ;;
    --demo-only)
        run_nova_demo
        exit $?
        ;;
    --cleanup)
        cleanup
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
