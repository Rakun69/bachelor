#!/usr/bin/env python3
"""
Simple test for ZoKrates Nova with correct JSON format
Based on the web search documentation
"""

import os
import json
import subprocess
from pathlib import Path

def test_simple_nova():
    """Test Nova with simple example from documentation"""
    
    # Create a simple test circuit
    simple_circuit = """
// Simple Nova circuit for testing
struct State {
    field sum;
}

struct StepInput {
    field element;
}

def main(public State state, private StepInput step_input) -> State {
    return State {
        sum: state.sum + step_input.element
    };
}
"""
    
    # Create test workspace
    test_dir = Path("nova_simple_test")
    test_dir.mkdir(exist_ok=True)
    
    # Save circuit
    circuit_file = test_dir / "sum.zok"
    with open(circuit_file, 'w') as f:
        f.write(simple_circuit)
    
    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    
    try:
        print("🧪 Testing simple Nova circuit...")
        
        # 1. Compile with Pallas curve
        print("📋 Compiling circuit...")
        result = subprocess.run(
            ["zokrates", "compile", "-i", "sum.zok", "--curve", "pallas"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
        
        print("✅ Circuit compiled successfully")
        
        # 2. Create init.json - initial state
        init_state = '"0"'  # Simple field value
        with open("init.json", 'w') as f:
            f.write(init_state)
        
        # 3. Create steps.json - array of steps
        steps = [["1"], ["7"], ["42"]]  # Each step is an array with one element
        with open("steps.json", 'w') as f:
            json.dump(steps, f)
        
        print("📄 Created init.json and steps.json")
        
        # 4. Generate Nova proof
        print("🔄 Generating Nova proof...")
        result = subprocess.run(
            ["zokrates", "nova", "prove"],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Nova proof generated successfully!")
            
            # 5. Try to compress
            print("📦 Compressing proof...")
            compress_result = subprocess.run(
                ["zokrates", "nova", "compress"],
                capture_output=True, text=True, timeout=30
            )
            
            if compress_result.returncode == 0:
                print("✅ Proof compressed successfully")
            else:
                print(f"⚠️  Compression failed: {compress_result.stderr}")
            
            # 6. Verify
            print("🔍 Verifying proof...")
            verify_result = subprocess.run(
                ["zokrates", "nova", "verify"],
                capture_output=True, text=True, timeout=30
            )
            
            if verify_result.returncode == 0:
                print("✅ Proof verified successfully!")
                print("🎉 Simple Nova test PASSED!")
                return True
            else:
                print(f"❌ Verification failed: {verify_result.stderr}")
                return False
        else:
            print(f"❌ Nova proof failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_simple_nova()
    if success:
        print("\n🎯 Simple Nova test successful!")
        print("Now the IoT circuit should work with similar format...")
    else:
        print("\n❌ Simple Nova test failed")
        print("Check ZoKrates Nova configuration")
