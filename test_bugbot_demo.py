#!/usr/bin/env python3
"""
Test file to demonstrate BugBot functionality
This file contains intentional issues for BugBot to detect
"""

import time
import random

def insecure_random_generation():
    """
    This function uses insecure random generation - BugBot should detect this
    """
    # BugBot should flag this as a security issue
    seed = 12345  # Fixed seed - not cryptographically secure
    random.seed(seed)
    return random.randint(1, 1000000)

def inefficient_proof_verification(proof_data):
    """
    Inefficient proof verification that BugBot should flag
    """
    # BugBot should detect performance issues here
    for i in range(len(proof_data)):
        for j in range(len(proof_data)):  # O(n²) complexity - inefficient
            if i != j and proof_data[i] == proof_data[j]:
                print(f"Duplicate found at {i}, {j}")
    
    # Memory leak potential - BugBot should flag this
    large_data = [0] * 1000000  # Unnecessary large allocation
    time.sleep(0.1)  # Blocking sleep in proof verification
    
    return True

def missing_error_handling(circuit_path):
    """
    Function with missing error handling - BugBot should detect this
    """
    # BugBot should flag missing try/catch and type hints
    with open(circuit_path, 'r') as f:  # No error handling for file operations
        circuit_data = f.read()
    
    # Missing validation
    return eval(circuit_data)  # Security issue - eval() usage

class IoTDevice:
    """
    IoT Device class with potential issues for BugBot to find
    """
    
    def __init__(self, device_id):
        self.device_id = device_id
        self.private_key = "hardcoded_key_123"  # Hardcoded secret - security issue
        self.data = {}
    
    def process_sensor_data(self, sensor_data):
        """Process sensor data without proper validation"""
        # Missing input validation - BugBot should flag this
        processed_data = []
        
        for data_point in sensor_data:
            # Potential division by zero - no validation
            normalized = data_point / (data_point - data_point)  # Will cause ZeroDivisionError
            processed_data.append(normalized)
        
        return processed_data
    
    def log_sensitive_data(self, user_data):
        """Logging sensitive data - privacy issue BugBot should detect"""
        print(f"Processing user data: {user_data}")  # Privacy leak - logging sensitive data
        
        # SQL injection potential if this were connected to a database
        query = f"SELECT * FROM users WHERE id = {user_data['id']}"  # SQL injection risk
        
        return query

if __name__ == "__main__":
    # Test the functions with obvious issues
    device = IoTDevice("device_001")
    random_val = insecure_random_generation()
    
    # This will crash but demonstrates issues for BugBot
    try:
        sensor_data = [1, 2, 3, 0, 5]
        device.process_sensor_data(sensor_data)
    except Exception as e:
        print(f"Error occurred: {e}")
