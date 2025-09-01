# BugBot Configuration for IoT Zero-Knowledge Proof System

## Project Context
This is a Bachelor thesis implementation of an IoT Zero-Knowledge Proof System comparing ZoKrates SNARK and Nova recursive proofs.

## Review Focus Areas

### 🔐 Security & Privacy
- Pay special attention to cryptographic implementations
- Review zero-knowledge proof generation and verification logic  
- Check for potential side-channel attacks in timing measurements
- Validate privacy-preserving data handling

### ⚡ Performance & Optimization
- Review proof generation and verification performance
- Check memory usage in large-scale benchmarks
- Validate batch processing optimizations
- Monitor recursive proof accumulation efficiency

### 🏠 IoT Constraints
- Ensure resource-constrained device compatibility
- Review Docker containerization for embedded systems
- Check power consumption considerations
- Validate network bandwidth optimization

### 🧪 Test Coverage
- Ensure comprehensive test coverage for proof systems
- Review crossover point analysis accuracy
- Validate benchmark consistency
- Check edge cases in large data scenarios

### 📊 Data Analysis
- Review statistical analysis methods
- Validate crossover point calculations
- Check data visualization accuracy
- Ensure reproducible results

## Specific Code Patterns to Check

### Python Best Practices
- Async/await usage in proof generation
- Error handling in cryptographic operations
- Memory management for large datasets
- Type hints for complex data structures

### ZoKrates Integration
- Circuit compilation validation
- Witness generation correctness
- Proof serialization/deserialization
- ABI handling

### Nova Implementation
- Recursive proof folding validation
- Step function correctness
- Public parameter generation
- Verification key management

## Ignore Patterns
- Temporary benchmark files in `/data/`
- Generated proof files (*.proof)
- Compiled circuits (*.r1cs, out/)
- Virtual environment artifacts

## Performance Thresholds
- Proof generation should complete within reasonable time limits
- Memory usage should not exceed available IoT device constraints
- Batch processing should scale linearly

## Security Requirements
- All cryptographic randomness must be secure
- Private data must never leak in logs or outputs
- Proof verification must be constant-time where possible
