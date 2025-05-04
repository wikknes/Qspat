# Understanding Data Usage in Qspat Framework

This document clarifies how data is handled in the Qspat framework, particularly the distinction between real algorithms and synthetic data.

## Real Algorithms vs. Synthetic Data

In the Qspat framework and benchmarking tools:

### Real Algorithms

The Qspat framework uses **real quantum algorithms**, not simulations. This means:

- **VQE Implementation**: Uses actual Variational Quantum Eigensolver algorithm, creating and optimizing parameterized quantum circuits
- **QAOA Implementation**: Uses actual Quantum Approximate Optimization Algorithm with problem and mixer Hamiltonians
- **Quantum Backends**: Can run on Qiskit simulators or real quantum hardware
- **Quantum Circuit Construction**: Creates real circuit definitions, not mathematical approximations
- **Hamiltonian Encoding**: Uses actual Pauli operators and quantum encoding

### Synthetic Data for Testing

While the algorithms are real, the framework currently uses synthetic spatial transcriptomics data for demonstration and benchmarking. This is because:

1. Real spatial transcriptomics datasets are often very large (thousands of spots, thousands of genes)
2. Synthetic data allows for controlled testing of specific spatial patterns
3. The patterns in the synthetic data mimic real biological phenomena (gradients, clusters, etc.)
4. Synthetic data enables reproducible testing across different systems

The synthetic data generation creates realistic patterns that represent common spatial transcriptomics phenomena:

- Gradient patterns (expression gradients across tissue)
- Central peaks (highly expressed genes in specific regions)
- Cluster patterns (multiple expression hotspots)
- Stripe patterns (structured expression zones)

## Using Your Own Data

The Qspat framework is designed to work with real spatial transcriptomics data. To use your own data:

1. Format your data as a CSV file with columns:
   - `x` and `y` coordinates for each spot
   - Gene expression values in additional columns (one gene per column)

2. Use the framework with your data file:
   ```python
   # For VQE (maximum finding)
   vqe_solver = RealSpatialVQE(
       'path/to/your_data.csv',
       target_gene='YourGene',
       max_spots=32  # Adjust based on your data size
   )
   
   # For QAOA (region detection)
   qaoa_solver = RealSpatialQAOA(
       'path/to/your_data.csv',
       target_gene='YourGene',
       max_spots=32
   )
   ```

3. For benchmarking:
   ```bash
   python benchmark_updated.py --data path/to/your_data.csv --gene YourGene
   ```

## Data Preprocessing

All data (synthetic or real) undergoes the same preprocessing steps:

1. **Normalization**: Expression values scaled to [0,1] range
2. **Filtering**: Low-quality spots/genes can be filtered out
3. **Downsampling**: If data exceeds qubit capacity, strategic downsampling is performed
4. **Encoding**: Expression values encoded into quantum states or Hamiltonians

## Computational Considerations

When using real data, consider these factors:

1. **Qubit Limitations**: Current quantum simulators and hardware have limited qubit counts, so data size may need to be reduced
2. **Processing Time**: Larger datasets require more preprocessing and quantum circuit complexity
3. **Memory Requirements**: Statevector simulators have exponential memory requirements with qubit count
4. **Shot Count**: For measurement-based simulators and real hardware, more shots will give more accurate results

## Future Extensions

The framework is being extended to:

1. Support larger, real-world spatial transcriptomics datasets more efficiently
2. Add automatic preprocessing for common data formats (10x Visium, Slide-seq, etc.)
3. Implement multi-gene analysis capabilities
4. Optimize for specific quantum hardware backends

## Conclusion

The Qspat framework uses real quantum algorithms but is typically demonstrated and benchmarked with synthetic data for reproducibility and controlled testing. The framework is fully capable of processing real spatial transcriptomics data with the appropriate formatting and sizing considerations.