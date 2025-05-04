# Quantum Spatial Transcriptomics Framework (Qspat)

A hybrid framework for analyzing spatial transcriptomics data using quantum computing to predict coordinates of gene expression hotspots.

## Overview

This framework uses variational quantum algorithms (VQE and QAOA) with Hamiltonian encoding to:

1. Find the coordinates with maximum expression of a specified gene
2. Identify regions of high gene expression
3. Visualize expression patterns and quantum predictions

## Features

- **Data preprocessing**: Normalize, filter, and prepare spatial transcriptomics data
- **Quantum encoding**: Multiple methods to encode data into quantum states 
- **VQE implementation**: Find maximum expression locations using Variational Quantum Eigensolver
- **QAOA implementation**: Identify high expression regions using Quantum Approximate Optimization Algorithm
- **Visualization**: Comprehensive tools for visualizing results

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib, Seaborn
- Qiskit (or PennyLane)
- Scanpy (optional)

```bash
pip install numpy pandas scipy matplotlib seaborn
pip install qiskit qiskit-aer
pip install scanpy
```

## Project Structure

```
Qspat/
├── data/
│   └── synthetic_data.csv
├── src/
│   ├── preprocess.py       # Data loading, filtering, normalization
│   ├── encoding.py         # Quantum data encoding methods
│   ├── circuits.py         # Quantum circuit construction
│   ├── run_vqe.py          # VQE implementation
│   ├── run_qaoa.py         # QAOA implementation
│   └── visualize.py        # Visualization utilities
└── notebooks/
    └── demo.ipynb          # End-to-end example
```

## Usage Examples

### Finding Maximum Expression Location

```python
from src.run_vqe import SpatialVQE

# Create VQE solver
vqe_solver = SpatialVQE(
    'data/my_expression_data.csv',
    target_gene='MyGene',
    max_spots=16,
    optimizer='cobyla'
)

# Run the algorithm
vqe_solver.preprocess()
vqe_solver.setup()
result = vqe_solver.run()

# Get the predicted maximum location
max_idx, max_coords, probabilities = vqe_solver.analyze_results()
print(f"Maximum expression at: ({max_coords[0]:.2f}, {max_coords[1]:.2f})")

# Visualize the result
vqe_solver.visualize(save_path='output/max_expression.png')
```

### Identifying High Expression Regions

```python
from src.run_qaoa import SpatialQAOA

# Create QAOA solver
qaoa_solver = SpatialQAOA(
    'data/my_expression_data.csv',
    target_gene='MyGene',
    max_spots=16,
    p_steps=1,
    alpha=1.0,
    beta=0.1
)

# Run the algorithm
qaoa_solver.preprocess()
qaoa_solver.setup()
result = qaoa_solver.run()

# Get the predicted region
region_mask, region_indices, probabilities = qaoa_solver.analyze_results(threshold=0.4)
print(f"Found high expression region with {sum(region_mask)} spots")

# Visualize the result
qaoa_solver.visualize(region_mask, save_path='output/expression_region.png')
```

## Get Started

Try the demo notebook for a complete walkthrough:

```bash
jupyter notebook notebooks/demo.ipynb
```

## Extending the Framework

- **Additional algorithms**: Implement other quantum algorithms (e.g., QSVMs)
- **Multi-gene analysis**: Analyze patterns across multiple genes simultaneously 
- **Real hardware execution**: Connect to IBM Quantum or other quantum hardware
- **Additional visualization**: Develop more specialized visualizations for biological interpretation

## References

This implementation is based on the theoretical foundations described in:
- app1.md: "Quantum Computing Framework for Spatial Transcriptomics"
- app2.md: "Quantum Spatial Transcriptomics Framework Specification"

## Citation

If you use this framework in your research, please cite:

```
@software{qspat,
  author = {Your Name},
  title = {Quantum Spatial Transcriptomics Framework (Qspat)},
  year = {2023},
  url = {https://github.com/yourusername/qspat}
}
```