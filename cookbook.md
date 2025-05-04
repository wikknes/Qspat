# Qspat Cookbook

This cookbook provides practical examples, tutorials, and best practices for using the Quantum Spatial Transcriptomics Framework (Qspat) to analyze spatial transcriptomics data.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Core Modules](#core-modules)
5. [VQE Analysis: Finding Expression Maxima](#vqe-analysis-finding-expression-maxima)
6. [QAOA Analysis: Identifying Expression Regions](#qaoa-analysis-identifying-expression-regions)
7. [Visualization Techniques](#visualization-techniques)
8. [Advanced Customization](#advanced-customization)
9. [Tips and Best Practices](#tips-and-best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

Qspat is a hybrid framework that combines spatial transcriptomics with quantum computing algorithms. It uses variational quantum algorithms (VQE and QAOA) to:

1. Find coordinates with maximum expression of specific genes
2. Identify regions of high gene expression
3. Visualize expression patterns and quantum predictions

The framework leverages quantum computing capabilities to potentially provide computational advantages for these tasks, especially as datasets grow in size and complexity.

## Installation

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib, Seaborn
- Qiskit (or PennyLane)
- Scanpy (optional)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/qspat.git
cd qspat

# Install required packages
pip install numpy pandas scipy matplotlib seaborn
pip install qiskit qiskit-aer
pip install scanpy

# Optional: Create a virtual environment
python -m venv qspat-env
source qspat-env/bin/activate  # Linux/macOS
qspat-env\Scripts\activate     # Windows
```

## Data Preparation

Qspat works with various spatial transcriptomics data formats including:

- CSV files with gene expression matrices
- Anndata (.h5ad) files from Scanpy
- Separate expression and coordinate files

### Data Format Requirements

Your expression data should have:
- Rows representing spatial spots or cells
- Columns representing genes
- If coordinates are included in the same file, columns named 'x' and 'y'

### Sample Data Generation

Qspat includes a utility to generate synthetic data for testing:

```bash
# Generate synthetic data
python data/generate_synthetic_data.py --spots 100 --genes 20 --output data/synthetic_data.csv
```

### Preprocessing Pipeline

The framework includes a comprehensive preprocessing pipeline:

1. **Loading data**: Supports CSV and H5AD formats
2. **Filtering**: Removes low-quality spots and low-expression genes
3. **Normalization**: Options include log1p, CPM, and CPM+log1p
4. **Feature selection**: Selects genes by variance, PCA, or specific list
5. **Downsampling**: Reduces spots to fit qubit budget
6. **Quantum preparation**: Normalizes values and prepares for quantum encoding

Example preprocessing:

```python
from src.preprocess import process_data

# Process data for a specific gene
expr_vector, coord_map, n_qubits, index_map = process_data(
    'data/my_dataset.csv',
    target_gene='MyGene',
    max_spots=16,
    normalize_method='log1p'
)
```

## Core Modules

Qspat includes several core modules that handle different aspects of the analysis:

### preprocess.py

Responsible for data loading, filtering, normalization, and preparation:

```python
# Load data directly
from src.preprocess import load_data
df = load_data('data/expression.csv', 'data/coordinates.csv')

# Filter low-quality data
from src.preprocess import filter_data
filtered_df = filter_data(df, min_spots=5, min_expr=0.1)

# Normalize expression values
from src.preprocess import normalize_data
norm_df = normalize_data(filtered_df, method='log1p')
```

### encoding.py

Provides methods to encode classical data into quantum states:

- **Amplitude encoding**: Encodes data into quantum state amplitudes
- **Angle encoding**: Encodes data as rotation angles
- **Binary encoding**: Encodes binary data with X gates
- **Hamiltonian encoding**: Creates Hamiltonians for optimization

```python
from src.encoding import angle_encoding, hamiltonian_encoding
import numpy as np

# Encode a vector as rotation angles
data = np.array([0.2, 0.5, 0.8, 0.3])
circuit = angle_encoding(data)
print(circuit)

# Create a Hamiltonian for VQE
hamiltonian = hamiltonian_encoding(data)
```

### circuits.py

Contains functions to create quantum circuits for VQE and QAOA:

```python
from src.circuits import create_vqe_ansatz, add_measurements

# Create a VQE ansatz circuit
n_qubits = 4
ansatz, parameters = create_vqe_ansatz(n_qubits, depth=2)
print(ansatz)

# Add measurement operations
measured_circuit = add_measurements(ansatz)
```

### run_vqe.py and run_qaoa.py

These modules implement the VQE and QAOA algorithms specifically for spatial transcriptomics data.

## VQE Analysis: Finding Expression Maxima

VQE (Variational Quantum Eigensolver) is used to find the spot with maximum gene expression.

### Basic VQE Analysis

```python
from src.run_vqe import SpatialVQE

# Create VQE solver
vqe_solver = SpatialVQE(
    'data/expression_data.csv',
    target_gene='GeneA',
    max_spots=16,
    optimizer='cobyla'
)

# Run the complete analysis
vqe_solver.preprocess()  # Preprocess the data
vqe_solver.setup()       # Set up the VQE components
result = vqe_solver.run()  # Run the optimization

# Analyze results to find maximum expression location
max_idx, max_coords, probabilities = vqe_solver.analyze_results()
print(f"Maximum expression at: ({max_coords[0]:.2f}, {max_coords[1]:.2f})")

# Visualize the result
vqe_solver.visualize(save_path='output/max_expression.png')
```

### VQE Configuration Options

The VQE solver can be customized with several parameters:

- **optimizer**: 'cobyla', 'spsa', or 'slsqp' (different optimization algorithms)
- **ansatz_depth**: Controls the depth of the quantum circuit (default: 2)
- **shots**: Number of measurements in quantum simulation (default: 1024)
- **backend**: Custom Qiskit backend (default: statevector_simulator)

```python
# Advanced VQE configuration
from qiskit import Aer

# Use a custom backend and optimizer with more circuit depth
custom_backend = Aer.get_backend('qasm_simulator')
vqe_solver = SpatialVQE(
    'data/expression_data.csv',
    target_gene='GeneA',
    max_spots=8,
    backend=custom_backend,
    optimizer='spsa',
    ansatz_depth=3,
    shots=4096
)
```

### Running VQE from Command Line

The framework provides a command-line interface for VQE analysis:

```bash
python src/run_vqe.py --expression data/expression_data.csv --gene GeneA --max_spots 16 --optimizer cobyla --depth 2 --output results/vqe_result.png
```

## QAOA Analysis: Identifying Expression Regions

QAOA (Quantum Approximate Optimization Algorithm) is used to identify regions of high gene expression.

### Basic QAOA Analysis

```python
from src.run_qaoa import SpatialQAOA

# Create QAOA solver
qaoa_solver = SpatialQAOA(
    'data/expression_data.csv',
    target_gene='GeneA',
    max_spots=16,
    p_steps=1,
    alpha=1.0,
    beta=0.1
)

# Run the complete analysis
qaoa_solver.preprocess()  # Preprocess the data
qaoa_solver.setup()       # Set up the QAOA components
result = qaoa_solver.run()  # Run the optimization

# Analyze results to find high expression region
region_mask, region_indices, probabilities = qaoa_solver.analyze_results(threshold=0.4)
print(f"Found high expression region with {sum(region_mask)} spots")

# Visualize the result
qaoa_solver.visualize(region_mask, save_path='output/expression_region.png')
```

### QAOA Configuration Options

The QAOA solver can be customized with several parameters:

- **p_steps**: Number of QAOA layers (default: 1)
- **alpha**: Weight for expression terms in Hamiltonian (default: 1.0)
- **beta**: Weight for spatial correlation terms (default: 0.1)
- **threshold**: Probability threshold for region detection (default: 0.5)
- **distance_scale**: Scaling factor for spatial proximity (default: 1.0)

```python
# Advanced QAOA configuration
qaoa_solver = SpatialQAOA(
    'data/expression_data.csv',
    target_gene='GeneA',
    max_spots=16,
    optimizer='spsa',
    p_steps=2,
    alpha=2.0,
    beta=0.5,
    distance_scale=0.8
)
```

### Running QAOA from Command Line

The framework provides a command-line interface for QAOA analysis:

```bash
python src/run_qaoa.py --expression data/expression_data.csv --gene GeneA --max_spots 16 --optimizer cobyla --p 1 --alpha 1.0 --beta 0.1 --threshold 0.4 --output results/qaoa_result.png
```

## Visualization Techniques

Qspat provides built-in visualization capabilities:

### VQE Visualization

VQE visualization shows the probability distribution of maximum expression:

```python
# Generate and customize VQE visualization
import matplotlib.pyplot as plt

fig = vqe_solver.visualize(show=False)
ax = fig.gca()

# Customize the plot
ax.set_title('Custom VQE Visualization', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)

# Save and show
plt.tight_layout()
plt.savefig('custom_vqe_vis.png', dpi=300)
plt.show()
```

### QAOA Visualization

QAOA visualization shows regions of high expression:

```python
# Generate and customize QAOA visualization
fig = qaoa_solver.visualize(region_mask, show=False)
ax = fig.gca()

# Add background tissue image (if available)
from matplotlib.image import imread
tissue_img = imread('tissue_background.png')
ax.imshow(tissue_img, extent=[min(xs), max(xs), min(ys), max(ys)], 
          alpha=0.3, zorder=-1)

# Customize the plot
ax.set_title('Custom QAOA Region Visualization', fontsize=14)

# Save and show
plt.tight_layout()
plt.savefig('custom_qaoa_vis.png', dpi=300)
plt.show()
```

### Custom Visualizations

You can create fully custom visualizations using the raw results:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Get data from QAOA solver
region_mask, _, probabilities = qaoa_solver.analyze_results(threshold=0.3)
xs = qaoa_solver.coord_map[:, 0]
ys = qaoa_solver.coord_map[:, 1]
expr_values = qaoa_solver.expr_vector

# Create custom colormap
colors = [(0.8, 0.8, 0.8), (1, 0.8, 0), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all spots
scatter = ax.scatter(xs, ys, c=expr_values, cmap=cmap, 
                   s=100, alpha=0.7, edgecolors='k')

# Highlight region
region_xs = xs[region_mask]
region_ys = ys[region_mask]
ax.scatter(region_xs, region_ys, facecolors='none', 
         edgecolors='blue', s=150, linewidths=2, label='Region')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Expression Value')

# Add labels and title
ax.set_title('Custom Expression Map with QAOA Region')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.legend()

plt.tight_layout()
plt.savefig('custom_visualization.png', dpi=300)
plt.show()
```

## Advanced Customization

### Custom Quantum Circuits

You can customize the quantum circuits used in the framework:

```python
from qiskit import QuantumCircuit
from src.circuits import create_vqe_ansatz

# Create a custom VQE ansatz function
def my_custom_ansatz(n_qubits, depth=2):
    """Create a custom ansatz with specific gate pattern."""
    from qiskit.circuit import ParameterVector
    
    # Create parameters
    num_params = n_qubits * depth * 2
    params = ParameterVector('θ', num_params)
    
    # Create circuit
    circuit = QuantumCircuit(n_qubits)
    
    # Add initial Hadamard gates
    for q in range(n_qubits):
        circuit.h(q)
    
    # Add parameterized layers
    param_idx = 0
    for d in range(depth):
        # Add RY rotations
        for q in range(n_qubits):
            circuit.ry(params[param_idx], q)
            param_idx += 1
        
        # Add RZ rotations
        for q in range(n_qubits):
            circuit.rz(params[param_idx], q)
            param_idx += 1
        
        # Add entangling gates (CX)
        for q in range(n_qubits-1):
            circuit.cx(q, q+1)
        
        # Connect last qubit to first qubit
        if n_qubits > 2:
            circuit.cx(n_qubits-1, 0)
    
    return circuit, params

# Use this custom ansatz in the VQE solver
from src.run_vqe import SpatialVQE

# Create VQE solver
vqe_solver = SpatialVQE(
    'data/expression_data.csv',
    target_gene='GeneA',
    max_spots=16
)

# Preprocess the data
vqe_solver.preprocess()

# Replace the default ansatz with custom one
vqe_solver.ansatz, _ = my_custom_ansatz(vqe_solver.n_qubits, depth=2)

# Continue with setup and run
vqe_solver.setup()
result = vqe_solver.run()
```

### Custom Data Processing

You can customize the preprocessing pipeline:

```python
import pandas as pd
import numpy as np
from src.preprocess import normalize_data, select_features
from src.run_vqe import SpatialVQE

# Load and preprocess data manually
df = pd.read_csv('data/expression_data.csv', index_col=0)

# Add custom normalization steps
df_scaled = df.copy()
expr_cols = [col for col in df.columns if col not in ['x', 'y']]
df_scaled[expr_cols] = df_scaled[expr_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Apply standard normalization
df_norm = normalize_data(df_scaled, method='log1p')

# Custom feature selection
df_selected, _ = select_features(df_norm, n_features=5, method='var')

# Initialize VQE solver without preprocessing
vqe_solver = SpatialVQE(
    'data/expression_data.csv',  # This file path won't be used
    target_gene='GeneA',
    max_spots=16
)

# Manually set the preprocessed data
from src.preprocess import prepare_for_quantum
expr_vector, coord_map, n_qubits = prepare_for_quantum(
    df_selected, target_gene='GeneA'
)

vqe_solver.expr_vector = expr_vector
vqe_solver.coord_map = coord_map
vqe_solver.n_qubits = n_qubits

# Continue with standard workflow
vqe_solver.setup()
result = vqe_solver.run()
```

### Integrating with Other Quantum Frameworks

While Qspat uses Qiskit by default, you can adapt it to use other quantum frameworks:

```python
# Example of adapting to PennyLane
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires=4)

# Define a VQE cost function using PennyLane
@qml.qnode(dev)
def cost_function(params, hamiltonian_coeffs, hamiltonian_ops):
    # Prepare initial state
    for i in range(4):
        qml.Hadamard(wires=i)
    
    # Apply parameterized gates
    param_idx = 0
    for d in range(2):  # depth=2
        for i in range(4):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
        
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
    
    # Return expectation value of the Hamiltonian
    return qml.expval(qml.Hermitian(hamiltonian_coeffs, wires=range(4)))

# Convert Qiskit Hamiltonian to PennyLane format
# (This is a simplified example - actual conversion would be more complex)
def convert_hamiltonian_to_pennylane(qiskit_hamiltonian):
    # Extract coefficients and operators in PennyLane-compatible format
    # ...
    return hamiltonian_coeffs, hamiltonian_ops

# Use in optimization
from scipy.optimize import minimize

# Get data and Hamiltonian from Qspat
from src.preprocess import process_data
from src.encoding import hamiltonian_encoding

expr_vector, coord_map, n_qubits, _ = process_data(
    'data/expression_data.csv',
    target_gene='GeneA',
    max_spots=4  # Using 4 qubits for this example
)

hamiltonian = hamiltonian_encoding(expr_vector)
pl_hamiltonian_coeffs, pl_hamiltonian_ops = convert_hamiltonian_to_pennylane(hamiltonian)

# Initialize random parameters
initial_params = np.random.uniform(0, 2*np.pi, size=8)  # 4 qubits * 2 layers

# Run optimization
result = minimize(
    lambda params: cost_function(params, pl_hamiltonian_coeffs, pl_hamiltonian_ops),
    initial_params,
    method='COBYLA'
)

# Process optimization results
optimal_params = result.x
optimal_value = result.fun
```

## Tips and Best Practices

### Performance Optimization

1. **Data Preparation**:
   - Normalize data appropriately (log1p is often good for RNA-seq data)
   - Filter out low-quality spots and low-expression genes
   - Downsample to fit qubit budget (16-20 qubits is practical)

2. **Quantum Algorithm Settings**:
   - Start with simpler circuits (depth=1 or 2) and increase complexity if needed
   - For VQE, COBYLA optimizer usually works well for smaller problems
   - For QAOA, start with p=1 and increase if needed
   - Adjust alpha/beta parameters in QAOA to balance expression vs. spatial coherence

3. **Resource Management**:
   - Use statevector simulators for smaller problems (< 20 qubits)
   - Use shot-based simulators to approximate quantum hardware behavior
   - Balance shots vs. accuracy tradeoff (more shots = more accurate but slower)

### Best Practices for Analysis

1. **Gene Selection**:
   - Focus on genes with known biological significance
   - Look for genes with spatially variable expression
   - Try multiple normalization methods to ensure robustness

2. **Validation**:
   - Compare quantum results with classical algorithms
   - Validate findings against biological knowledge
   - Run multiple trials with different initializations

3. **Visualization**:
   - Always visualize results in spatial context
   - Use probability thresholds appropriate for your data
   - Consider overlaying results on histological images when available

## Troubleshooting

### Common Issues and Solutions

1. **Installation Problems**:
   ```
   Error: No module named 'qiskit'
   ```
   Solution: Install Qiskit using `pip install qiskit qiskit-aer`

2. **Data Loading Issues**:
   ```
   Error: Coordinates not found in data
   ```
   Solution: Ensure your data has 'x' and 'y' columns or provide a separate coordinates file

3. **Memory Errors**:
   ```
   MemoryError during quantum simulation
   ```
   Solution: Reduce the number of qubits by downsampling more aggressively

4. **Slow Performance**:
   ```
   Warning: VQE optimization taking too long
   ```
   Solution: Reduce circuit depth, use a different optimizer, or reduce max_spots

5. **Visualization Issues**:
   ```
   ValueError: Coordinates don't match data dimensions
   ```
   Solution: Ensure coordinate mapping is correctly aligned with expression data

### Debugging Tools

1. **Enable Logging**:
   ```python
   import logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

2. **Inspect Intermediate Results**:
   ```python
   # Check preprocessing output
   print(f"Expression vector: {expr_vector[:5]}...")
   print(f"Shape: {expr_vector.shape}, Min: {expr_vector.min()}, Max: {expr_vector.max()}")
   
   # Inspect quantum circuit
   print(vqe_solver.ansatz)
   
   # Check optimization trajectory
   import matplotlib.pyplot as plt
   plt.figure()
   plt.plot(vqe_solver.result.cost_function_evals)
   plt.xlabel('Iteration')
   plt.ylabel('Cost')
   plt.title('VQE Optimization Trajectory')
   plt.savefig('vqe_optimization.png')
   ```

3. **Test with Simplified Data**:
   ```python
   # Create simple test data
   import numpy as np
   import pandas as pd
   
   # Create a 4x4 grid with a clear maximum
   x = np.linspace(0, 3, 4)
   y = np.linspace(0, 3, 4)
   X, Y = np.meshgrid(x, y)
   
   # Create expression values with peak at (2,2)
   Z = np.exp(-((X-2)**2 + (Y-2)**2))
   
   # Create dataframe
   test_data = []
   for i in range(4):
       for j in range(4):
           test_data.append({
               'x': x[j],
               'y': y[i],
               'TestGene': Z[i, j]
           })
   
   df = pd.DataFrame(test_data)
   df.to_csv('test_simple.csv', index=False)
   
   # Run VQE on test data
   vqe_solver = SpatialVQE(
       'test_simple.csv',
       target_gene='TestGene',
       max_spots=16
   )
   
   vqe_solver.preprocess()
   vqe_solver.setup()
   result = vqe_solver.run()
   
   # Check if maximum is correctly identified
   max_idx, max_coords, _ = vqe_solver.analyze_results()
   print(f"Found maximum at: ({max_coords[0]:.1f}, {max_coords[1]:.1f})")
   # Should be close to (2.0, 2.0)
   ```

---

By following this cookbook, you should be able to effectively use the Qspat framework to analyze spatial transcriptomics data using quantum computing algorithms. For more detailed information, refer to the source code and example notebooks provided in the repository.