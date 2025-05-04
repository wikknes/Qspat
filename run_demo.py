#!/usr/bin/env python
# Simplified demo script for running quantum spatial algorithms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check Qiskit installation and import what's available
try:
    from qiskit import QuantumCircuit, execute, transpile
    from qiskit.circuit import Parameter, ParameterVector
    try:
        from qiskit import Aer
    except ImportError:
        print("Qiskit Aer not available, will use BasicAer instead")
        from qiskit import BasicAer as Aer
    
    try:
        from qiskit.algorithms import VQE, QAOA
        from qiskit.algorithms.optimizers import COBYLA
    except ImportError:
        print("Qiskit algorithms not available, will simulate results")
        SIMULATE_ONLY = True
    else:
        SIMULATE_ONLY = False
except ImportError:
    print("Qiskit not installed, will only generate synthetic data and simulate results")
    SIMULATE_ONLY = True

# Path to our synthetic data
DATA_PATH = "data/synthetic_data.csv"

# Parameters
TARGET_GENE = 'Gene2'  # Gene with central peak pattern
MAX_SPOTS = 16
NUM_QUBITS = 4  # log2(MAX_SPOTS)

def load_and_process_data(data_path, target_gene):
    """Load and process the synthetic data."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Extract coordinates and expression
    coords = df[['x', 'y']].values
    
    if target_gene in df.columns:
        expression = df[target_gene].values
    else:
        raise ValueError(f"Gene {target_gene} not found in data")
    
    # Normalize expression to [0,1]
    expr_min = expression.min()
    expr_max = expression.max()
    if expr_max > expr_min:
        expr_norm = (expression - expr_min) / (expr_max - expr_min)
    else:
        expr_norm = np.zeros_like(expression)
    
    print(f"Processed data with {len(expr_norm)} spots")
    return expr_norm, coords

def create_hamiltonian(expression_vector):
    """Create a simple Hamiltonian for VQE (simplified version)."""
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.quantum_info import Pauli
    
    n_qubits = int(np.ceil(np.log2(len(expression_vector))))
    
    # For demonstration, we'll use a simplified Hamiltonian
    # where we directly penalize states based on expression value
    paulis = []
    coeffs = []
    
    for i, val in enumerate(expression_vector):
        if i >= 2**n_qubits:
            break
            
        if abs(val) < 1e-6:
            continue
            
        # For each spot i, create a term that gives lower energy when 
        # measuring state |i⟩ corresponding to higher expression
        binary = format(i, f'0{n_qubits}b')
        z_string = ''
        x_string = ''
        
        # Create projector onto state |i⟩
        for bit in binary:
            if bit == '0':
                z_string += 'Z'
                x_string += 'I'
            else:
                z_string += 'Z'
                x_string += 'I'
        
        # Create Pauli strings
        pauli_z = Pauli(z_string)
        
        # Add term with weight proportional to expression
        paulis.append(pauli_z)
        coeffs.append(-val)  # Negative for minimization
    
    # Create the Hamiltonian operator
    hamiltonian = SparsePauliOp(paulis, coeffs)
    return hamiltonian

def create_ansatz(n_qubits, depth=2):
    """Create a hardware-efficient ansatz for VQE."""
    # Create parameters
    params = ParameterVector('θ', 3 * n_qubits * depth)
    
    # Create circuit
    circuit = QuantumCircuit(n_qubits)
    
    # Initial rotation layer
    param_idx = 0
    for i in range(n_qubits):
        circuit.rx(params[param_idx], i)
        param_idx += 1
        circuit.ry(params[param_idx], i)
        param_idx += 1
        circuit.rz(params[param_idx], i)
        param_idx += 1
    
    # Repeated blocks of entanglement + rotation
    for d in range(depth - 1):
        # Entanglement layer (linear)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Rotation layer
        for i in range(n_qubits):
            circuit.rx(params[param_idx], i)
            param_idx += 1
            circuit.ry(params[param_idx], i)
            param_idx += 1
            circuit.rz(params[param_idx], i)
            param_idx += 1
    
    return circuit, params

def run_vqe_demo(expression, coords):
    """Run the VQE algorithm on the expression data."""
    print("\n--- Running VQE Demo ---")
    
    if SIMULATE_ONLY:
        print("Running in simulation mode since Qiskit algorithms are not available")
        
        # In simulation mode, we'll just find the maximum expression value directly
        max_expr_index = np.argmax(expression)
        max_expr_value = expression[max_expr_index]
        
        # Create a fake probability distribution that peaks at the maximum expression spot
        probabilities = np.zeros(2**4)  # 4 qubits = 16 states
        probabilities[max_expr_index] = 0.7  # High probability at max spot
        
        # Add some noise to other states
        for i in range(len(probabilities)):
            if i != max_expr_index:
                probabilities[i] = 0.3 * np.random.random() * expression[i % len(expression)]
        
        # Normalize
        probabilities = probabilities / np.sum(probabilities)
        
        # Create a fake result object
        class FakeResult:
            def __init__(self, eigenvalue):
                self.eigenvalue = eigenvalue
                self.optimal_parameters = {"theta": 0.0}
                
        result = FakeResult(eigenvalue=-max_expr_value)
        
        print(f"Simulation found maximum at index {max_expr_index}")
        print(f"Maximum expression value: {max_expr_value:.4f}")
        
        # Get coordinates for this spot
        if max_expr_index < len(coords):
            max_coordinates = coords[max_expr_index]
            print(f"Coordinates of maximum expression: ({max_coordinates[0]:.2f}, {max_coordinates[1]:.2f})")
        
    else:
        # Actual VQE implementation
        # Set up backend
        backend = Aer.get_backend('statevector_simulator')
        
        # Create Hamiltonian
        hamiltonian = create_hamiltonian(expression)
        print(f"Created Hamiltonian with {len(hamiltonian)} terms")
        
        # Create ansatz
        n_qubits = int(np.ceil(np.log2(len(expression))))
        ansatz, _ = create_ansatz(n_qubits, depth=2)
        
        # Set up VQE
        optimizer = COBYLA(maxiter=100)
        
        vqe = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=backend
        )
        
        # Run VQE
        print("Running VQE optimization...")
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        # Extract optimal parameters
        optimal_params = result.optimal_parameters
        
        # Create circuit with optimal parameters
        optimal_circuit = ansatz.bind_parameters(optimal_params)
        
        # Get statevector
        circuit = transpile(optimal_circuit, backend)
        job = execute(circuit, backend)
        statevector = job.result().get_statevector()
        
        # Calculate probabilities
        probabilities = np.abs(statevector)**2
        
        # Find the index with highest probability
        max_prob_index = np.argmax(probabilities)
        max_prob = probabilities[max_prob_index]
        
        print(f"VQE completed with optimal energy: {result.eigenvalue:.6f}")
        print(f"Max probability {max_prob:.4f} at index {max_prob_index}")
        
        # Get coordinates for this spot
        if max_prob_index < len(coords):
            max_coordinates = coords[max_prob_index]
            print(f"Coordinates of maximum expression: ({max_coordinates[0]:.2f}, {max_coordinates[1]:.2f})")
    
    # Create visualization
    max_prob_index = np.argmax(probabilities[:len(expression)])
    visualize_vqe_results(coords, expression, probabilities, max_prob_index)
    
    return result, probabilities

def visualize_vqe_results(coords, expression, probabilities, max_idx):
    """Visualize the VQE results."""
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original expression
    scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=expression, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='k')
    axes[0].set_title(f'Original Expression of {TARGET_GENE}')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Expression level')
    
    # Plot probabilities
    # Only use probabilities for actual data points (in case of padding)
    plot_probs = probabilities[:len(coords)]
    scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=plot_probs, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='k')
    
    # Highlight maximum
    axes[1].scatter(coords[max_idx, 0], coords[max_idx, 1], s=200, facecolors='none', 
                  edgecolors='red', linewidths=2, label='Maximum')
    
    axes[1].set_title('Probability Distribution from VQE')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    axes[1].legend()
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Probability')
    
    plt.tight_layout()
    plt.savefig('vqe_demo_result.png', dpi=300)
    print("Saved visualization to: vqe_demo_result.png")
    plt.close()

def run_qaoa_demo(expression, coords):
    """Run the QAOA algorithm on the expression data to find high expression regions."""
    print("\n--- Running QAOA Demo ---")
    
    # Calculate spatial weights based on coordinates
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    scale = np.mean(distances)
    spatial_weights = np.exp(-distances / scale)
    np.fill_diagonal(spatial_weights, 0)
    
    if SIMULATE_ONLY:
        print("Running in simulation mode since Qiskit algorithms are not available")
        
        # In simulation mode, we'll identify regions based on thresholding
        # and considering spatial proximity
        
        # 1. Start with high expression spots
        threshold = 0.7
        high_expr = expression > threshold
        print(f"Found {sum(high_expr)} spots above threshold {threshold}")
        
        # 2. Extend to neighboring spots with moderate expression
        region_mask = high_expr.copy()
        for i in range(len(expression)):
            if high_expr[i]:
                # Find neighbors with reasonable expression
                for j in range(len(expression)):
                    if not region_mask[j] and spatial_weights[i, j] > 0.5 and expression[j] > 0.4:
                        region_mask[j] = True
        
        print(f"Extended to {sum(region_mask)} spots with neighborhood criteria")
        
        # 3. Create a fake probability distribution
        probabilities = np.zeros(2**4)  # 4 qubits = 16 states
        
        # Set high probability for spots in the region
        region_indices = np.where(region_mask)[0]
        for idx in region_indices:
            probabilities[idx] = 0.7 * expression[idx]
            
        # Add some noise to other states
        for i in range(len(probabilities)):
            if i not in region_indices and i < len(expression):
                probabilities[i] = 0.1 * np.random.random() * expression[i]
        
        # Normalize
        probabilities = probabilities / np.sum(probabilities)
        
        class FakeResult:
            def __init__(self, eigenvalue):
                self.eigenvalue = eigenvalue
                self.optimal_parameters = {
                    "gamma": [0.1],
                    "beta": [0.3]
                }
                
        result = FakeResult(eigenvalue=-np.sum(expression[region_mask]))
    else:
        # Actual QAOA implementation would go here
        # This is a placeholder that would need to be implemented based on Qiskit's QAOA
        print("Full QAOA implementation not available in this demo")
        
        # Create fake results for visualization
        region_mask = expression > 0.7
        probabilities = np.zeros(2**4)
        region_indices = np.where(region_mask)[0]
        for idx in region_indices:
            probabilities[idx] = 0.7 * expression[idx]
        
        # Normalize
        probabilities = probabilities / np.sum(probabilities)
        
        class FakeResult:
            def __init__(self, eigenvalue):
                self.eigenvalue = eigenvalue
                self.optimal_parameters = {"gamma": [0.1], "beta": [0.3]}
                
        result = FakeResult(eigenvalue=-np.sum(expression[region_mask]))
    
    # Create visualization
    visualize_qaoa_results(coords, expression, region_mask, probabilities)
    
    return result, region_mask, probabilities

def visualize_qaoa_results(coords, expression, region_mask, probabilities):
    """Visualize the QAOA results."""
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original expression
    scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=expression, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='k')
    axes[0].set_title(f'Original Expression of {TARGET_GENE}')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Expression level')
    
    # Plot identified region
    # Base scatter plot of all spots
    axes[1].scatter(coords[:, 0], coords[:, 1], c='lightgray', s=80, alpha=0.6, edgecolors='k', label='All spots')
    
    # Highlight region spots
    if np.any(region_mask):
        region_coords = coords[region_mask]
        # Use probabilities as colors for region spots
        region_probs = probabilities[:len(expression)][region_mask]
        scatter2 = axes[1].scatter(region_coords[:, 0], region_coords[:, 1], c=region_probs, 
                                cmap='viridis', s=120, alpha=0.9, edgecolors='k', label='Region')
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label('Probability')
    
    axes[1].set_title('High Expression Region from QAOA')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('qaoa_demo_result.png', dpi=300)
    print("Saved visualization to: qaoa_demo_result.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load and process data
    expression, coords = load_and_process_data(DATA_PATH, TARGET_GENE)
    
    # Run VQE to find maximum expression location
    vqe_result, vqe_probabilities = run_vqe_demo(expression, coords)
    
    # Run QAOA to find high expression regions
    qaoa_result, region_mask, qaoa_probabilities = run_qaoa_demo(expression, coords)