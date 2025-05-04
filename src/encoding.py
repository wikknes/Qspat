#!/usr/bin/env python
# Quantum data encoding for the spatial transcriptomics framework

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from qiskit.circuit import ParameterVector

def amplitude_encoding(data_vector, n_qubits=None):
    """
    Encode data vector into quantum state amplitudes.
    
    Args:
        data_vector: Normalized vector to encode (must be normalized to 1)
        n_qubits: Number of qubits to use (derived from vector length if None)
        
    Returns:
        QuantumCircuit with initialized state
    """
    if n_qubits is None:
        n_qubits = int(np.ceil(np.log2(len(data_vector))))
    
    # Ensure vector is normalized
    norm = np.linalg.norm(data_vector)
    if abs(norm - 1.0) > 1e-6:
        data_vector = data_vector / norm
    
    # Pad to power of 2 if needed
    target_len = 2**n_qubits
    if len(data_vector) < target_len:
        data_vector = np.pad(data_vector, (0, target_len - len(data_vector)))
    
    # Initialize circuit with the state
    circuit = QuantumCircuit(n_qubits)
    init_gate = Initialize(data_vector)
    circuit.append(init_gate, range(n_qubits))
    
    return circuit

def angle_encoding(data_vector):
    """
    Encode data vector into rotation angles (RY gates).
    
    Args:
        data_vector: Vector to encode (will be scaled to [0,π])
        
    Returns:
        QuantumCircuit with parameterized rotations
    """
    n_qubits = len(data_vector)
    
    # Scale to [0,π]
    min_val = min(data_vector)
    max_val = max(data_vector)
    if max_val > min_val:  # Avoid division by zero
        scaled_data = [np.pi * (x - min_val) / (max_val - min_val) for x in data_vector]
    else:
        scaled_data = [0] * n_qubits
    
    # Create circuit with RY rotations
    circuit = QuantumCircuit(n_qubits)
    for i, angle in enumerate(scaled_data):
        circuit.ry(angle, i)
    
    return circuit

def angle_encoding_parameterized(n_qubits):
    """
    Create parameterized circuit for angle encoding.
    
    Args:
        n_qubits: Number of qubits/parameters
        
    Returns:
        QuantumCircuit with parameters and parameter mapping function
    """
    # Create parameters
    params = ParameterVector('theta', n_qubits)
    
    # Create circuit
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.ry(params[i], i)
    
    # Create binding function
    def bind_parameters(data_vector):
        # Scale to [0,π]
        min_val = min(data_vector)
        max_val = max(data_vector)
        if max_val > min_val:  # Avoid division by zero
            scaled_data = [np.pi * (x - min_val) / (max_val - min_val) for x in data_vector]
        else:
            scaled_data = [0] * n_qubits
            
        # Create parameter dictionary
        param_dict = {params[i]: scaled_data[i] for i in range(n_qubits)}
        return circuit.bind_parameters(param_dict)
    
    return circuit, bind_parameters

def binary_encoding(data_vector, threshold=0.5):
    """
    Encode binary data vector (apply X gate for 1s).
    
    Args:
        data_vector: Vector to encode (will be thresholded)
        threshold: Value above which to encode as 1
        
    Returns:
        QuantumCircuit with X gates
    """
    n_qubits = len(data_vector)
    
    # Binarize data
    binary_data = [1 if x >= threshold else 0 for x in data_vector]
    
    # Create circuit
    circuit = QuantumCircuit(n_qubits)
    for i, bit in enumerate(binary_data):
        if bit == 1:
            circuit.x(i)
    
    return circuit

def hamiltonian_encoding(data_vector):
    """
    Create Hamiltonian operator for data vector.
    For Qiskit, returns a SparsePauliOp or WeightedPauliOperator.
    
    Args:
        data_vector: Vector of weights (expression values)
        
    Returns:
        Hamiltonian operator object
    """
    try:
        # Try import SparsePauliOp (Qiskit 0.20+)
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.quantum_info import Pauli
        
        n_qubits = int(np.ceil(np.log2(len(data_vector))))
        
        # Construct Hamiltonian for maximizing expression
        # H = -sum_i data[i]|i⟩⟨i|
        paulis = []
        coeffs = []
        
        for i, val in enumerate(data_vector):
            if abs(val) < 1e-10:  # Skip near-zero values
                continue
                
            # Convert i to binary string and create Z-string 
            binary = format(i, f'0{n_qubits}b')
            z_string = ''
            
            for bit in binary:
                if bit == '0':
                    z_string += 'Z'  # |0⟩⟨0| = (I+Z)/2
                else:
                    z_string += 'Z'  # |1⟩⟨1| = (I-Z)/2
            
            # Create full Pauli string with identities
            pauli_str = 'I' * n_qubits
            
            # Project onto the specific state
            proj_coeff = -val  # Negative for minimization
            paulis.append(Pauli(z_string))
            coeffs.append(proj_coeff)
        
        # Create the Hamiltonian operator
        hamiltonian = SparsePauliOp(paulis, coeffs)
        return hamiltonian
        
    except ImportError:
        # Fallback for older Qiskit
        from qiskit.quantum_info import Pauli
        from qiskit.opflow import PauliOp, PauliSumOp
        
        n_qubits = int(np.ceil(np.log2(len(data_vector))))
        hamiltonian = None
        
        for i, val in enumerate(data_vector):
            if abs(val) < 1e-10:  # Skip near-zero values
                continue
                
            # Convert i to binary string and create Z-string 
            binary = format(i, f'0{n_qubits}b')
            z_string = ''
            
            for bit in binary:
                if bit == '0':
                    z_string += 'Z'
                else:
                    z_string += 'Z'
            
            # Create Pauli operator
            pauli = PauliOp(Pauli(z_string), -val)
            
            if hamiltonian is None:
                hamiltonian = pauli
            else:
                hamiltonian += pauli
        
        return hamiltonian

def hamiltonian_for_region_detection(data_vector, spatial_weights=None, alpha=1.0, beta=0.1):
    """
    Create Hamiltonian for detecting regions of high expression.
    Includes spatial proximity weights to encourage contiguous regions.
    
    Args:
        data_vector: Vector of expression values
        spatial_weights: Matrix of spatial proximity weights between spots
        alpha: Weight for expression terms
        beta: Weight for spatial correlation terms
        
    Returns:
        Hamiltonian operator object
    """
    try:
        # Try import SparsePauliOp (Qiskit 0.20+)
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.quantum_info import Pauli
        
        n_qubits = len(data_vector)
        
        # Initialize empty lists for Paulis and coefficients
        paulis = []
        coeffs = []
        
        # Add expression terms: -alpha * sum_i data[i] * Z_i
        for i, val in enumerate(data_vector):
            if abs(val) < 1e-10:  # Skip near-zero values
                continue
                
            # Create Z operator for qubit i
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli = ''.join(pauli_str)
            
            # Add to Hamiltonian with weight proportional to expression
            paulis.append(Pauli(pauli))
            coeffs.append(-alpha * val)  # Negative for minimization
        
        # Add spatial correlation terms if provided
        if spatial_weights is not None:
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    weight = spatial_weights[i, j]
                    if abs(weight) < 1e-10:  # Skip near-zero weights
                        continue
                    
                    # Create ZZ operator between qubits i and j
                    pauli_str = ['I'] * n_qubits
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli = ''.join(pauli_str)
                    
                    # Add to Hamiltonian with weight proportional to spatial proximity
                    paulis.append(Pauli(pauli))
                    coeffs.append(-beta * weight)  # Negative to encourage correlation
        
        # Create the Hamiltonian operator
        return SparsePauliOp(paulis, coeffs)
        
    except ImportError:
        # Fallback for older Qiskit
        from qiskit.quantum_info import Pauli
        from qiskit.opflow import PauliOp, PauliSumOp
        
        n_qubits = len(data_vector)
        hamiltonian = None
        
        # Add expression terms
        for i, val in enumerate(data_vector):
            if abs(val) < 1e-10:
                continue
                
            # Create Pauli Z string
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli = ''.join(pauli_str)
            
            # Create operator
            term = PauliOp(Pauli(pauli), -alpha * val)
            
            if hamiltonian is None:
                hamiltonian = term
            else:
                hamiltonian += term
        
        # Add spatial correlation terms
        if spatial_weights is not None:
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    weight = spatial_weights[i, j]
                    if abs(weight) < 1e-10:
                        continue
                    
                    # Create Pauli ZZ string
                    pauli_str = ['I'] * n_qubits
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli = ''.join(pauli_str)
                    
                    # Create operator
                    term = PauliOp(Pauli(pauli), -beta * weight)
                    hamiltonian += term
        
        return hamiltonian

if __name__ == "__main__":
    # Simple test and example
    test_data = np.array([0.1, 0.5, 0.2, 0.7])
    
    # Normalize for amplitude encoding
    test_data_norm = test_data / np.linalg.norm(test_data)
    
    # Test amplitude encoding
    circuit_amp = amplitude_encoding(test_data_norm)
    print("Amplitude encoding circuit:")
    print(circuit_amp)
    
    # Test angle encoding
    circuit_angle = angle_encoding(test_data)
    print("\nAngle encoding circuit:")
    print(circuit_angle)
    
    # Test binary encoding
    circuit_bin = binary_encoding(test_data, threshold=0.3)
    print("\nBinary encoding circuit:")
    print(circuit_bin)
    
    # Test Hamiltonian encoding
    try:
        hamiltonian = hamiltonian_encoding(test_data)
        print("\nHamiltonian representation:")
        print(hamiltonian)