#!/usr/bin/env python
# Example script to run VQE on synthetic data

import sys
import os

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add parent directory to path
sys.path.append(current_dir)

from src.run_vqe import SpatialVQE

# Path to our synthetic data
DATA_PATH = "data/synthetic_data.csv"

# Create VQE solver for Gene2 (which has a central peak pattern)
vqe_solver = SpatialVQE(
    DATA_PATH,
    target_gene='Gene2',
    max_spots=16,
    optimizer='cobyla',
    ansatz_depth=2,
    shots=1024
)

print("Starting VQE preprocessing...")
vqe_solver.preprocess()

print("Setting up VQE components...")
vqe_solver.setup()

print("Running VQE optimization...")
result = vqe_solver.run()

print("\nAnalyzing results...")
max_idx, max_coords, probabilities = vqe_solver.analyze_results()
print(f"Maximum expression found at index {max_idx}")
if max_coords is not None:
    print(f"Coordinates: ({max_coords[0]:.2f}, {max_coords[1]:.2f})")

print("\nCreating visualization...")
vqe_solver.visualize(save_path='vqe_result.png')
print("Visualization saved to: vqe_result.png")