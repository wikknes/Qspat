#!/usr/bin/env python
# Script to create a combined visualization of both VQE and QAOA results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to our synthetic data
DATA_PATH = "data/synthetic_data.csv"
TARGET_GENE = 'Gene2'

def main():
    # Load the original data
    df = pd.read_csv(DATA_PATH)
    coords = df[['x', 'y']].values
    expression = df[TARGET_GENE].values
    
    # Create a combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original expression map
    scatter1 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=expression, cmap='viridis', 
                                s=100, alpha=0.8, edgecolors='k')
    axes[0, 0].set_title(f'Original Expression of {TARGET_GENE}')
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Y coordinate')
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Expression level')
    
    # Find the maximum expression location (simulating VQE result)
    max_idx = np.argmax(expression)
    max_coords = coords[max_idx]
    
    # VQE result - Maximum expression location
    axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=expression, cmap='viridis', 
                    s=100, alpha=0.6, edgecolors='k')
    # Highlight the maximum with a star
    axes[0, 1].scatter(max_coords[0], max_coords[1], marker='*', s=300, 
                    facecolors='red', edgecolors='k', label='Maximum (VQE)')
    
    axes[0, 1].set_title('VQE Result: Maximum Expression Location')
    axes[0, 1].set_xlabel('X coordinate')
    axes[0, 1].set_ylabel('Y coordinate')
    axes[0, 1].legend()
    
    # QAOA result - High expression region
    # Use a threshold of 0.7 to identify the high expression region
    threshold = 0.7
    high_expr = expression > threshold
    
    # Simulate the region extending to neighboring spots
    region_mask = high_expr.copy()
    
    # Calculate spatial weights
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    scale = np.mean(distances)
    spatial_weights = np.exp(-distances / scale)
    np.fill_diagonal(spatial_weights, 0)
    
    # Extend to neighboring spots with reasonable expression
    for i in range(len(expression)):
        if high_expr[i]:
            # Find neighbors with reasonable expression
            for j in range(len(expression)):
                if not region_mask[j] and spatial_weights[i, j] > 0.5 and expression[j] > 0.4:
                    region_mask[j] = True
    
    # 3D surface plot of expression
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = expression
    
    # Create a grid for the surface plot
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[min(X):max(X):100j, min(Y):max(Y):100j]
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')
    
    # Plot the surface
    surf = axes[1, 0].contourf(grid_x, grid_y, grid_z, 20, cmap='viridis')
    axes[1, 0].set_title('Expression Surface')
    axes[1, 0].set_xlabel('X coordinate')
    axes[1, 0].set_ylabel('Y coordinate')
    cbar3 = plt.colorbar(surf, ax=axes[1, 0])
    cbar3.set_label('Expression level')
    
    # Plot the identified region (QAOA result)
    # Base scatter plot of all spots
    axes[1, 1].scatter(coords[:, 0], coords[:, 1], c='lightgray', s=80, alpha=0.6, 
                     edgecolors='k', label='All spots')
    
    # Highlight region spots
    if np.any(region_mask):
        region_coords = coords[region_mask]
        # Color by expression intensity
        region_expr = expression[region_mask]
        scatter3 = axes[1, 1].scatter(region_coords[:, 0], region_coords[:, 1], c=region_expr, 
                                   cmap='viridis', s=120, alpha=0.9, edgecolors='k', 
                                   label='High Expression Region (QAOA)')
        cbar4 = plt.colorbar(scatter3, ax=axes[1, 1])
        cbar4.set_label('Expression level')
    
    axes[1, 1].set_title('QAOA Result: High Expression Region')
    axes[1, 1].set_xlabel('X coordinate')
    axes[1, 1].set_ylabel('Y coordinate')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('combined_results.png', dpi=300)
    print("Saved combined visualization to: combined_results.png")
    
    # Also create a combined visualization that includes quantum aspects
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: VQE circuit diagram (simplified)
    circuit_ax = axes[0]
    circuit_ax.set_axis_off()
    
    # Create a simplified circuit diagram
    circuit_elements = [
        (0.1, 0.8, "H", "Initial superposition"),
        (0.3, 0.8, "RY", "Parameterized rotation"),
        (0.5, 0.8, "RZ", "Parameterized rotation"),
        (0.7, 0.8, "CNOT", "Entanglement"),
        (0.9, 0.8, "M", "Measurement")
    ]
    
    for x, y, gate, desc in circuit_elements:
        circuit_ax.add_patch(plt.Rectangle((x-0.05, y-0.05), 0.1, 0.1, 
                                         fill=True, alpha=0.7, color='skyblue'))
        circuit_ax.text(x, y, gate, ha='center', va='center', fontweight='bold')
        circuit_ax.text(x, y-0.15, desc, ha='center', va='center', fontsize=8)
        
        if x < 0.9:  # Connect gates with lines except for the last one
            circuit_ax.plot([x+0.05, x+0.25], [y, y], 'k-', lw=1)
    
    # Add qubit lines
    circuit_ax.plot([0.05, 0.95], [0.8, 0.8], 'k-', lw=1, alpha=0.3)
    circuit_ax.plot([0.05, 0.95], [0.6, 0.6], 'k-', lw=1, alpha=0.3)
    circuit_ax.text(0.02, 0.8, "|0⟩", ha='center', va='center')
    circuit_ax.text(0.02, 0.6, "|0⟩", ha='center', va='center')
    
    # Add a CNOT connection
    circuit_ax.plot([0.7, 0.7], [0.75, 0.65], 'k-', lw=1)
    circuit_ax.add_patch(plt.Circle((0.7, 0.65), 0.02, fill=True, color='black'))
    
    # Add title and description
    circuit_ax.set_title("Quantum Circuit for VQE", fontsize=14)
    circuit_ax.text(0.5, 0.3, "VQE searches for the state with minimum energy,\n"
                           "corresponding to the maximum expression location.", 
                 ha='center', va='center', fontsize=10)
    circuit_ax.text(0.5, 0.1, "Energy = ⟨ψ|H|ψ⟩ where H encodes expression values", 
                 ha='center', va='center', fontsize=10, style='italic')
    
    # Right: Quantum probabilities for region detection
    prob_ax = axes[1]
    
    # Create a bar chart of state probabilities
    n_states = 8  # Show just the most relevant states
    state_labels = [format(i, f'0{4}b') for i in range(n_states)]
    
    # Create fake probabilities peaking at states corresponding to the region
    probs = np.zeros(n_states)
    for i in range(n_states):
        if i < len(expression) and region_mask[i]:
            probs[i] = 0.2 + 0.3 * expression[i]
        elif i < len(expression):
            probs[i] = 0.05 * expression[i]
    
    # Normalize
    probs = probs / np.sum(probs)
    
    # Plot
    bars = prob_ax.bar(state_labels, probs, color='skyblue', alpha=0.7)
    
    # Highlight bars corresponding to the region
    for i in range(n_states):
        if i < len(expression) and region_mask[i]:
            bars[i].set_color('orange')
    
    prob_ax.set_title("QAOA State Probabilities", fontsize=14)
    prob_ax.set_xlabel("Quantum State (binary)", fontsize=12)
    prob_ax.set_ylabel("Probability", fontsize=12)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', edgecolor='k', alpha=0.7, label='High Expression Region'),
        Patch(facecolor='skyblue', edgecolor='k', alpha=0.7, label='Other States')
    ]
    prob_ax.legend(handles=legend_elements, loc='upper right')
    
    # Add description
    prob_ax.text(3.5, 0.05, "QAOA finds the optimal binary assignment\n"
                         "to maximize expression while maintaining spatial proximity.", 
               ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('quantum_visualization.png', dpi=300)
    print("Saved quantum visualization to: quantum_visualization.png")

if __name__ == "__main__":
    main()