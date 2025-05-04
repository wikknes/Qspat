#!/usr/bin/env python
# Script to generate synthetic spatial transcriptomics data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
NUM_SPOTS = 16  # Number of spatial spots/locations
NUM_GENES = 5   # Number of genes to simulate
GRID_SIZE = 4   # Grid size for spot arrangement
RANDOM_SEED = 42  # For reproducibility

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

def generate_spatial_grid(grid_size, noise=0.1):
    """Generate spatial coordinates in a grid pattern with some noise."""
    # Create a grid of coordinates
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Reshape to get pairs of coordinates
    coords = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Add some noise to make it more realistic
    coords += np.random.normal(0, noise, coords.shape)
    
    return coords

def generate_expression_patterns(coords, num_genes=5):
    """Generate synthetic gene expression patterns based on spatial coordinates."""
    num_spots = coords.shape[0]
    expression = np.zeros((num_spots, num_genes))
    
    # Gene 1: Gradient from bottom-left to top-right (distance from origin)
    expression[:, 0] = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    
    # Gene 2: Central peak (Gaussian)
    center = np.array([0.5, 0.5])
    distances = np.sqrt(np.sum((coords - center)**2, axis=1))
    expression[:, 1] = np.exp(-10 * distances**2)
    
    # Gene 3: Two clusters (mixture of Gaussians)
    center1 = np.array([0.25, 0.25])
    center2 = np.array([0.75, 0.75])
    distances1 = np.sqrt(np.sum((coords - center1)**2, axis=1))
    distances2 = np.sqrt(np.sum((coords - center2)**2, axis=1))
    expression[:, 2] = 0.7 * np.exp(-15 * distances1**2) + 0.8 * np.exp(-15 * distances2**2)
    
    # Gene 4: Stripe pattern along x-axis
    expression[:, 3] = 0.5 + 0.5 * np.sin(6 * np.pi * coords[:, 0])
    
    # Gene 5: Random pattern (with spatial autocorrelation)
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    cov = np.exp(-3 * distances)  # Exponential decay covariance
    expression[:, 4] = np.random.multivariate_normal(np.zeros(num_spots), cov)
    
    # Normalize to [0,1] range per gene
    for i in range(num_genes):
        min_val = expression[:, i].min()
        max_val = expression[:, i].max()
        if max_val > min_val:  # Avoid division by zero
            expression[:, i] = (expression[:, i] - min_val) / (max_val - min_val)
    
    # Add some noise
    expression += np.random.normal(0, 0.05, expression.shape)
    expression = np.clip(expression, 0, 1)  # Clip to [0,1]
    
    return expression

def create_data_frame(coords, expression):
    """Create a pandas DataFrame with coordinates and expression data."""
    # Create DataFrame with coordinates
    df = pd.DataFrame(coords, columns=['x', 'y'])
    
    # Add gene expression columns
    for i in range(expression.shape[1]):
        df[f'Gene{i+1}'] = expression[:, i]
    
    return df

def visualize_data(df):
    """Visualize the synthetic data."""
    num_genes = len(df.columns) - 2  # Subtract x and y columns
    
    fig, axes = plt.subplots(1, num_genes, figsize=(4*num_genes, 4))
    
    for i in range(num_genes):
        gene_col = f'Gene{i+1}'
        ax = axes[i]
        scatter = ax.scatter(df['x'], df['y'], c=df[gene_col], s=100, cmap='viridis')
        ax.set_title(f'Expression of {gene_col}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.colorbar(scatter, ax=ax, label='Expression level')
    
    plt.tight_layout()
    plt.savefig('synthetic_expression_patterns.png', dpi=300)
    plt.close()

# Generate data
coords = generate_spatial_grid(GRID_SIZE)
expression = generate_expression_patterns(coords, NUM_GENES)
df = create_data_frame(coords, expression)

# Save data
df.to_csv('synthetic_data.csv', index=False)
print(f"Created synthetic data with {NUM_SPOTS} spots and {NUM_GENES} genes")
print(f"Saved to: synthetic_data.csv")

# Visualize
visualize_data(df)
print("Visualization saved to: synthetic_expression_patterns.png")

# Show the first few rows of the data
print("\nData preview:")
print(df.head())