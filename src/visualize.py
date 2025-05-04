#!/usr/bin/env python
# Visualization module for spatial transcriptomics quantum analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_expression_heatmap(coords, values, title="Gene Expression", 
                          cmap='viridis', size=100, alpha=0.8, 
                          highlight_max=True, ax=None, fig=None):
    """
    Plot a spatial heatmap of expression values.
    
    Args:
        coords: Array of (x,y) coordinates
        values: Expression values for each spot
        title: Plot title
        cmap: Colormap
        size: Marker size
        alpha: Transparency
        highlight_max: Whether to highlight maximum
        ax: Matplotlib axis to plot on
        fig: Matplotlib figure
        
    Returns:
        Figure and axis
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=values, cmap=cmap, 
        s=size, alpha=alpha, 
        edgecolors='k'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Expression Value')
    
    # Highlight maximum value
    if highlight_max:
        max_idx = np.argmax(values)
        ax.scatter(
            coords[max_idx, 0], coords[max_idx, 1], 
            s=size*2, facecolors='none', 
            edgecolors='red', linewidths=2, 
            label='Maximum'
        )
        ax.legend()
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return fig, ax

def plot_probability_distribution(probabilities, n_top=10, title="State Probabilities",
                               ax=None, fig=None):
    """
    Plot the probability distribution of quantum states.
    
    Args:
        probabilities: Array of probabilities for each state
        n_top: Number of top states to show
        title: Plot title
        ax: Matplotlib axis to plot on
        fig: Matplotlib figure
        
    Returns:
        Figure and axis
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort probabilities
    sorted_indices = np.argsort(probabilities)[::-1]
    top_indices = sorted_indices[:n_top]
    top_probs = probabilities[top_indices]
    
    # Convert indices to binary strings
    n_qubits = int(np.log2(len(probabilities)))
    labels = [format(idx, f'0{n_qubits}b') for idx in top_indices]
    
    # Create bar plot
    bars = ax.bar(range(len(top_probs)), top_probs)
    
    # Customize plot
    ax.set_xticks(range(len(top_probs)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Bitstring')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, min(1.0, max(top_probs) * 1.2))
    
    # Add values above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', rotation=0
        )
    
    ax.grid(axis='y', alpha=0.3)
    
    return fig, ax

def plot_region_map(coords, region_mask, expression=None, 
                  title="High Expression Region", ax=None, fig=None):
    """
    Plot a map of the detected high-expression region.
    
    Args:
        coords: Array of (x,y) coordinates
        region_mask: Boolean mask indicating spots in the region
        expression: Optional expression values for coloring region spots
        title: Plot title
        ax: Matplotlib axis to plot on
        fig: Matplotlib figure
        
    Returns:
        Figure and axis
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all spots in gray
    ax.scatter(
        coords[:, 0], coords[:, 1], 
        c='lightgray', s=80, alpha=0.6, 
        edgecolors='k', label='All spots'
    )
    
    # Highlight region spots
    if np.any(region_mask):
        region_coords = coords[region_mask]
        
        if expression is not None:
            # Color by expression
            region_expr = expression[region_mask]
            scatter = ax.scatter(
                region_coords[:, 0], region_coords[:, 1], 
                c=region_expr, cmap='viridis', 
                s=120, alpha=0.9, edgecolors='k', 
                label='Region'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Expression Value')
        else:
            # Single color for region
            ax.scatter(
                region_coords[:, 0], region_coords[:, 1], 
                c='red', s=120, alpha=0.9, 
                edgecolors='k', label='Region'
            )
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    
    return fig, ax

def plot_expression_gradient(coords, values, arrows=20, title="Expression Gradient",
                           cmap='viridis', ax=None, fig=None):
    """
    Plot expression values with gradient directions.
    
    Args:
        coords: Array of (x,y) coordinates
        values: Expression values for each spot
        arrows: Number of gradient arrows to show
        title: Plot title
        cmap: Colormap
        ax: Matplotlib axis to plot on
        fig: Matplotlib figure
        
    Returns:
        Figure and axis
    """
    from scipy.interpolate import Rbf
    from scipy.ndimage import gaussian_filter
    
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot expression values
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=values, cmap=cmap, 
        s=80, alpha=0.7, 
        edgecolors='k'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Expression Value')
    
    # Create a grid for interpolation
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    margin_x = (max_x - min_x) * 0.05
    margin_y = (max_y - min_y) * 0.05
    
    grid_x = np.linspace(min_x - margin_x, max_x + margin_x, 50)
    grid_y = np.linspace(min_y - margin_y, max_y + margin_y, 50)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate expression values
    try:
        rbf = Rbf(coords[:, 0], coords[:, 1], values, function='thin_plate')
        grid_values = rbf(grid_X, grid_Y)
    except:
        # Fallback to simple nearest-neighbor interpolation
        from scipy.interpolate import griddata
        grid_values = griddata(coords, values, (grid_X, grid_Y), method='nearest')
    
    # Smooth the interpolation
    grid_values = gaussian_filter(grid_values, sigma=1)
    
    # Calculate gradients
    gy, gx = np.gradient(grid_values)
    
    # Downsample gradient field for clarity
    skip = (len(grid_x) // arrows)
    if skip < 1:
        skip = 1
        
    # Plot gradient arrows
    ax.quiver(
        grid_X[::skip, ::skip], 
        grid_Y[::skip, ::skip], 
        gx[::skip, ::skip], 
        gy[::skip, ::skip], 
        color='black', scale=30, width=0.003,
        scale_units='width', alpha=0.7
    )
    
    # Highlight maximum
    max_idx = np.argmax(values)
    ax.scatter(
        coords[max_idx, 0], coords[max_idx, 1], 
        s=160, facecolors='none', 
        edgecolors='red', linewidths=2, 
        label='Maximum'
    )
    ax.legend()
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return fig, ax

def create_multi_panel_visualization(coords, values, probabilities, region_mask=None):
    """
    Create a multi-panel visualization of quantum analysis results.
    
    Args:
        coords: Array of (x,y) coordinates
        values: Expression values for each spot
        probabilities: Quantum state probabilities
        region_mask: Boolean mask indicating spots in the region
        
    Returns:
        Figure object
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Panel 1: Expression heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    plot_expression_heatmap(
        coords, values, title="Gene Expression", 
        highlight_max=True, ax=ax1, fig=fig
    )
    
    # Panel 2: Probability distribution
    ax2 = fig.add_subplot(2, 2, 2)
    plot_probability_distribution(
        probabilities, n_top=8, 
        title="Quantum State Probabilities", 
        ax=ax2, fig=fig
    )
    
    # Panel 3: Region map
    ax3 = fig.add_subplot(2, 2, 3)
    if region_mask is not None:
        plot_region_map(
            coords, region_mask, expression=values,
            title="High Expression Region", 
            ax=ax3, fig=fig
        )
    else:
        # Create a region mask from probabilities if not provided
        n_qubits = int(np.log2(len(probabilities)))
        temp_mask = np.zeros(len(coords), dtype=bool)
        max_idx = np.argmax(probabilities)
        binary = format(max_idx, f'0{n_qubits}b')
        
        for i, bit in enumerate(reversed(binary)):
            if bit == '1' and i < len(coords):
                temp_mask[i] = True
                
        plot_region_map(
            coords, temp_mask, expression=values,
            title="Predicted High Expression Region", 
            ax=ax3, fig=fig
        )
    
    # Panel 4: Expression gradient
    ax4 = fig.add_subplot(2, 2, 4)
    plot_expression_gradient(
        coords, values, arrows=15,
        title="Expression Gradient", 
        ax=ax4, fig=fig
    )
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_results_to_csv(filepath, coords, values, probabilities, region_mask=None,
                       index_map=None, spot_ids=None):
    """
    Save analysis results to a CSV file.
    
    Args:
        filepath: Path to save the CSV
        coords: Array of (x,y) coordinates
        values: Expression values for each spot
        probabilities: Quantum state probabilities
        region_mask: Boolean mask indicating spots in the region
        index_map: Mapping from new indices to original indices
        spot_ids: Original spot IDs if available
        
    Returns:
        DataFrame with saved results
    """
    # Create DataFrame
    results = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'expression': values
    })
    
    # Add spot IDs if available
    if spot_ids is not None:
        results['spot_id'] = spot_ids
    
    # Add original indices if available
    if index_map is not None:
        results['original_index'] = [index_map.get(i, i) for i in range(len(coords))]
    
    # Add region mask if available
    if region_mask is not None:
        results['in_region'] = region_mask
    
    # Add top state probabilities (limited to actual data points)
    n_qubits = int(np.log2(len(probabilities)))
    n_spots = len(coords)
    
    # Get top states
    top_indices = np.argsort(probabilities)[::-1][:5]
    
    for i, idx in enumerate(top_indices):
        if idx < n_spots:  # Only include states that map to actual spots
            binary = format(idx, f'0{n_qubits}b')
            results[f'state_{binary}'] = probabilities[idx]
    
    # Add probability of the state representing each spot
    spot_probs = []
    for i in range(n_spots):
        if i < len(probabilities):
            spot_probs.append(probabilities[i])
        else:
            spot_probs.append(0)
    
    results['probability'] = spot_probs
    
    # Save to CSV
    results.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize spatial transcriptomics quantum analysis')
    parser.add_argument('--results', required=True, help='Results file (CSV)')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--title', default='Gene Expression Analysis', help='Plot title')
    
    args = parser.parse_args()
    
    # Load results
    results = pd.read_csv(args.results)
    
    # Extract data
    coords = results[['x', 'y']].values
    values = results['expression'].values
    
    # Check if probabilities and region are available
    if 'probability' in results.columns:
        probabilities = results['probability'].values
    else:
        probabilities = None
    
    if 'in_region' in results.columns:
        region_mask = results['in_region'].values.astype(bool)
    else:
        region_mask = None
    
    # Create visualization
    if probabilities is not None:
        fig = create_multi_panel_visualization(coords, values, probabilities, region_mask)
        plt.suptitle(args.title, fontsize=16)
    else:
        fig, ax = plot_expression_heatmap(coords, values, title=args.title)
    
    # Save figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.output}")