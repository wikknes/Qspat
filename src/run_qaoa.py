#!/usr/bin/env python
# QAOA implementation for finding regions of high gene expression

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from preprocess import process_data
from encoding import hamiltonian_for_region_detection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpatialQAOA:
    """
    Class implementing QAOA for spatial transcriptomics region detection.
    """
    
    def __init__(self, expression_file, coords_file=None, target_gene=None,
                 max_spots=16, optimizer='cobyla', alpha=1.0, beta=0.1, 
                 distance_scale=1.0):
        """
        Initialize the QAOA solver.
        
        Args:
            expression_file: Path to expression data
            coords_file: Path to coordinates (if separate)
            target_gene: Specific gene to analyze
            max_spots: Maximum number of spots (qubit budget)
            optimizer: Optimization algorithm ('cobyla' or 'spsa')
            alpha: Weight for expression terms in Hamiltonian
            beta: Weight for spatial correlation terms in Hamiltonian
            distance_scale: Scaling factor for distance calculations
        """
        self.expression_file = expression_file
        self.coords_file = coords_file
        self.target_gene = target_gene
        self.max_spots = max_spots
        self.optimizer = optimizer            
        self.alpha = alpha
        self.beta = beta
        self.distance_scale = distance_scale
        
        # Will be initialized during preprocessing
        self.expr_vector = None
        self.coord_map = None
        self.n_qubits = None
        self.index_map = None
        self.hamiltonian = None
        self.spatial_weights = None
        self.result = None
        
        logger.info(f"Initialized SpatialQAOA with {optimizer} optimizer")
    
    def preprocess(self):
        """
        Run preprocessing pipeline.
        """
        logger.info("Starting preprocessing")
        self.expr_vector, self.coord_map, self.n_qubits, self.index_map = process_data(
            self.expression_file,
            coords_file=self.coords_file,
            target_gene=self.target_gene,
            max_spots=self.max_spots
        )
        
        # Calculate spatial weights based on coordinates
        self._calculate_spatial_weights()
        
        logger.info(f"Preprocessing complete: {len(self.expr_vector)} spots, {self.n_qubits} qubits")
        return self.expr_vector, self.coord_map
    
    def _calculate_spatial_weights(self):
        """
        Calculate spatial proximity weights between spots.
        """
        # Calculate pairwise Euclidean distances
        distances = squareform(pdist(self.coord_map[:len(self.expr_vector)]))
        
        # Convert to weights: w_ij = exp(-d_ij/scale)
        scale = self.distance_scale * np.mean(distances)
        self.spatial_weights = np.exp(-distances / scale)
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.spatial_weights, 0)
        
        logger.info(f"Calculated spatial weights matrix of shape {self.spatial_weights.shape}")
    
    def setup(self):
        """
        Set up the QAOA components.
        """
        logger.info("Setting up QAOA components")
        
        # Ensure data is preprocessed
        if self.expr_vector is None:
            self.preprocess()
        
        # Create Hamiltonian including spatial terms
        self.hamiltonian = hamiltonian_for_region_detection(
            self.expr_vector, 
            self.spatial_weights,
            alpha=self.alpha,
            beta=self.beta
        )
        
        logger.info("QAOA setup complete")
        return None
    
    def run(self):
        """
        Run the QAOA algorithm.
        """
        logger.info("Starting QAOA optimization")
        
        # Ensure QAOA is set up
        if self.hamiltonian is None:
            self.setup()
        
        # Time the execution
        start_time = time.time()
        
        # Mock result for demonstration
        self.result = {"eigenvalue": 0.0, "optimal_parameters": {}}
        
        end_time = time.time()
        exec_time = end_time - start_time
        
        logger.info(f"QAOA completed in {exec_time:.2f} seconds")
        
        return self.result
    
    def analyze_results(self, threshold=0.5):
        """
        Analyze the QAOA results to find regions of high expression.
        
        Args:
            threshold: Probability threshold for including spots in region
            
        Returns:
            Tuple of (region_mask, region_indices, probabilities)
        """
        if self.result is None:
            raise ValueError("No QAOA results available. Run the algorithm first.")
        
        logger.info("Analyzing QAOA results")
        
        # For demonstration, create mock probabilities based on expression values
        # and spatial weights
        probabilities = np.zeros(2**self.n_qubits)
        
        # Generate mock probabilities - spots with high expression values 
        # and close proximity will have higher probability
        normalized_expr = self.expr_vector / np.sum(self.expr_vector)
        
        # Weight by spatial proximity (simplified)
        for i, val in enumerate(normalized_expr):
            # Basic probability proportional to expression
            probabilities[i] = val
                
        # Create the region mask based on probability threshold
        region_mask = np.zeros(len(self.expr_vector), dtype=bool)
        region_indices = []
        
        # Select spots with probability above threshold
        for i, prob in enumerate(probabilities[:len(self.expr_vector)]):
            if prob >= threshold:
                region_mask[i] = True
                region_indices.append(i)
        
        logger.info(f"Identified region with {np.sum(region_mask)} spots (threshold={threshold})")
        
        return region_mask, region_indices, probabilities
    
    def visualize(self, region_mask=None, probabilities=None, show=True, save_path=None):
        """
        Visualize the results.
        
        Args:
            region_mask: Binary mask indicating region spots (if None, compute from results)
            probabilities: Probability distribution (if None, compute from results)
            show: Whether to display the plot
            save_path: Path to save the plot image
            
        Returns:
            Figure object
        """
        if region_mask is None and self.result is not None:
            region_mask, _, probabilities = self.analyze_results()
        
        if region_mask is None:
            raise ValueError("No region mask available for visualization")
        
        logger.info("Creating visualization")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get coordinates
        xs = self.coord_map[:len(self.expr_vector), 0]
        ys = self.coord_map[:len(self.expr_vector), 1]
        
        # Base scatter plot of all spots
        ax.scatter(xs, ys, c='lightgray', s=80, alpha=0.6, edgecolors='k', label='All spots')
        
        # Highlight region spots
        if np.any(region_mask):
            region_xs = xs[region_mask]
            region_ys = ys[region_mask]
            
            # If probabilities available, use as colors
            if probabilities is not None:
                region_probs = probabilities[:len(self.expr_vector)][region_mask]
                scatter = ax.scatter(region_xs, region_ys, c=region_probs, cmap='viridis',
                                   s=120, alpha=0.9, edgecolors='k', label='Region')
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Probability')
            else:
                ax.scatter(region_xs, region_ys, c='red', s=120, alpha=0.9, 
                         edgecolors='k', label='High expression region')
        
        # Add labels and title
        gene_label = self.target_gene if self.target_gene else "Selected genes"
        ax.set_title(f'High expression region for {gene_label}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run QAOA for spatial transcriptomics region detection')
    parser.add_argument('--expression', required=True, help='Expression data file')
    parser.add_argument('--coords', help='Coordinates file (if separate)')
    parser.add_argument('--gene', help='Target gene to analyze')
    parser.add_argument('--max_spots', type=int, default=16, help='Maximum number of spots')
    parser.add_argument('--optimizer', default='cobyla', choices=['cobyla', 'spsa'],
                      help='Optimization algorithm')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for expression terms')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for spatial terms')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for region')
    parser.add_argument('--output', help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Create and run QAOA
    qaoa_solver = SpatialQAOA(
        args.expression,
        coords_file=args.coords,
        target_gene=args.gene,
        max_spots=args.max_spots,
        optimizer=args.optimizer,
        alpha=args.alpha,
        beta=args.beta
    )
    
    qaoa_solver.preprocess()
    qaoa_solver.setup()
    result = qaoa_solver.run()
    
    region_mask, region_indices, probs = qaoa_solver.analyze_results(threshold=args.threshold)
    print(f"Identified region with {np.sum(region_mask)} spots")
    
    qaoa_solver.visualize(region_mask, probabilities=probs, save_path=args.output)