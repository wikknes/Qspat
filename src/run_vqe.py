#!/usr/bin/env python
# VQE implementation for spatial transcriptomics analysis

import numpy as np
import time
import logging
import matplotlib.pyplot as plt

from preprocess import process_data
from encoding import hamiltonian_encoding

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpatialVQE:
    """
    Class implementing VQE for spatial transcriptomics.
    """
    
    def __init__(self, expression_file, coords_file=None, target_gene=None,
                 max_spots=16, optimizer='cobyla'):
        """
        Initialize the VQE solver.
        
        Args:
            expression_file: Path to expression data
            coords_file: Path to coordinates (if separate)
            target_gene: Specific gene to analyze
            max_spots: Maximum number of spots (qubit budget)
            optimizer: Optimization algorithm ('cobyla', 'spsa', or 'slsqp')
        """
        self.expression_file = expression_file
        self.coords_file = coords_file
        self.target_gene = target_gene
        self.max_spots = max_spots
        self.optimizer = optimizer
        
        # Will be initialized during preprocessing
        self.expr_vector = None
        self.coord_map = None
        self.n_qubits = None
        self.index_map = None
        self.hamiltonian = None
        self.result = None
        
        logger.info(f"Initialized SpatialVQE with {optimizer} optimizer")
    
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
        
        logger.info(f"Preprocessing complete: {len(self.expr_vector)} spots, {self.n_qubits} qubits")
        return self.expr_vector, self.coord_map
    
    def setup(self):
        """
        Set up the VQE components.
        """
        logger.info("Setting up VQE components")
        
        # Ensure data is preprocessed
        if self.expr_vector is None:
            self.preprocess()
        
        # Create Hamiltonian
        self.hamiltonian = hamiltonian_encoding(self.expr_vector)
        
        logger.info("VQE setup complete")
        return None
    
    def run(self):
        """
        Run the VQE algorithm.
        """
        logger.info("Starting VQE optimization")
        
        # Ensure VQE is set up
        if self.hamiltonian is None:
            self.setup()
        
        # Time the execution
        start_time = time.time()
        
        # Mock result for demonstration
        self.result = {"eigenvalue": 0.0, "optimal_parameters": {}}
        
        end_time = time.time()
        exec_time = end_time - start_time
        
        logger.info(f"VQE completed in {exec_time:.2f} seconds")
        
        return self.result
    
    def analyze_results(self):
        """
        Analyze the VQE results to find the maximum expression location.
        
        Returns:
            Tuple of (max_index, max_coordinates, probabilities)
        """
        if self.result is None:
            raise ValueError("No VQE results available. Run the algorithm first.")
        
        logger.info("Analyzing VQE results")
        
        # For demonstration, create mock probabilities based on expression values
        probabilities = np.zeros(2**self.n_qubits)
        normalized_expr = self.expr_vector / np.sum(self.expr_vector)
        for i, val in enumerate(normalized_expr):
            probabilities[i] = val
        
        # Find the index with highest probability
        max_prob_index = np.argmax(probabilities)
        max_prob = probabilities[max_prob_index]
        
        # Convert to original index if we have an index map
        if self.index_map and max_prob_index in self.index_map:
            original_index = self.index_map[max_prob_index]
            logger.info(f"Max probability {max_prob:.4f} at index {max_prob_index} (original index {original_index})")
        else:
            original_index = max_prob_index
            logger.info(f"Max probability {max_prob:.4f} at index {max_prob_index}")
        
        # Get coordinates for this spot
        if max_prob_index < len(self.coord_map):
            max_coordinates = self.coord_map[max_prob_index]
            logger.info(f"Coordinates of maximum expression: ({max_coordinates[0]:.2f}, {max_coordinates[1]:.2f})")
        else:
            max_coordinates = None
            logger.warning(f"Max index {max_prob_index} exceeds coordinate map size {len(self.coord_map)}")
        
        return max_prob_index, max_coordinates, probabilities
    
    def visualize(self, probabilities=None, show=True, save_path=None):
        """
        Visualize the results.
        
        Args:
            probabilities: Probability distribution (if None, compute from results)
            show: Whether to display the plot
            save_path: Path to save the plot image
            
        Returns:
            Figure object
        """
        if probabilities is None and self.result is not None:
            _, _, probabilities = self.analyze_results()
        
        if probabilities is None:
            raise ValueError("No probabilities available for visualization")
        
        logger.info("Creating visualization")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot coordinates with probability as color
        xs = self.coord_map[:, 0]
        ys = self.coord_map[:, 1]
        
        # Only use probabilities for actual data points (in case of padding)
        plot_probs = probabilities[:len(xs)]
        
        # Create scatter plot
        scatter = ax.scatter(xs, ys, c=plot_probs, cmap='viridis', 
                           s=100, alpha=0.8, edgecolors='k')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Probability')
        
        # Highlight maximum
        max_idx = np.argmax(plot_probs)
        ax.scatter(xs[max_idx], ys[max_idx], s=200, facecolors='none', 
                 edgecolors='red', linewidths=2, label='Maximum')
        
        # Add labels and title
        gene_label = self.target_gene if self.target_gene else "Selected genes"
        ax.set_title(f'Expression probability for {gene_label}')
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
    parser = argparse.ArgumentParser(description='Run VQE for spatial transcriptomics')
    parser.add_argument('--expression', required=True, help='Expression data file')
    parser.add_argument('--coords', help='Coordinates file (if separate)')
    parser.add_argument('--gene', help='Target gene to analyze')
    parser.add_argument('--max_spots', type=int, default=16, help='Maximum number of spots')
    parser.add_argument('--optimizer', default='cobyla', choices=['cobyla', 'spsa', 'slsqp'],
                      help='Optimization algorithm')
    parser.add_argument('--output', help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Create and run VQE
    vqe_solver = SpatialVQE(
        args.expression,
        coords_file=args.coords,
        target_gene=args.gene,
        max_spots=args.max_spots,
        optimizer=args.optimizer
    )
    
    vqe_solver.preprocess()
    vqe_solver.setup()
    result = vqe_solver.run()
    
    max_idx, max_coords, probs = vqe_solver.analyze_results()
    print(f"Maximum expression at index {max_idx}")
    if max_coords is not None:
        print(f"Coordinates: ({max_coords[0]:.2f}, {max_coords[1]:.2f})")
    
    vqe_solver.visualize(save_path=args.output)