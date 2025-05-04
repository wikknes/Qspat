#!/usr/bin/env python
# Benchmarking script to compare Qspat quantum methods with classical approaches

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
import os

# Import Qspat modules
from src.run_vqe import SpatialVQE
from src.run_qaoa import SpatialQAOA
from src.preprocess import process_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Benchmark configuration
DEFAULT_DATA_PATH = "data/synthetic_data.csv"
DEFAULT_TARGET_GENE = "Gene2"  # Gene with a central peak pattern
DEFAULT_MAX_SPOTS = 16
OUTPUT_DIR = "benchmark_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Benchmark:
    """
    Class for benchmarking Qspat against classical methods.
    """
    
    def __init__(self, data_path=DEFAULT_DATA_PATH, target_gene=DEFAULT_TARGET_GENE,
                 max_spots=DEFAULT_MAX_SPOTS):
        """
        Initialize the benchmark runner.
        
        Args:
            data_path: Path to expression data
            target_gene: Target gene to analyze
            max_spots: Maximum number of spots to consider
        """
        self.data_path = data_path
        self.target_gene = target_gene
        self.max_spots = max_spots
        
        # Will be initialized during data loading
        self.expr_vector = None
        self.coord_map = None
        self.n_qubits = None
        self.index_map = None
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized benchmark with target gene: {target_gene}")
    
    def load_data(self):
        """
        Load and preprocess the data.
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Use Qspat's preprocessing pipeline
        self.expr_vector, self.coord_map, self.n_qubits, self.index_map = process_data(
            self.data_path,
            target_gene=self.target_gene,
            max_spots=self.max_spots
        )
        
        logger.info(f"Loaded data with {len(self.expr_vector)} spots")
        return self.expr_vector, self.coord_map
    
    def _measure_performance(self, func, *args, **kwargs):
        """
        Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments for the function
            
        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        
        return result, exec_time
    
    #----------------------------------------------------------------------
    # Maximum Finding Benchmarks (VQE comparisons)
    #----------------------------------------------------------------------
    
    def run_qspat_vqe(self):
        """
        Run Qspat's VQE algorithm for maximum finding.
        
        Returns:
            Dict with results and metrics
        """
        logger.info("Running Qspat VQE")
        
        # Create VQE solver
        vqe_solver = SpatialVQE(
            self.data_path,
            target_gene=self.target_gene,
            max_spots=self.max_spots,
            optimizer='cobyla'
        )
        
        # Measure preprocessing time
        _, preprocess_time = self._measure_performance(vqe_solver.preprocess)
        
        # Measure setup time
        _, setup_time = self._measure_performance(vqe_solver.setup)
        
        # Measure execution time
        _, run_time = self._measure_performance(vqe_solver.run)
        
        # Measure analysis time
        (max_idx, max_coords, probabilities), analysis_time = self._measure_performance(vqe_solver.analyze_results)
        
        # Calculate total time
        total_time = preprocess_time + setup_time + run_time + analysis_time
        
        # Store results
        results = {
            "method": "Qspat VQE",
            "max_idx": max_idx,
            "max_coords": max_coords,
            "max_value": self.expr_vector[max_idx],
            "execution_times": {
                "preprocess": preprocess_time,
                "setup": setup_time,
                "run": run_time,
                "analysis": analysis_time,
                "total": total_time
            },
            "probabilities": probabilities
        }
        
        self.results["qspat_vqe"] = results
        logger.info(f"Qspat VQE found maximum at index {max_idx} in {total_time:.4f} seconds")
        
        return results
    
    def run_classical_max(self):
        """
        Run classical maximum finding algorithm (numpy.argmax).
        
        Returns:
            Dict with results and metrics
        """
        logger.info("Running classical maximum finding (numpy.argmax)")
        
        # Measure execution time
        def find_max():
            max_idx = np.argmax(self.expr_vector)
            max_coords = self.coord_map[max_idx]
            return max_idx, max_coords
        
        (max_idx, max_coords), exec_time = self._measure_performance(find_max)
        
        # Store results
        results = {
            "method": "Classical max (numpy.argmax)",
            "max_idx": max_idx,
            "max_coords": max_coords,
            "max_value": self.expr_vector[max_idx],
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["classical_max"] = results
        logger.info(f"Classical max found maximum at index {max_idx} in {exec_time:.4f} seconds")
        
        return results
    
    def run_hill_climbing(self, num_starts=3, max_steps=100):
        """
        Run hill climbing algorithm for maximum finding.
        
        Args:
            num_starts: Number of random starting points
            max_steps: Maximum steps per run
            
        Returns:
            Dict with results and metrics
        """
        logger.info(f"Running hill climbing with {num_starts} starts")
        
        # Define hill climbing function
        def hill_climbing():
            best_idx = None
            best_val = -float('inf')
            
            # Multiple random starts
            for _ in range(num_starts):
                # Random starting point
                current_idx = np.random.randint(0, len(self.expr_vector))
                current_val = self.expr_vector[current_idx]
                
                # Calculate pairwise distances to find neighbors
                distances = np.sqrt(np.sum((self.coord_map - self.coord_map[current_idx])**2, axis=1))
                neighbors = np.argsort(distances)[1:5]  # 4 nearest neighbors
                
                # Hill climbing steps
                for _ in range(max_steps):
                    # Find best neighbor
                    best_neighbor = None
                    best_neighbor_val = current_val
                    
                    for neighbor in neighbors:
                        if neighbor < len(self.expr_vector):
                            if self.expr_vector[neighbor] > best_neighbor_val:
                                best_neighbor = neighbor
                                best_neighbor_val = self.expr_vector[neighbor]
                    
                    # If no improvement, break
                    if best_neighbor is None or best_neighbor_val <= current_val:
                        break
                    
                    # Move to best neighbor
                    current_idx = best_neighbor
                    current_val = best_neighbor_val
                    
                    # Recalculate neighbors
                    distances = np.sqrt(np.sum((self.coord_map - self.coord_map[current_idx])**2, axis=1))
                    neighbors = np.argsort(distances)[1:5]  # 4 nearest neighbors
                
                # Update best overall
                if current_val > best_val:
                    best_idx = current_idx
                    best_val = current_val
            
            return best_idx, self.coord_map[best_idx]
        
        # Measure execution time
        (max_idx, max_coords), exec_time = self._measure_performance(hill_climbing)
        
        # Store results
        results = {
            "method": "Hill climbing",
            "max_idx": max_idx,
            "max_coords": max_coords,
            "max_value": self.expr_vector[max_idx],
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["hill_climbing"] = results
        logger.info(f"Hill climbing found maximum at index {max_idx} in {exec_time:.4f} seconds")
        
        return results
    
    def run_gradient_ascent(self, learning_rate=0.01, max_steps=100):
        """
        Run gradient ascent for maximum finding using interpolation.
        
        Args:
            learning_rate: Learning rate for gradient steps
            max_steps: Maximum steps
            
        Returns:
            Dict with results and metrics
        """
        logger.info("Running gradient ascent")
        
        from scipy.interpolate import LinearNDInterpolator
        
        # Define gradient ascent function
        def gradient_ascent():
            # Create interpolation function
            interp = LinearNDInterpolator(self.coord_map, self.expr_vector)
            
            # Random starting point within coord bounds
            x_min, y_min = np.min(self.coord_map, axis=0)
            x_max, y_max = np.max(self.coord_map, axis=0)
            
            # Multiple starting points
            num_starts = 5
            best_pos = None
            best_val = -float('inf')
            
            for _ in range(num_starts):
                # Random start
                current_pos = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max)
                ])
                
                # Gradient ascent steps
                for _ in range(max_steps):
                    # Evaluate current position
                    current_val = interp(current_pos[0], current_pos[1])
                    
                    # Calculate numerical gradient
                    eps = 1e-5
                    grad_x = (interp(current_pos[0] + eps, current_pos[1]) - 
                              interp(current_pos[0] - eps, current_pos[1])) / (2 * eps)
                    grad_y = (interp(current_pos[0], current_pos[1] + eps) - 
                              interp(current_pos[0], current_pos[1] - eps)) / (2 * eps)
                    
                    # Handle NaN gradient components
                    if np.isnan(grad_x):
                        grad_x = 0
                    if np.isnan(grad_y):
                        grad_y = 0
                    
                    # Break if gradient is very small
                    if abs(grad_x) < 1e-5 and abs(grad_y) < 1e-5:
                        break
                    
                    # Update position
                    current_pos += learning_rate * np.array([grad_x, grad_y])
                    
                    # Keep within bounds
                    current_pos[0] = max(x_min, min(x_max, current_pos[0]))
                    current_pos[1] = max(y_min, min(y_max, current_pos[1]))
                
                # Evaluate final position
                final_val = interp(current_pos[0], current_pos[1])
                
                # Update best
                if final_val > best_val:
                    best_pos = current_pos
                    best_val = final_val
            
            # Find closest actual data point
            distances = np.sqrt(np.sum((self.coord_map - best_pos)**2, axis=1))
            closest_idx = np.argmin(distances)
            
            return closest_idx, self.coord_map[closest_idx]
        
        # Measure execution time
        (max_idx, max_coords), exec_time = self._measure_performance(gradient_ascent)
        
        # Store results
        results = {
            "method": "Gradient ascent",
            "max_idx": max_idx,
            "max_coords": max_coords,
            "max_value": self.expr_vector[max_idx],
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["gradient_ascent"] = results
        logger.info(f"Gradient ascent found maximum at index {max_idx} in {exec_time:.4f} seconds")
        
        return results
    
    #----------------------------------------------------------------------
    # Region Detection Benchmarks (QAOA comparisons)
    #----------------------------------------------------------------------
    
    def run_qspat_qaoa(self, threshold=0.5):
        """
        Run Qspat's QAOA algorithm for region detection.
        
        Args:
            threshold: Probability threshold for region detection
            
        Returns:
            Dict with results and metrics
        """
        logger.info("Running Qspat QAOA")
        
        # Create QAOA solver
        qaoa_solver = SpatialQAOA(
            self.data_path,
            target_gene=self.target_gene,
            max_spots=self.max_spots,
            optimizer='cobyla',
            alpha=1.0,
            beta=0.1
        )
        
        # Measure preprocessing time
        _, preprocess_time = self._measure_performance(qaoa_solver.preprocess)
        
        # Measure setup time
        _, setup_time = self._measure_performance(qaoa_solver.setup)
        
        # Measure execution time
        _, run_time = self._measure_performance(qaoa_solver.run)
        
        # Measure analysis time
        (region_mask, region_indices, probabilities), analysis_time = self._measure_performance(
            qaoa_solver.analyze_results, threshold=threshold)
        
        # Calculate total time
        total_time = preprocess_time + setup_time + run_time + analysis_time
        
        # Store results
        results = {
            "method": "Qspat QAOA",
            "region_mask": region_mask,
            "region_indices": region_indices,
            "region_size": sum(region_mask),
            "region_avg_expr": np.mean(self.expr_vector[region_mask]) if sum(region_mask) > 0 else 0,
            "execution_times": {
                "preprocess": preprocess_time,
                "setup": setup_time,
                "run": run_time,
                "analysis": analysis_time,
                "total": total_time
            },
            "probabilities": probabilities
        }
        
        self.results["qspat_qaoa"] = results
        logger.info(f"Qspat QAOA found region with {sum(region_mask)} spots in {total_time:.4f} seconds")
        
        return results
    
    def run_threshold_region(self, threshold=0.7):
        """
        Run simple thresholding for region detection.
        
        Args:
            threshold: Expression threshold
            
        Returns:
            Dict with results and metrics
        """
        logger.info(f"Running threshold-based region detection (threshold={threshold})")
        
        # Define thresholding function
        def threshold_region():
            # Apply threshold
            region_mask = self.expr_vector >= threshold
            region_indices = np.where(region_mask)[0]
            
            return region_mask, region_indices
        
        # Measure execution time
        (region_mask, region_indices), exec_time = self._measure_performance(threshold_region)
        
        # Store results
        results = {
            "method": f"Thresholding (t={threshold})",
            "region_mask": region_mask,
            "region_indices": region_indices,
            "region_size": sum(region_mask),
            "region_avg_expr": np.mean(self.expr_vector[region_mask]) if sum(region_mask) > 0 else 0,
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["threshold_region"] = results
        logger.info(f"Thresholding found region with {sum(region_mask)} spots in {exec_time:.4f} seconds")
        
        return results
    
    def run_kmeans_region(self, n_clusters=2):
        """
        Run K-means clustering for region detection.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dict with results and metrics
        """
        logger.info(f"Running K-means clustering with {n_clusters} clusters")
        
        # Define K-means function
        def kmeans_region():
            # Features: coordinates and expression
            features = np.column_stack([self.coord_map, self.expr_vector.reshape(-1, 1)])
            
            # Normalize features
            features_normalized = features.copy()
            for i in range(features.shape[1]):
                if np.std(features[:, i]) > 0:
                    features_normalized[:, i] = (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features_normalized)
            
            # Find cluster with highest average expression
            avg_expr = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_mask = labels == i
                if np.sum(cluster_mask) > 0:
                    avg_expr[i] = np.mean(self.expr_vector[cluster_mask])
            
            # Use the cluster with highest average expression
            high_expr_cluster = np.argmax(avg_expr)
            region_mask = labels == high_expr_cluster
            region_indices = np.where(region_mask)[0]
            
            return region_mask, region_indices
        
        # Measure execution time
        (region_mask, region_indices), exec_time = self._measure_performance(kmeans_region)
        
        # Store results
        results = {
            "method": f"K-means (k={n_clusters})",
            "region_mask": region_mask,
            "region_indices": region_indices,
            "region_size": sum(region_mask),
            "region_avg_expr": np.mean(self.expr_vector[region_mask]) if sum(region_mask) > 0 else 0,
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["kmeans_region"] = results
        logger.info(f"K-means found region with {sum(region_mask)} spots in {exec_time:.4f} seconds")
        
        return results
    
    def run_dbscan_region(self, eps=0.2, min_samples=2):
        """
        Run DBSCAN clustering for region detection.
        
        Args:
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Dict with results and metrics
        """
        logger.info(f"Running DBSCAN clustering (eps={eps}, min_samples={min_samples})")
        
        # Define DBSCAN function
        def dbscan_region():
            # Features: coordinates and weighted expression
            # We'll weight expression more heavily to ensure it's a significant factor
            weight = 2.0
            features = np.column_stack([
                self.coord_map,
                weight * self.expr_vector.reshape(-1, 1)
            ])
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)
            
            # Find cluster (excluding noise) with highest average expression
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
            
            if len(unique_labels) == 0:
                # No clusters found, return top 25% by expression
                threshold = np.percentile(self.expr_vector, 75)
                region_mask = self.expr_vector >= threshold
                region_indices = np.where(region_mask)[0]
            else:
                # Find cluster with highest average expression
                avg_expr = np.zeros(len(unique_labels))
                for i, label in enumerate(unique_labels):
                    cluster_mask = labels == label
                    if np.sum(cluster_mask) > 0:
                        avg_expr[i] = np.mean(self.expr_vector[cluster_mask])
                
                # Use the cluster with highest average expression
                high_expr_cluster = unique_labels[np.argmax(avg_expr)]
                region_mask = labels == high_expr_cluster
                region_indices = np.where(region_mask)[0]
            
            return region_mask, region_indices
        
        # Measure execution time
        (region_mask, region_indices), exec_time = self._measure_performance(dbscan_region)
        
        # Store results
        results = {
            "method": f"DBSCAN (eps={eps})",
            "region_mask": region_mask,
            "region_indices": region_indices,
            "region_size": sum(region_mask),
            "region_avg_expr": np.mean(self.expr_vector[region_mask]) if sum(region_mask) > 0 else 0,
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["dbscan_region"] = results
        logger.info(f"DBSCAN found region with {sum(region_mask)} spots in {exec_time:.4f} seconds")
        
        return results
    
    def run_gmm_region(self, n_components=2):
        """
        Run Gaussian Mixture Model for region detection.
        
        Args:
            n_components: Number of components
            
        Returns:
            Dict with results and metrics
        """
        logger.info(f"Running GMM with {n_components} components")
        
        # Define GMM function
        def gmm_region():
            # Features: coordinates and expression
            features = np.column_stack([
                self.coord_map,
                2.0 * self.expr_vector.reshape(-1, 1)  # Weight expression more heavily
            ])
            
            # Apply GMM
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            labels = gmm.fit_predict(features)
            
            # Calculate average expression for each component
            avg_expr = np.zeros(n_components)
            for i in range(n_components):
                component_mask = labels == i
                if np.sum(component_mask) > 0:
                    avg_expr[i] = np.mean(self.expr_vector[component_mask])
            
            # Use the component with highest average expression
            high_expr_component = np.argmax(avg_expr)
            region_mask = labels == high_expr_component
            region_indices = np.where(region_mask)[0]
            
            return region_mask, region_indices
        
        # Measure execution time
        (region_mask, region_indices), exec_time = self._measure_performance(gmm_region)
        
        # Store results
        results = {
            "method": f"GMM (k={n_components})",
            "region_mask": region_mask,
            "region_indices": region_indices,
            "region_size": sum(region_mask),
            "region_avg_expr": np.mean(self.expr_vector[region_mask]) if sum(region_mask) > 0 else 0,
            "execution_times": {
                "total": exec_time
            }
        }
        
        self.results["gmm_region"] = results
        logger.info(f"GMM found region with {sum(region_mask)} spots in {exec_time:.4f} seconds")
        
        return results
    
    #----------------------------------------------------------------------
    # Visualization and Reporting
    #----------------------------------------------------------------------
    
    def visualize_max_finding_results(self):
        """
        Visualize maximum finding results from different methods.
        
        Returns:
            Figure object
        """
        logger.info("Creating maximum finding visualization")
        
        # Get results for VQE comparisons
        methods = ['qspat_vqe', 'classical_max', 'hill_climbing', 'gradient_ascent']
        available_methods = [m for m in methods if m in self.results]
        
        if not available_methods:
            logger.warning("No maximum finding results available for visualization")
            return None
        
        # Create figure
        n_methods = len(available_methods)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
        
        # Plot original expression
        scatter0 = axes[0].scatter(self.coord_map[:, 0], self.coord_map[:, 1], 
                              c=self.expr_vector, cmap='viridis', 
                              s=100, alpha=0.8, edgecolors='k')
        axes[0].set_title(f'Original Expression of {self.target_gene}')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        cbar0 = plt.colorbar(scatter0, ax=axes[0])
        cbar0.set_label('Expression level')
        
        # Plot each method's result
        for i, method_key in enumerate(available_methods):
            method_results = self.results[method_key]
            
            # Plot expression
            axes[i+1].scatter(self.coord_map[:, 0], self.coord_map[:, 1], 
                           c=self.expr_vector, cmap='viridis', 
                           s=100, alpha=0.5, edgecolors='k')
            
            # Highlight maximum
            max_idx = method_results['max_idx']
            max_coords = method_results['max_coords']
            
            # Circle the maximum
            axes[i+1].scatter(max_coords[0], max_coords[1], s=250, facecolors='none', 
                           edgecolors='red', linewidths=3, label='Maximum')
            
            # Add execution time to title
            axes[i+1].set_title(f"{method_results['method']}\nTime: {method_results['execution_times']['total']:.4f} s")
            axes[i+1].set_xlabel('X coordinate')
            axes[i+1].set_ylabel('Y coordinate')
            axes[i+1].legend()
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(OUTPUT_DIR, 'max_finding_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved maximum finding visualization to {save_path}")
        
        return fig
    
    def visualize_region_detection_results(self):
        """
        Visualize region detection results from different methods.
        
        Returns:
            Figure object
        """
        logger.info("Creating region detection visualization")
        
        # Get results for QAOA comparisons
        methods = ['qspat_qaoa', 'threshold_region', 'kmeans_region', 'dbscan_region', 'gmm_region']
        available_methods = [m for m in methods if m in self.results]
        
        if not available_methods:
            logger.warning("No region detection results available for visualization")
            return None
        
        # Create figure
        n_methods = len(available_methods)
        n_rows = (n_methods + 1) // 2 + 1  # Original + methods (2 per row)
        n_cols = min(2, n_methods + 1)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))
        
        # Ensure axes is always a 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot original expression in first subplot
        scatter0 = axes[0, 0].scatter(self.coord_map[:, 0], self.coord_map[:, 1], 
                                 c=self.expr_vector, cmap='viridis', 
                                 s=100, alpha=0.8, edgecolors='k')
        axes[0, 0].set_title(f'Original Expression of {self.target_gene}')
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Y coordinate')
        cbar0 = plt.colorbar(scatter0, ax=axes[0, 0])
        cbar0.set_label('Expression level')
        
        # If we have an odd number of methods, clear the unused bottom-right plot
        if n_methods % 2 == 0:
            if n_methods + 1 < n_rows * n_cols:
                axes[-1, -1].axis('off')
        
        # Plot each method's result
        for i, method_key in enumerate(available_methods):
            # Get subplot position
            if i == 0 and n_cols > 1:
                row, col = 0, 1
            else:
                adjustment = 1 if n_cols > 1 else 0
                row, col = (i + adjustment) // n_cols, (i + adjustment) % n_cols
                
            method_results = self.results[method_key]
            region_mask = method_results['region_mask']
            
            # Base scatter plot of all spots
            axes[row, col].scatter(self.coord_map[:, 0], self.coord_map[:, 1], 
                                c='lightgray', s=80, alpha=0.6, edgecolors='k')
            
            # Highlight region spots
            if np.any(region_mask):
                region_coords = self.coord_map[region_mask]
                
                # Use expression as colors for region spots
                region_expr = self.expr_vector[region_mask]
                scatter = axes[row, col].scatter(region_coords[:, 0], region_coords[:, 1], 
                                             c=region_expr, cmap='viridis', 
                                             s=120, alpha=0.9, edgecolors='k')
            
            # Add execution time and region size to title
            axes[row, col].set_title(
                f"{method_results['method']}\n"
                f"Time: {method_results['execution_times']['total']:.4f} s\n"
                f"Region size: {method_results['region_size']} spots"
            )
            axes[row, col].set_xlabel('X coordinate')
            axes[row, col].set_ylabel('Y coordinate')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(OUTPUT_DIR, 'region_detection_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved region detection visualization to {save_path}")
        
        return fig
    
    def create_time_comparison_chart(self):
        """
        Create a bar chart comparing execution times.
        
        Returns:
            Tuple of (max_finding_fig, region_detection_fig)
        """
        logger.info("Creating execution time comparison charts")
        
        # Get methods and times for maximum finding
        max_methods = ['qspat_vqe', 'classical_max', 'hill_climbing', 'gradient_ascent']
        max_available = [m for m in max_methods if m in self.results]
        
        if max_available:
            max_method_names = [self.results[m]['method'] for m in max_available]
            max_times = [self.results[m]['execution_times']['total'] for m in max_available]
            
            # Create figure for maximum finding
            max_fig, max_ax = plt.subplots(figsize=(10, 6))
            bars = max_ax.bar(max_method_names, max_times)
            
            # Add time labels
            for bar in bars:
                height = bar.get_height()
                max_ax.annotate(f'{height:.4f}s',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom')
            
            max_ax.set_ylabel('Execution Time (seconds)')
            max_ax.set_title('Maximum Finding Methods - Execution Time Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            max_save_path = os.path.join(OUTPUT_DIR, 'max_finding_time_comparison.png')
            plt.savefig(max_save_path, dpi=300, bbox_inches='tight')
        else:
            max_fig = None
        
        # Get methods and times for region detection
        region_methods = ['qspat_qaoa', 'threshold_region', 'kmeans_region', 'dbscan_region', 'gmm_region']
        region_available = [m for m in region_methods if m in self.results]
        
        if region_available:
            region_method_names = [self.results[m]['method'] for m in region_available]
            region_times = [self.results[m]['execution_times']['total'] for m in region_available]
            
            # Create figure for region detection
            region_fig, region_ax = plt.subplots(figsize=(10, 6))
            bars = region_ax.bar(region_method_names, region_times)
            
            # Add time labels
            for bar in bars:
                height = bar.get_height()
                region_ax.annotate(f'{height:.4f}s',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            region_ax.set_ylabel('Execution Time (seconds)')
            region_ax.set_title('Region Detection Methods - Execution Time Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            region_save_path = os.path.join(OUTPUT_DIR, 'region_detection_time_comparison.png')
            plt.savefig(region_save_path, dpi=300, bbox_inches='tight')
        else:
            region_fig = None
        
        return max_fig, region_fig
    
    def generate_markdown_report(self):
        """
        Generate a detailed Markdown report of benchmark results.
        
        Returns:
            Markdown string
        """
        logger.info("Generating Markdown report")
        
        # Get timestamp for report
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start report
        report = [
            "# Qspat Benchmarking Report",
            f"Generated: {timestamp}\n",
            "This report compares the performance of Qspat's quantum algorithms against classical methods for spatial transcriptomics analysis.\n",
            "## Overview",
            f"- Target Gene: {self.target_gene}",
            f"- Data Source: {self.data_path}",
            f"- Number of Spots: {len(self.expr_vector)}",
            "\n## Benchmarks Performed\n"
        ]
        
        # Add table of benchmarks
        bench_table = [
            "| Method Type | Algorithm | Description |",
            "| --- | --- | --- |",
        ]
        
        # Maximum finding methods
        if 'qspat_vqe' in self.results:
            bench_table.append("| Maximum Finding | Qspat VQE | Quantum Variational Eigensolver |")
        if 'classical_max' in self.results:
            bench_table.append("| Maximum Finding | numpy.argmax | Direct classical maximum detection |")
        if 'hill_climbing' in self.results:
            bench_table.append("| Maximum Finding | Hill Climbing | Local search optimization |")
        if 'gradient_ascent' in self.results:
            bench_table.append("| Maximum Finding | Gradient Ascent | Numerical gradient-based optimization |")
        
        # Region detection methods
        if 'qspat_qaoa' in self.results:
            bench_table.append("| Region Detection | Qspat QAOA | Quantum Approximate Optimization Algorithm |")
        if 'threshold_region' in self.results:
            bench_table.append("| Region Detection | Thresholding | Simple expression thresholding |")
        if 'kmeans_region' in self.results:
            bench_table.append("| Region Detection | K-means | K-means clustering |")
        if 'dbscan_region' in self.results:
            bench_table.append("| Region Detection | DBSCAN | Density-based clustering |")
        if 'gmm_region' in self.results:
            bench_table.append("| Region Detection | GMM | Gaussian Mixture Model |")
        
        report.extend(bench_table)
        report.append("\n")
        
        # Maximum Finding Results
        report.append("## Maximum Finding Results\n")
        
        max_methods = ['qspat_vqe', 'classical_max', 'hill_climbing', 'gradient_ascent']
        max_available = [m for m in max_methods if m in self.results]
        
        if max_available:
            # Create results table
            max_table = [
                "| Method | Location | Expression Value | Execution Time |",
                "| --- | --- | --- | --- |",
            ]
            
            for method in max_available:
                r = self.results[method]
                coords = r['max_coords']
                coords_str = f"({coords[0]:.2f}, {coords[1]:.2f})"
                max_table.append(f"| {r['method']} | {coords_str} | {r['max_value']:.4f} | {r['execution_times']['total']:.4f}s |")
            
            report.extend(max_table)
            report.append("\n")
            
            # Add visualization reference
            report.append("### Visualization\n")
            report.append("![Maximum Finding Comparison](max_finding_comparison.png)\n")
            report.append("![Maximum Finding Time Comparison](max_finding_time_comparison.png)\n")
            
            # Add analysis
            report.append("### Analysis\n")
            
            # Find fastest method
            fastest_method = min(max_available, key=lambda m: self.results[m]['execution_times']['total'])
            fastest_time = self.results[fastest_method]['execution_times']['total']
            
            # Find most accurate method (compared to classical_max which is the ground truth)
            if 'classical_max' in self.results:
                ground_truth_idx = self.results['classical_max']['max_idx']
                accuracy = {}
                for method in max_available:
                    if method != 'classical_max':
                        if self.results[method]['max_idx'] == ground_truth_idx:
                            accuracy[method] = 1.0
                        else:
                            # Calculate relative error in expression value
                            true_val = self.results['classical_max']['max_value']
                            method_val = self.results[method]['max_value']
                            accuracy[method] = method_val / true_val if true_val > 0 else 0
                
                if accuracy:
                    most_accurate = max(accuracy.items(), key=lambda x: x[1])
                    report.append(f"- Fastest method: **{self.results[fastest_method]['method']}** ({fastest_time:.4f}s)")
                    report.append(f"- Most accurate method: **{self.results[most_accurate[0]]['method']}** (accuracy: {most_accurate[1]:.2%})")
                    
                    # Compare Qspat VQE with best classical
                    if 'qspat_vqe' in self.results:
                        vqe_time = self.results['qspat_vqe']['execution_times']['total']
                        best_classical = min([m for m in max_available if m != 'qspat_vqe'], 
                                           key=lambda m: self.results[m]['execution_times']['total'])
                        best_classical_time = self.results[best_classical]['execution_times']['total']
                        
                        speedup = best_classical_time / vqe_time if vqe_time > 0 else float('inf')
                        if speedup > 1:
                            report.append(f"- Qspat VQE is **{speedup:.2f}x faster** than the best classical method ({self.results[best_classical]['method']})")
                        else:
                            report.append(f"- Best classical method ({self.results[best_classical]['method']}) is **{1/speedup:.2f}x faster** than Qspat VQE")
                        
                        # Compare accuracy
                        if 'qspat_vqe' in accuracy:
                            vqe_accuracy = accuracy['qspat_vqe']
                            report.append(f"- Qspat VQE found a solution with {vqe_accuracy:.2%} of the maximum expression value")
            else:
                report.append(f"- Fastest method: **{self.results[fastest_method]['method']}** ({fastest_time:.4f}s)")
        else:
            report.append("No maximum finding benchmarks were performed.")
        
        # Region Detection Results
        report.append("\n## Region Detection Results\n")
        
        region_methods = ['qspat_qaoa', 'threshold_region', 'kmeans_region', 'dbscan_region', 'gmm_region']
        region_available = [m for m in region_methods if m in self.results]
        
        if region_available:
            # Create results table
            region_table = [
                "| Method | Region Size | Avg. Expression | Execution Time |",
                "| --- | --- | --- | --- |",
            ]
            
            for method in region_available:
                r = self.results[method]
                region_table.append(f"| {r['method']} | {r['region_size']} spots | {r['region_avg_expr']:.4f} | {r['execution_times']['total']:.4f}s |")
            
            report.extend(region_table)
            report.append("\n")
            
            # Add visualization reference
            report.append("### Visualization\n")
            report.append("![Region Detection Comparison](region_detection_comparison.png)\n")
            report.append("![Region Detection Time Comparison](region_detection_time_comparison.png)\n")
            
            # Add analysis
            report.append("### Analysis\n")
            
            # Find fastest method
            fastest_method = min(region_available, key=lambda m: self.results[m]['execution_times']['total'])
            fastest_time = self.results[fastest_method]['execution_times']['total']
            
            # Find method with highest average expression
            highest_expr_method = max(region_available, key=lambda m: self.results[m]['region_avg_expr'])
            highest_expr = self.results[highest_expr_method]['region_avg_expr']
            
            report.append(f"- Fastest method: **{self.results[fastest_method]['method']}** ({fastest_time:.4f}s)")
            report.append(f"- Method with highest average expression: **{self.results[highest_expr_method]['method']}** ({highest_expr:.4f})")
            
            # Compare Qspat QAOA with best classical
            if 'qspat_qaoa' in self.results:
                qaoa_time = self.results['qspat_qaoa']['execution_times']['total']
                best_classical = min([m for m in region_available if m != 'qspat_qaoa'], 
                                   key=lambda m: self.results[m]['execution_times']['total'])
                best_classical_time = self.results[best_classical]['execution_times']['total']
                
                speedup = best_classical_time / qaoa_time if qaoa_time > 0 else float('inf')
                if speedup > 1:
                    report.append(f"- Qspat QAOA is **{speedup:.2f}x faster** than the best classical method ({self.results[best_classical]['method']})")
                else:
                    report.append(f"- Best classical method ({self.results[best_classical]['method']}) is **{1/speedup:.2f}x faster** than Qspat QAOA")
                
                # Compare region quality
                qaoa_expr = self.results['qspat_qaoa']['region_avg_expr']
                best_expr_classical = max([m for m in region_available if m != 'qspat_qaoa'], 
                                       key=lambda m: self.results[m]['region_avg_expr'])
                best_expr_classical_val = self.results[best_expr_classical]['region_avg_expr']
                
                expr_ratio = qaoa_expr / best_expr_classical_val if best_expr_classical_val > 0 else float('inf')
                if expr_ratio > 1:
                    report.append(f"- Qspat QAOA finds a region with **{expr_ratio:.2f}x higher** average expression than the best classical method ({self.results[best_expr_classical]['method']})")
                else:
                    report.append(f"- Best classical method ({self.results[best_expr_classical]['method']}) finds a region with **{1/expr_ratio:.2f}x higher** average expression than Qspat QAOA")
        else:
            report.append("No region detection benchmarks were performed.")
        
        # Overall Conclusion
        report.append("\n## Conclusion\n")
        
        if max_available and region_available:
            # Add general conclusion about quantum vs classical performance
            report.append("### Strengths of Quantum Approaches\n")
            report.append("The Qspat quantum algorithms showed the following strengths:\n")
            
            quantum_pros = []
            
            # VQE analysis
            if 'qspat_vqe' in self.results:
                quantum_pros.append("- VQE can potentially scale better for larger datasets with a logarithmic qubit requirement")
                quantum_pros.append("- VQE's probabilistic approach allows for exploration of multiple high-expression regions simultaneously")
            
            # QAOA analysis
            if 'qspat_qaoa' in self.results:
                quantum_pros.append("- QAOA directly optimizes for both expression level and spatial coherence in a single objective function")
                quantum_pros.append("- QAOA can represent complex spatial relationships that may be hard to encode in classical algorithms")
            
            if quantum_pros:
                report.extend(quantum_pros)
            else:
                report.append("- More detailed analysis required with larger datasets")
            
            report.append("\n### Strengths of Classical Approaches\n")
            report.append("The classical algorithms showed the following strengths:\n")
            
            classical_pros = [
                "- Classical methods are currently more resource-efficient for small to medium-sized datasets",
                "- Direct optimization approaches are very effective for these specific benchmarking tasks",
                "- Classical clustering methods provide a good balance between computation speed and region quality"
            ]
            
            report.extend(classical_pros)
            
            report.append("\n### Future Work\n")
            report.append("Based on these benchmarks, the following future work is recommended:\n")
            
            future_work = [
                "- Benchmark with larger, real-world spatial transcriptomics datasets",
                "- Implement more advanced quantum encoding methods to improve scaling",
                "- Explore hybrid quantum-classical approaches that leverage strengths of both paradigms",
                "- Optimize the Qspat framework for better performance with quantum simulators and real hardware",
                "- Add multi-gene analysis capabilities to both quantum and classical methods"
            ]
            
            report.extend(future_work)
        else:
            report.append("Insufficient data to draw comprehensive conclusions. Please run both maximum finding and region detection benchmarks.")
        
        # Join report lines
        markdown_report = "\n".join(report)
        
        # Save report to file
        report_path = os.path.join(OUTPUT_DIR, 'benchmark_report.md')
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Saved benchmark report to {report_path}")
        
        return markdown_report

def main():
    """Run the benchmark script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmarks comparing Qspat to classical methods')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, help='Path to expression data file')
    parser.add_argument('--gene', default=DEFAULT_TARGET_GENE, help='Target gene to analyze')
    parser.add_argument('--max_spots', type=int, default=DEFAULT_MAX_SPOTS, help='Maximum number of spots')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory for results')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--max_only', action='store_true', help='Run only maximum finding benchmarks')
    group.add_argument('--region_only', action='store_true', help='Run only region detection benchmarks')
    
    args = parser.parse_args()
    
    # Update output directory if specified
    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create benchmark runner
    benchmark = Benchmark(
        data_path=args.data,
        target_gene=args.gene,
        max_spots=args.max_spots
    )
    
    # Load data
    benchmark.load_data()
    
    # Run maximum finding benchmarks
    if not args.region_only:
        # Qspat VQE
        benchmark.run_qspat_vqe()
        
        # Classical methods
        benchmark.run_classical_max()
        benchmark.run_hill_climbing()
        benchmark.run_gradient_ascent()
    
    # Run region detection benchmarks
    if not args.max_only:
        # Qspat QAOA
        benchmark.run_qspat_qaoa()
        
        # Classical methods
        benchmark.run_threshold_region()
        benchmark.run_kmeans_region()
        benchmark.run_dbscan_region()
        benchmark.run_gmm_region()
    
    # Create visualizations
    if not args.region_only:
        benchmark.visualize_max_finding_results()
    
    if not args.max_only:
        benchmark.visualize_region_detection_results()
    
    # Create time comparison charts
    benchmark.create_time_comparison_chart()
    
    # Generate report
    benchmark.generate_markdown_report()
    
    logger.info(f"Benchmarking complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()