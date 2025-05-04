#!/usr/bin/env python
# Preprocessing module for the Quantum Spatial Transcriptomics framework

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

def load_data(expression_file, coords_file=None):
    """
    Load spatial transcriptomics data from files.
    
    Args:
        expression_file: Path to expression matrix (CSV or H5)
        coords_file: Path to coordinates file (if separate from expression)
        
    Returns:
        DataFrame with expression values and coordinates
    """
    # Determine file type and load
    if expression_file.endswith('.h5') or expression_file.endswith('.h5ad'):
        adata = sc.read_h5ad(expression_file)
        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        if 'spatial' in adata.obsm:
            coords = pd.DataFrame(
                adata.obsm['spatial'], 
                index=adata.obs_names,
                columns=['x', 'y']
            )
            df = pd.concat([coords, df], axis=1)
            return df
    else:
        # Assume CSV format
        df = pd.read_csv(expression_file, index_col=0)
    
    # If coordinates are in a separate file
    if coords_file:
        coords = pd.read_csv(coords_file, index_col=0)
        if set(coords.index) != set(df.index):
            warnings.warn("Coordinate indices don't match expression indices")
        df = pd.concat([coords[['x', 'y']], df], axis=1)
    
    return df

def filter_data(df, min_spots=3, min_expr=1):
    """
    Filter low-quality spots and low-expression genes.
    
    Args:
        df: DataFrame with gene expression
        min_spots: Minimum number of spots a gene must be expressed in
        min_expr: Minimum expression value to consider a gene expressed
        
    Returns:
        Filtered DataFrame
    """
    # Get coordinates
    coords = df[['x', 'y']]
    expr_df = df.drop(['x', 'y'], axis=1)
    
    # Find genes expressed in enough spots
    gene_counts = (expr_df > min_expr).sum(axis=0)
    keep_genes = gene_counts[gene_counts >= min_spots].index
    
    # Filter and recombine
    filtered_expr = expr_df[keep_genes]
    result = pd.concat([coords, filtered_expr], axis=1)
    
    print(f"Filtered from {expr_df.shape[1]} to {filtered_expr.shape[1]} genes")
    return result

def normalize_data(df, method='log1p'):
    """
    Normalize gene expression values.
    
    Args:
        df: DataFrame with gene expression
        method: Normalization method ('log1p', 'cpm', or 'cpm_log1p')
        
    Returns:
        DataFrame with normalized values
    """
    # Get coordinates
    coords = df[['x', 'y']]
    expr_df = df.drop(['x', 'y'], axis=1)
    
    if method == 'log1p':
        # Simple log(x+1) transformation
        norm_expr = np.log1p(expr_df)
    elif method == 'cpm':
        # Counts per million
        totals = expr_df.sum(axis=1)
        norm_expr = expr_df.div(totals, axis=0) * 1e6
    elif method == 'cpm_log1p':
        # Counts per million followed by log(x+1)
        totals = expr_df.sum(axis=1)
        cpm = expr_df.div(totals, axis=0) * 1e6
        norm_expr = np.log1p(cpm)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Recombine with coordinates
    result = pd.concat([coords, norm_expr], axis=1)
    return result

def select_features(df, n_features=10, method='var', gene_list=None):
    """
    Select features (genes) for analysis.
    
    Args:
        df: DataFrame with gene expression
        n_features: Number of features to select
        method: Selection method ('var', 'pca', or 'list')
        gene_list: List of specific genes to select (if method='list')
        
    Returns:
        DataFrame with selected features and PCA result if applicable
    """
    # Get coordinates
    coords = df[['x', 'y']]
    expr_df = df.drop(['x', 'y'], axis=1)
    
    if method == 'var':
        # Select top variable genes
        gene_var = expr_df.var(axis=0).sort_values(ascending=False)
        selected_genes = gene_var.index[:n_features]
        selected_df = expr_df[selected_genes]
        pca_result = None
        
    elif method == 'pca':
        # Perform PCA
        pca = PCA(n_components=n_features)
        pca_result = pca.fit_transform(expr_df)
        
        # Create dataframe with PCA components
        selected_df = pd.DataFrame(
            pca_result, 
            index=expr_df.index,
            columns=[f'PC{i+1}' for i in range(n_features)]
        )
        
    elif method == 'list':
        # Select specific genes
        if not gene_list:
            raise ValueError("gene_list must be provided when method='list'")
        missing = set(gene_list) - set(expr_df.columns)
        if missing:
            warnings.warn(f"Some genes not found in data: {missing}")
        selected_genes = [g for g in gene_list if g in expr_df.columns]
        selected_df = expr_df[selected_genes]
        pca_result = None
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Recombine with coordinates
    result = pd.concat([coords, selected_df], axis=1)
    return result, pca_result

def downsample_spots(df, max_spots=16, method='kmeans'):
    """
    Reduce number of spots to fit qubit budget.
    
    Args:
        df: DataFrame with gene expression
        max_spots: Maximum number of spots to keep
        method: Downsampling method ('kmeans' or 'random')
        
    Returns:
        Downsampled DataFrame and mapping to original indices
    """
    if df.shape[0] <= max_spots:
        return df, np.arange(df.shape[0])
    
    # Get coordinates and expression
    coords = df[['x', 'y']]
    expr_df = df.drop(['x', 'y'], axis=1)
    
    if method == 'kmeans':
        # K-means clustering
        kmeans = KMeans(n_clusters=max_spots, random_state=42)
        clusters = kmeans.fit_predict(expr_df)
        
        # Take spot closest to each centroid
        centers = kmeans.cluster_centers_
        selected_indices = []
        
        for i in range(max_spots):
            # Find spots in this cluster
            cluster_spots = np.where(clusters == i)[0]
            if len(cluster_spots) == 0:
                continue
                
            # Get spot nearest to centroid
            cluster_data = expr_df.iloc[cluster_spots]
            dists = ((cluster_data - centers[i])**2).sum(axis=1)
            nearest_idx = dists.index[dists.argmin()]
            selected_indices.append(nearest_idx)
            
    elif method == 'random':
        # Random sampling
        selected_indices = expr_df.sample(max_spots, random_state=42).index
    else:
        raise ValueError(f"Unknown downsampling method: {method}")
    
    # Create mapping from new indices to original
    index_map = {i: idx for i, idx in enumerate(selected_indices)}
    
    # Return downsampled data
    return df.loc[selected_indices].reset_index(), index_map

def prepare_for_quantum(df, target_gene=None, binary_threshold=None):
    """
    Prepare final vector for quantum processing.
    
    Args:
        df: DataFrame with selected features
        target_gene: Gene to analyze (if None, use all columns except x,y)
        binary_threshold: Threshold for binarization (if None, no binarization)
        
    Returns:
        Vector and coordinates mapping
    """
    # Get coordinates
    coords = df[['x', 'y']]
    
    if target_gene:
        if target_gene not in df.columns:
            raise ValueError(f"Target gene {target_gene} not found in data")
        expr_vector = df[target_gene].values
    else:
        # Use all non-coordinate columns
        expr_df = df.drop(['x', 'y'], axis=1)
        
        # If multiple columns, take mean or first principal component
        if expr_df.shape[1] > 1:
            # Use first principal component
            pca = PCA(n_components=1)
            expr_vector = pca.fit_transform(expr_df).flatten()
        else:
            expr_vector = expr_df.iloc[:, 0].values
    
    # Normalize to [0,1] range
    expr_min = expr_vector.min()
    expr_max = expr_vector.max()
    if expr_max > expr_min:  # Avoid division by zero
        expr_norm = (expr_vector - expr_min) / (expr_max - expr_min)
    else:
        expr_norm = np.zeros_like(expr_vector)
    
    # Binarize if threshold provided
    if binary_threshold is not None:
        expr_norm = (expr_norm >= binary_threshold).astype(float)
    
    # Ensure length is power of 2 by padding with zeros if needed
    n_qubits = int(np.ceil(np.log2(len(expr_norm))))
    padded_length = 2**n_qubits
    if len(expr_norm) < padded_length:
        expr_norm = np.pad(expr_norm, (0, padded_length - len(expr_norm)))
    
    # Create coordinate mapping
    coord_map = coords.values
    
    return expr_norm, coord_map, n_qubits

# Main processing pipeline
def process_data(expression_file, coords_file=None, target_gene=None, 
                 max_spots=16, n_features=10, normalize_method='log1p',
                 feature_method='var', binary_threshold=None):
    """
    Full preprocessing pipeline.
    
    Args:
        expression_file: Path to expression data
        coords_file: Path to coordinates (if separate)
        target_gene: Specific gene to analyze
        max_spots: Maximum number of spots (qubit budget)
        n_features: Number of features to select
        normalize_method: Normalization method
        feature_method: Feature selection method
        binary_threshold: Optional threshold for binarization
        
    Returns:
        Prepared vector, coordinates mapping, and number of qubits
    """
    # Load data
    print("Loading data...")
    df = load_data(expression_file, coords_file)
    
    # Filter data
    print("Filtering data...")
    df = filter_data(df)
    
    # Normalize
    print("Normalizing data...")
    df = normalize_data(df, method=normalize_method)
    
    # Feature selection
    print("Selecting features...")
    if feature_method == 'list' and target_gene:
        gene_list = [target_gene]
        df, _ = select_features(df, method='list', gene_list=gene_list)
    else:
        df, _ = select_features(df, n_features=n_features, method=feature_method)
    
    # Downsample if needed
    print("Downsampling spots...")
    df, index_map = downsample_spots(df, max_spots=max_spots)
    
    # Prepare for quantum
    print("Preparing for quantum processing...")
    expr_vector, coord_map, n_qubits = prepare_for_quantum(
        df, target_gene=target_gene, binary_threshold=binary_threshold
    )
    
    print(f"Final vector length: {len(expr_vector)} (using {n_qubits} qubits)")
    return expr_vector, coord_map, n_qubits, index_map

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess spatial transcriptomics data')
    parser.add_argument('--expression', required=True, help='Expression data file')
    parser.add_argument('--coords', help='Coordinates file (if separate)')
    parser.add_argument('--gene', help='Target gene to analyze')
    parser.add_argument('--max_spots', type=int, default=16, help='Maximum number of spots')
    parser.add_argument('--normalize', default='log1p', help='Normalization method')
    parser.add_argument('--binary', type=float, help='Binarization threshold')
    
    args = parser.parse_args()
    
    vector, coords, qubits, index_map = process_data(
        args.expression, 
        coords_file=args.coords,
        target_gene=args.gene,
        max_spots=args.max_spots,
        normalize_method=args.normalize,
        binary_threshold=args.binary
    )
    
    print(f"Processed data: vector shape {vector.shape}, {qubits} qubits needed")