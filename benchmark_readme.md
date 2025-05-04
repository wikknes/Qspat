# Qspat Framework Benchmarking Tool

This benchmarking tool compares the quantum algorithms in Qspat against classical approaches for spatial transcriptomics analysis. The tool evaluates both maximum expression detection (comparing VQE) and region detection (comparing QAOA) against traditional classical methods.

## Overview

The benchmark compares:

### Maximum Expression Detection (VQE Comparisons)
- **Qspat VQE**: Quantum Variational Eigensolver
- **Classical Maximum**: Direct numpy.argmax approach 
- **Hill Climbing**: Local search optimization from multiple starting points
- **Gradient Ascent**: Continuous optimization with interpolation

### High Expression Region Detection (QAOA Comparisons)
- **Qspat QAOA**: Quantum Approximate Optimization Algorithm
- **Thresholding**: Simple expression value thresholding
- **K-means Clustering**: Feature-based clustering with k=2
- **DBSCAN**: Density-based spatial clustering
- **Gaussian Mixture Model (GMM)**: Probabilistic model-based clustering

## Requirements

- All Qspat dependencies
- Additional packages:
  - scikit-learn (for classical clustering algorithms)
  - scipy (for interpolation and distance calculations)
  - matplotlib (for visualization)

## Usage

Run the benchmark with default settings:

```bash
python benchmark.py
```

Customize the benchmark:

```bash
python benchmark.py --data path/to/data.csv --gene GeneOfInterest --max_spots 32 --output results_dir
```

Run only specific benchmark types:

```bash
# Only run maximum finding benchmarks
python benchmark.py --max_only

# Only run region detection benchmarks
python benchmark.py --region_only
```

## Command Line Arguments

- `--data`: Path to expression data file (default: `data/synthetic_data.csv`)
- `--gene`: Target gene to analyze (default: `Gene2`)
- `--max_spots`: Maximum number of spots to consider (default: `16`)
- `--output`: Output directory for results (default: `benchmark_results/`)
- `--max_only`: Run only maximum finding benchmarks
- `--region_only`: Run only region detection benchmarks

## Output

The benchmark produces the following in the output directory:

1. **Visualizations**:
   - `max_finding_comparison.png`: Visual comparison of maximum finding methods
   - `region_detection_comparison.png`: Visual comparison of region detection methods
   - `max_finding_time_comparison.png`: Execution time comparison for maximum finding
   - `region_detection_time_comparison.png`: Execution time comparison for region detection

2. **Report**:
   - `benchmark_report.md`: A comprehensive markdown report with results, comparisons, and analysis

## Interpreting Results

The benchmark evaluates algorithms on two key metrics:

1. **Execution Time**: How fast the algorithm runs on the given dataset
2. **Result Quality**:
   - For maximum finding: Proximity to the true maximum and expression value
   - For region detection: Region size, average expression, and spatial coherence

The report provides insights into the trade-offs between quantum and classical approaches and makes recommendations for when each approach might be most suitable.

## Example Report

The benchmark report includes:

- Tables of numerical results
- Performance comparisons
- Visual comparisons
- Analysis of strengths and weaknesses
- Recommendations for future work

## Extending the Benchmark

To add a new classical method:

1. Add a new method in the `Benchmark` class (e.g., `run_new_method()`)
2. Ensure it follows the same return structure as existing methods
3. Add the method to the visualization methods
4. Add the method to the report generation section
5. Update the `main()` function to call the new method

## Limitations

- The benchmark currently runs on simulated quantum algorithms
- Performance on larger datasets may vary
- The provided classical methods are not necessarily optimized for maximum performance
- Execution times should be interpreted with caution on different hardware