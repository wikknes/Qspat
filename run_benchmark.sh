#!/bin/bash
# Script to run Qspat benchmarks and generate the report

# Create output directory if it doesn't exist
BENCHMARK_DIR="benchmark_results"
mkdir -p $BENCHMARK_DIR

# Print banner
echo "=============================================="
echo "     Qspat Quantum vs. Classical Benchmark    "
echo "=============================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas, matplotlib, scipy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install numpy pandas matplotlib scipy scikit-learn
fi

# Run the benchmark
echo "Running benchmark..."
echo "This may take a few minutes..."
python3 benchmark.py --output $BENCHMARK_DIR

# Check if benchmark completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Benchmark failed to run"
    exit 1
fi

# Print completion message
echo
echo "Benchmark completed successfully!"
echo "Results saved to: $BENCHMARK_DIR/"
echo
echo "Report: $BENCHMARK_DIR/benchmark_report.md"
echo "Visualizations:"
echo "  - $BENCHMARK_DIR/max_finding_comparison.png"
echo "  - $BENCHMARK_DIR/region_detection_comparison.png"
echo "  - $BENCHMARK_DIR/max_finding_time_comparison.png"
echo "  - $BENCHMARK_DIR/region_detection_time_comparison.png"
echo

# Try to open the report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening report..."
    open "$BENCHMARK_DIR/benchmark_report.md" 2>/dev/null || echo "Could not open report automatically"
fi

echo "Thank you for using Qspat Benchmarking Tool"
echo "=============================================="