# ITC-Impurity-Benchmark

A comprehensive benchmarking framework for evaluating 23 impurity metrics in decision tree construction, including the novel **ITC (Integrated Tsallis Combination)** metric.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This repository provides a complete implementation and evaluation of impurity metrics for decision tree learning, with a focus on the **ITC metric** - a novel hybrid approach combining Tsallis entropy with polarization measures.

## 📊 Key Results

Based on comprehensive benchmarks across 7 datasets with 5 runs each:

### Top 5 Metrics by Average Accuracy

| Rank | Metric | Accuracy | Performance |
|------|--------|----------|-------------|
| 🥇 | **Tsallis (α=0.5)** | **91.17%** | Best overall |
| 🥈 | **Rényi (α=0.5)** | **90.85%** | Excellent |
| 🥉 | **Shannon-Polarization** | **90.64%** | Strong hybrid |
| 4 | Shannon Entropy | 90.57% | Classical baseline |
| 5 | Tsallis (α=2.0) / Gini | 90.20% | Traditional methods |

### ITC Performance Highlights

- **ITC (Standard)**: 88.38% accuracy
- **ITC (α=1.3)**: 89.16% accuracy  
- **ITC (α=1.7)**: 88.59% accuracy
- **Optimal parameters**: α=2.0, β=4.5, γ=0.4 (Score: 0.8882)

### Statistical Significance

- **Friedman Test**: χ² = 3.89, p-value = 0.692 (no significant global differences)
- **Effect Sizes vs Gini**:
  - Tsallis (α=0.5): Cohen's d = 0.458 (small effect)
  - Rényi (α=0.5): Cohen's d = 0.262 (small effect)
  - Shannon-Polarization: Cohen's d = 0.161 (negligible)

## 🏗️ Repository Structure

```
ITC-Impurity-Benchmark/
├── data/                  # Dataset loading and management
├── metrics/              # Implementation of all 23 impurity metrics
├── tree/                 # Custom decision tree implementation
├── benchmarks/           # Benchmarking scripts and experiments
│   ├── individual_metrics.py
│   ├── hybrid_comparison.py
│   └── sensitivity_analysis.py
├── results/              # Experimental results
│   ├── raw_results/     # Detailed per-run results
│   ├── tables/          # Aggregated performance tables
│   └── figures/         # Visualization outputs
├── utils/               # Utilities and visualization tools
├── scripts/             # Utility scripts
├── notebooks/           # Analysis notebooks
├── tests/               # Unit tests
└── run_all_benchmarks.py  # Main execution script
```

## 📈 Evaluated Metrics

### Classical Metrics
- **Gini Impurity** - Traditional baseline
- **Shannon Entropy** - Information-theoretic approach
- **Misclassification Error** - Simple error-based metric

### Generalized Entropy Measures
- **Rényi Entropy** (α=0.5, 2.0)
- **Tsallis Entropy** (α=0.5, 1.3, 2.0)
- **Normalized Tsallis** (α=1.3)

### Divergence-Based Metrics
- **Cross Entropy**
- **KL Divergence** (Kullback-Leibler)
- **JS Divergence** (Jensen-Shannon)
- **Bregman Divergence** (squared, entropy)

### Distance-Based Metrics
- **Hellinger Distance**
- **Energy Distance**

### Specialized Metrics
- **Kumaraswamy Distribution**
- **Polarization Index** (α=3.5)

### Hybrid Metrics
- **ITC** (Integrated Tsallis Combination)
  - Standard ITC
  - ITC variants (α=1.3, 1.7)
- **Shannon-Polarization**
- **Tsallis-Hellinger**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/edlansiaux/ITC-Impurity-Benchmark.git
cd ITC-Impurity-Benchmark

# Install dependencies
pip install -r requirements.txt
```

### Running Benchmarks

**Full benchmark suite** (all experiments):
```bash
python run_all_benchmarks.py
```

**Individual components**:

```python
# 1. Individual metrics benchmark
from benchmarks.individual_metrics import run_individual_benchmarks
results, aggregated = run_individual_benchmarks()

# 2. Hybrid comparison
from benchmarks.hybrid_comparison import run_hybrid_comparison
hybrid_results = run_hybrid_comparison()

# 3. Sensitivity analysis
from benchmarks.sensitivity_analysis import run_sensitivity_analysis
sensitivity_results = run_sensitivity_analysis()
```

## 📊 Datasets

The benchmark evaluates metrics on:

### Scikit-learn Datasets
- **Iris** (150 samples, 4 features, 3 classes)
- **Wine** (178 samples, 13 features, 3 classes)
- **Breast Cancer** (569 samples, 30 features, 2 classes)
- **Digits** (1797 samples, 64 features, 10 classes)

### Synthetic Datasets
- **Binary Balanced** (1000 samples, 10 features, 2 classes)
- **Binary Imbalanced** (1000 samples, 10 features, 2 classes, 30:70 ratio)
- **Multiclass** (1500 samples, 12 features, 4 classes)

## 📉 Evaluation Metrics

Each impurity metric is evaluated on:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Classification accuracy on test set |
| **Tree Depth** | Average depth of the decision tree |
| **Tree Size** | Total number of nodes |
| **Training Time** | Time to train the model (seconds) |
| **Stability** | Consistency across multiple runs (std dev) |

## 🔬 Results Details

### Performance by Dataset

Results stored in `results/tables/`:
- `individual_results.csv` - Complete benchmark results
- `hybrid_comparison.csv` - Hybrid metrics comparison
- `statistical_tests.csv` - Statistical significance tests
- `sensitivity_analysis.csv` - Parameter sensitivity results

### Visualizations

Generated figures in `results/figures/`:
- Performance comparison charts
- Parameter sensitivity surfaces
- Statistical analysis plots
- Tree structure visualizations

## 🔧 ITC Metric Details

The **Integrated Tsallis Combination (ITC)** metric is defined as:

```
ITC(p) = α · Tsallis_q(p) + β · Polarization(p) + γ · Balance(p)
```

Where:
- `Tsallis_q(p)` is the Tsallis entropy with parameter q
- `Polarization(p)` measures class separation
- `Balance(p)` encourages balanced splits
- `α, β, γ` are weighting parameters

**Optimal parameters found**:
- α = 2.0 (Tsallis entropy weight)
- β = 4.5 (Polarization weight)  
- γ = 0.4 (Balance weight)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{itc2025,
  title={ITC: A Novel Hybrid Impurity Measure for Optimal Decision Tree Construction},
  author={Edouard Lansiaux},
  journal={preprint},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Edouard Lansiaux**

- GitHub: [@edlansiaux](https://github.com/edlansiaux)
- Repository: [ITC-Impurity-Benchmark](https://github.com/edlansiaux/ITC-Impurity-Benchmark)

## 🙏 Acknowledgments

- Scikit-learn for providing the base datasets
- NumPy and Pandas communities for excellent scientific computing tools
- All contributors to information theory and decision tree research

---

**Last Updated**: November 1, 2025  
**Benchmark Duration**: ~2h 30min for full suite  
**Python Version**: 3.8+
