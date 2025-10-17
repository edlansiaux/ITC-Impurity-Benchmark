# ITC-Impurity-Benchmark: Benchmarking Impurity Metrics for Decision Trees

This repository contains the full implementation of the ITC (Integrated Tsallis Combination) metric and an extensive benchmarking of 16 different impurity metrics.

## 📋 Project Structure

```yaml
ITC-Impurity-Benchmark/
├── data/ # Dataset loading and management
├── metrics/ # Implementation of all impurity metrics
├── tree/ # Decision tree implementation
├── benchmarks/ # Benchmarking scripts
├── results/ # Experimental results
├── utils/ # Utilities and visualization
├── scripts/ # Utilitary Scripts
├── notebooks/ # Analysis Notebooks
└── tests/ # Unitary tests
└── run_all_benchmarks.py # Main script
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/username/ITC-Impurity-Benchmark.git
cd ITC-Impurity-Benchmark
```

2. **Install depencies** :
   
```bash
pip install -r requirements.txt
```

## Use

Run the principal benchmark:
```bash
python benchmarks/run_benchmarks.py
```
Run individual metrics benchmark:
```bash
from benchmarks.individual_metrics import run_individual_benchmarks
results, aggregated = run_individual_benchmarks()
```
## 📊 Implemented Metrics

### Classic
- Gini - Gini Impurity
- Shannon - Shannon Entropy
- Misclassification - Error rate
  
### Parametric
- Rényi (α=0.5, 2.0) - Rényi Entropy
- Tsallis (α=0.5, 1.3, 2.0) - Tsallis Entropy
- Kumaraswamy - Kumaraswamy Distribution

### Probabilist
- Cross Entropy - Cross Entropy
- KL Divergence - Kullback-Leibler Divergence
- JS Divergence - Jensen-Shannon Divergence

### Distance-based
- Hellinger - Hellinger Distance
- Energy - Energetic Distance
- Polarization - Polarization Index

### Theoric
- Bregman - Bregman Divergence

### Hybrides
- ITC - Integrated Tsallis Combination
- ITC variants - Variations paramétriques
- Shannon-Polarization - Hybridation alternative
- Tsallis-Hellinger - Hybridation alternative

## 📈 Results
Results are stored in the results/ folder:
- raw_results/ - Detailed results per run
- tables/ - Aggregated tables
- figures/ - Visualizations

### Key Metrics Evaluated
- Accuracy - Classification accuracy  
- Tree Depth - Depth of the tree  
- Tree Size - Number of nodes  
- Training Time - Training time  
- Stability - Run-to-run stability  

### 🔬 Main Findings
Our ITC metric shows:  
✅ +8.7% reduction in depth vs Gini  
✅ +5.6% improvement in stability  
✅ +2.1% average accuracy gain  
✅ Computational complexity comparable to Gini  

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
## 🤝 Contribution
Contributions are welcome! Please:  
1. Fork the project  
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
