import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm

from tree.decision_tree import DecisionTree
from metrics import HYBRID_METRICS
from data.data_loader import DataLoader

class HybridComparisonBenchmark:
    def __init__(self, n_runs=10, test_size=0.3, random_state=42):
        self.n_runs = n_runs
        self.test_size = test_size
        self.random_state = random_state
        
    def run_hybrid_comparison(self, datasets, hybrid_metrics=None):
        if hybrid_metrics is None:
            hybrid_metrics = HYBRID_METRICS.keys()
            
        # Métriques de référence
        reference_metrics = ['gini', 'shannon', 'tsallis_1.3', 'js_divergence']
        all_metrics = reference_metrics + list(hybrid_metrics)
        
        all_results = []
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\nComparaison hybride sur: {dataset_name}")
            
            dataset_results = self._benchmark_dataset(X, y, dataset_name, all_metrics)
            all_results.extend(dataset_results)
            
        return pd.DataFrame(all_results)
    
    def _benchmark_dataset(self, X, y, dataset_name, metrics):
        dataset_results = []
        
        for run in tqdm(range(self.n_runs), desc=f"Runs {dataset_name}"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                random_state=self.random_state + run,
                stratify=y
            )
            
            for metric_name in tqdm(metrics, desc="Métriques", leave=False):
                try:
                    tree = DecisionTree(
                        impurity_measure=metric_name,
                        max_depth=20,
                        min_samples_split=2,
                        random_state=self.random_state + run
                    )
                    
                    start_time = time.time()
                    tree.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    y_pred = tree.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    tree_depth = tree.get_tree_depth()
                    tree_size = tree.get_tree_size()
                    
                    result = {
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'run': run,
                        'accuracy': accuracy,
                        'tree_depth': tree_depth,
                        'tree_size': tree_size,
                        'training_time': training_time,
                        'category': 'hybrid' if metric_name in HYBRID_METRICS else 'reference'
                    }
                    
                    dataset_results.append(result)
                    
                except Exception as e:
                    print(f"Erreur avec {metric_name}: {e}")
                    continue
                    
        return dataset_results
    
    def analyze_hybrid_performance(self, results_df):
        """Analyse comparative des performances des métriques hybrides"""
        analysis_results = []
        
        for dataset in results_df['dataset'].unique():
            dataset_data = results_df[results_df['dataset'] == dataset]
            
            # Performance de référence (Gini)
            gini_perf = dataset_data[dataset_data['metric'] == 'gini']['accuracy'].mean()
            
            for metric in dataset_data['metric'].unique():
                if metric == 'gini':
                    continue
                    
                metric_data = dataset_data[dataset_data['metric'] == metric]
                metric_mean = metric_data['accuracy'].mean()
                improvement = ((metric_mean - gini_perf) / gini_perf) * 100
                
                depth_reduction = (
                    (dataset_data[dataset_data['metric'] == 'gini']['tree_depth'].mean() - 
                     metric_data['tree_depth'].mean()) / 
                    dataset_data[dataset_data['metric'] == 'gini']['tree_depth'].mean() * 100
                )
                
                analysis_results.append({
                    'dataset': dataset,
                    'metric': metric,
                    'accuracy_mean': metric_mean,
                    'improvement_vs_gini': improvement,
                    'depth_reduction_vs_gini': depth_reduction,
                    'category': 'hybrid' if metric in HYBRID_METRICS else 'reference'
                })
                
        return pd.DataFrame(analysis_results)

def run_hybrid_comparison():
    """Fonction principale pour la comparaison des métriques hybrides"""
    data_loader = DataLoader()
    data_loader.load_sklearn_datasets()
    datasets = data_loader.get_all_datasets()
    
    benchmark = HybridComparisonBenchmark(n_runs=5)
    results = benchmark.run_hybrid_comparison(datasets)
    
    # Sauvegarde
    results.to_csv('results/raw_results/hybrid_comparison_results.csv', index=False)
    
    # Analyse
    analysis = benchmark.analyze_hybrid_performance(results)
    analysis.to_csv('results/tables/hybrid_performance_analysis.csv', index=False)
    
    return results, analysis

if __name__ == "__main__":
    results, analysis = run_hybrid_comparison()
    print("Comparaison hybride terminée!")
    print(analysis.sort_values('improvement_vs_gini', ascending=False).head(10))
