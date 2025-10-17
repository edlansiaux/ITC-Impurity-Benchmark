import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import resample
import time
from tqdm import tqdm
import joblib

from tree.decision_tree import DecisionTree
from metrics import ALL_METRICS, METRICS_BY_CATEGORY
from data.data_loader import DataLoader

class IndividualMetricsBenchmark:
    def __init__(self, n_runs=10, test_size=0.3, random_state=42):
        self.n_runs = n_runs
        self.test_size = test_size
        self.random_state = random_state
        self.results = []
        
    def run_benchmark(self, datasets, metrics=None):
        if metrics is None:
            metrics = ALL_METRICS.keys()
            
        all_results = []
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\nBenchmark sur le dataset: {dataset_name}")
            print(f"Taille: {X.shape}, Classes: {len(np.unique(y))}")
            
            dataset_results = self._benchmark_dataset(X, y, dataset_name, metrics)
            all_results.extend(dataset_results)
            
        return pd.DataFrame(all_results)
    
    def _benchmark_dataset(self, X, y, dataset_name, metrics):
        dataset_results = []
        n_classes = len(np.unique(y))
        
        for run in tqdm(range(self.n_runs), desc=f"Runs {dataset_name}"):
            # Split train-test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, 
                random_state=self.random_state + run,
                stratify=y
            )
            
            for metric_name in tqdm(metrics, desc="Métriques", leave=False):
                try:
                    # Entraînement de l'arbre
                    start_time = time.time()
                    tree = DecisionTree(
                        impurity_measure=metric_name,
                        max_depth=20,
                        min_samples_split=2,
                        random_state=self.random_state + run
                    )
                    tree.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Prédictions
                    y_pred = tree.predict(X_test)
                    y_pred_proba = self._get_prediction_probabilities(tree, X_test, n_classes)
                    
                    # Métriques d'évaluation
                    accuracy = accuracy_score(y_test, y_pred)
                    test_log_loss = log_loss(y_test, y_pred_proba) if n_classes > 1 else 0
                    tree_depth = tree.get_tree_depth()
                    tree_size = tree.get_tree_size()
                    
                    # Stockage des résultats
                    result = {
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'run': run,
                        'accuracy': accuracy,
                        'log_loss': test_log_loss,
                        'tree_depth': tree_depth,
                        'tree_size': tree_size,
                        'training_time': training_time,
                        'n_features': X.shape[1],
                        'n_classes': n_classes,
                        'n_samples': len(X)
                    }
                    
                    dataset_results.append(result)
                    
                except Exception as e:
                    print(f"Erreur avec {metric_name} sur {dataset_name}: {e}")
                    continue
                    
        return dataset_results
    
    def _get_prediction_probabilities(self, tree, X, n_classes):
        """Estime les probabilités de prédiction (simplifié)"""
        # Implémentation basique - dans la pratique, il faudrait modifier l'arbre
        # pour stocker les distributions de classes aux feuilles
        predictions = tree.predict(X)
        probas = np.zeros((len(X), n_classes))
        for i, pred in enumerate(predictions):
            probas[i, pred] = 1.0
        return probas
    
    def aggregate_results(self, results_df):
        """Agrège les résultats sur les différentes runs"""
        aggregated = results_df.groupby(['dataset', 'metric']).agg({
            'accuracy': ['mean', 'std'],
            'log_loss': ['mean', 'std'],
            'tree_depth': ['mean', 'std'],
            'tree_size': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        # Aplatir les colonnes
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        return aggregated.reset_index()

def run_individual_benchmarks():
    """Fonction principale pour lancer le benchmark des métriques individuelles"""
    # Chargement des données
    data_loader = DataLoader()
    data_loader.load_sklearn_datasets()
    synthetic_datasets = data_loader.create_synthetic_datasets()
    all_datasets = {**data_loader.get_all_datasets(), **synthetic_datasets}
    
    # Benchmark
    benchmark = IndividualMetricsBenchmark(n_runs=5)  # Réduire pour les tests
    results = benchmark.run_benchmark(all_datasets)
    
    # Sauvegarde
    results.to_csv('results/raw_results/individual_metrics_results.csv', index=False)
    
    # Agrégation
    aggregated = benchmark.aggregate_results(results)
    aggregated.to_csv('results/tables/individual_metrics_aggregated.csv', index=False)
    
    return results, aggregated

if __name__ == "__main__":
    results, aggregated = run_individual_benchmarks()
    print("Benchmark terminé!")
    print(aggregated.head())
