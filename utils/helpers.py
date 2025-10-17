import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import pickle
import os
from typing import Any, Dict, List, Optional

class BenchmarkHelpers:
    @staticmethod
    def ensure_directory(path):
        """S'assure qu'un dossier existe"""
        os.makedirs(path, exist_ok=True)
        
    @staticmethod
    def save_results(results, filename, format='csv'):
        """Sauvegarde les résultats dans différents formats"""
        BenchmarkHelpers.ensure_directory(os.path.dirname(filename))
        
        if format == 'csv':
            if isinstance(results, pd.DataFrame):
                results.to_csv(filename, index=False)
            else:
                pd.DataFrame(results).to_csv(filename, index=False)
                
        elif format == 'json':
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        elif format == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
                
    @staticmethod
    def load_results(filename, format='csv'):
        """Charge les résultats depuis différents formats"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Fichier {filename} non trouvé")
            
        if format == 'csv':
            return pd.read_csv(filename)
        elif format == 'json':
            with open(filename, 'r') as f:
                return json.load(f)
        elif format == 'pkl':
            with open(filename, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def calculate_confidence_interval(data, confidence=0.95):
        """Calcule l'intervalle de confiance"""
        if len(data) == 0:
            return 0, 0, 0
            
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        if n <= 1:
            return mean, mean, mean
            
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * (std / np.sqrt(n))
        
        return mean, mean - margin_error, mean + margin_error
    
    @staticmethod
    def format_duration(seconds):
        """Formate une durée en secondes en string lisible"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)}m {seconds:.2f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
    
    @staticmethod
    def get_timestamp():
        """Retourne un timestamp formaté"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def create_experiment_id(prefix="exp"):
        """Crée un ID unique pour une expérience"""
        return f"{prefix}_{BenchmarkHelpers.get_timestamp()}"
    
    @staticmethod
    def memory_usage():
        """Retourne l'utilisation mémoire actuelle (approximative)"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

class ResultAnalyzer:
    """Classe utilitaire pour analyser les résultats de benchmark"""
    
    @staticmethod
    def compute_rankings(results_df, metric_columns=['accuracy', 'tree_depth', 'training_time']):
        """Calcule les classements pour différentes métriques"""
        rankings = {}
        
        for column in metric_columns:
            if column in results_df.columns:
                # Pour accuracy: plus haut = mieux
                if column == 'accuracy':
                    ranked = results_df.groupby('metric')[column].mean().sort_values(ascending=False)
                # Pour les autres: plus bas = mieux
                else:
                    ranked = results_df.groupby('metric')[column].mean().sort_values(ascending=True)
                    
                rankings[column] = ranked.reset_index()
                rankings[column]['rank'] = range(1, len(rankings[column]) + 1)
                
        return rankings
    
    @staticmethod
    def compute_composite_score(results_df, weights=None):
        """Calcule un score composite pondéré"""
        if weights is None:
            weights = {'accuracy': 0.5, 'tree_depth': 0.3, 'training_time': 0.2}
            
        # Normalisation des métriques
        normalized = results_df.copy()
        
        # Accuracy: normalisation 0-1
        max_acc = results_df['accuracy'].max()
        min_acc = results_df['accuracy'].min()
        if max_acc > min_acc:
            normalized['accuracy_norm'] = (results_df['accuracy'] - min_acc) / (max_acc - min_acc)
        else:
            normalized['accuracy_norm'] = 1.0
            
        # Tree depth: inversion (plus bas = mieux)
        max_depth = results_df['tree_depth'].max()
        min_depth = results_df['tree_depth'].min()
        if max_depth > min_depth:
            normalized['depth_norm'] = 1 - (results_df['tree_depth'] - min_depth) / (max_depth - min_depth)
        else:
            normalized['depth_norm'] = 1.0
            
        # Training time: inversion (plus bas = mieux)
        max_time = results_df['training_time'].max()
        min_time = results_df['training_time'].min()
        if max_time > min_time:
            normalized['time_norm'] = 1 - (results_df['training_time'] - min_time) / (max_time - min_time)
        else:
            normalized['time_norm'] = 1.0
            
        # Score composite
        normalized['composite_score'] = (
            weights.get('accuracy', 0) * normalized['accuracy_norm'] +
            weights.get('tree_depth', 0) * normalized['depth_norm'] +
            weights.get('training_time', 0) * normalized['time_norm']
        )
        
        return normalized
    
    @staticmethod
    def find_best_metric_per_dataset(results_df):
        """Trouve la meilleure métrique pour chaque dataset"""
        best_metrics = {}
        
        for dataset in results_df['dataset'].unique():
            dataset_data = results_df[results_df['dataset'] == dataset]
            
            # Meilleure accuracy
            best_acc = dataset_data.loc[dataset_data['accuracy'].idxmax()]
            best_metrics[f"{dataset}_accuracy"] = {
                'metric': best_acc['metric'],
                'value': best_acc['accuracy'],
                'type': 'accuracy'
            }
            
            # Meilleure profondeur (plus basse)
            best_depth = dataset_data.loc[dataset_data['tree_depth'].idxmin()]
            best_metrics[f"{dataset}_depth"] = {
                'metric': best_depth['metric'],
                'value': best_depth['tree_depth'],
                'type': 'depth'
            }
            
            # Meilleur temps (plus bas)
            best_time = dataset_data.loc[dataset_data['training_time'].idxmin()]
            best_metrics[f"{dataset}_time"] = {
                'metric': best_time['metric'],
                'value': best_time['training_time'],
                'type': 'time'
            }
            
        return best_metrics

class ProgressLogger:
    """Logger pour suivre la progression des benchmarks"""
    
    def __init__(self, total_steps, name="Benchmark"):
        self.total_steps = total_steps
        self.current_step = 0
        self.name = name
        self.start_time = time.time()
        
    def update(self, step=1, message=""):
        """Met à jour la progression"""
        self.current_step += step
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            progress = self.current_step / self.total_steps
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            
            progress_bar = self._create_progress_bar(progress)
            
            print(f"\r{self.name}: {progress_bar} {self.current_step}/{self.total_steps} "
                  f"({progress:.1%}) - Temps restant: {BenchmarkHelpers.format_duration(remaining)} "
                  f"{message}", end="", flush=True)
        
    def complete(self):
        """Marque la completion"""
        total_time = time.time() - self.start_time
        print(f"\n{self.name} terminé en {BenchmarkHelpers.format_duration(total_time)}")
        
    def _create_progress_bar(self, progress, width=20):
        """Crée une barre de progression textuelle"""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

# Export des classes principales
__all__ = ['BenchmarkHelpers', 'ResultAnalyzer', 'ProgressLogger']
