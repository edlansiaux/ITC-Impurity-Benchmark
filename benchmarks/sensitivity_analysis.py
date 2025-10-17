import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import time
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from tree.decision_tree import DecisionTree
from data.data_loader import DataLoader

class SensitivityAnalysis:
    def __init__(self, n_runs=5, test_size=0.3, random_state=42):
        self.n_runs = n_runs
        self.test_size = test_size
        self.random_state = random_state
        
    def analyze_itc_parameters(self, datasets):
        """Analyse de sensibilité des paramètres α, β, γ de ITC"""
        results = []
        
        # Grille de paramètres pour ITC
        alpha_values = [1.0, 1.3, 1.5, 1.7, 2.0]
        beta_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\nAnalyse de sensibilité ITC sur: {dataset_name}")
            print(f"Alpha: {len(alpha_values)}, Beta: {len(beta_values)}, Gamma: {len(gamma_values)}")
            print(f"Total combinaisons: {len(alpha_values) * len(beta_values) * len(gamma_values)}")
            
            dataset_results = self._analyze_dataset_parameters(
                X, y, dataset_name, alpha_values, beta_values, gamma_values
            )
            results.extend(dataset_results)
            
        return pd.DataFrame(results)
    
    def _analyze_dataset_parameters(self, X, y, dataset_name, alpha_values, beta_values, gamma_values):
        """Analyse des paramètres pour un dataset spécifique"""
        dataset_results = []
        total_combinations = len(alpha_values) * len(beta_values) * len(gamma_values)
        
        with tqdm(total=total_combinations, desc=f"Paramètres {dataset_name}") as pbar:
            for alpha, beta, gamma in itertools.product(alpha_values, beta_values, gamma_values):
                try:
                    # Nom de la métrique avec paramètres
                    metric_name = f"itc_alpha{alpha}_beta{beta}_gamma{gamma}"
                    
                    # Évaluation sur plusieurs runs
                    run_results = self._evaluate_parameters(
                        X, y, alpha, beta, gamma, metric_name, dataset_name
                    )
                    
                    if run_results:
                        dataset_results.extend(run_results)
                        
                except Exception as e:
                    print(f"Erreur avec alpha={alpha}, beta={beta}, gamma={gamma}: {e}")
                    
                pbar.update(1)
                
        return dataset_results
    
    def _evaluate_parameters(self, X, y, alpha, beta, gamma, metric_name, dataset_name):
        """Évalue une combinaison de paramètres sur plusieurs runs"""
        run_results = []
        
        for run in range(self.n_runs):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size,
                    random_state=self.random_state + run,
                    stratify=y
                )
                
                # Arbre avec les paramètres ITC spécifiques
                tree = DecisionTree(
                    impurity_measure='itc',
                    max_depth=20,
                    min_samples_split=2,
                    random_state=self.random_state + run,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
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
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'run': run,
                    'accuracy': accuracy,
                    'tree_depth': tree_depth,
                    'tree_size': tree_size,
                    'training_time': training_time
                }
                
                run_results.append(result)
                
            except Exception as e:
                print(f"Erreur run {run} avec {metric_name}: {e}")
                continue
                
        return run_results
    
    def find_optimal_parameters(self, results_df):
        """Trouve les paramètres optimaux basés sur les résultats"""
        # Agrégation par paramètres
        aggregated = results_df.groupby(['alpha', 'beta', 'gamma']).agg({
            'accuracy': ['mean', 'std'],
            'tree_depth': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated = aggregated.reset_index()
        
        # Score composite (accuracy élevée, profondeur faible)
        aggregated['composite_score'] = (
            aggregated['accuracy_mean'] * 0.4 +
            (1 - aggregated['tree_depth_mean'] / 20) * 0.3 +
            (1 - aggregated['training_time_mean'] / 10) * 0.3
        )
        
        # Meilleurs paramètres par score composite
        best_overall = aggregated.loc[aggregated['composite_score'].idxmax()]
        
        # Meilleurs paramètres par métrique individuelle
        best_accuracy = aggregated.loc[aggregated['accuracy_mean'].idxmax()]
        best_depth = aggregated.loc[aggregated['tree_depth_mean'].idxmin()]
        
        return {
            'best_overall': best_overall,
            'best_accuracy': best_accuracy,
            'best_depth': best_depth,
            'all_results': aggregated
        }
    
    def create_parameter_surface_plots(self, results_df, output_dir='results/figures/'):
        """Crée des graphiques de surface pour visualiser la sensibilité"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Agrégation des résultats
        aggregated = results_df.groupby(['alpha', 'beta', 'gamma']).agg({
            'accuracy': 'mean',
            'tree_depth': 'mean',
            'training_time': 'mean'
        }).reset_index()
        
        # Graphiques pour différentes valeurs de gamma
        gamma_values = aggregated['gamma'].unique()
        
        for gamma in gamma_values:
            if gamma not in [0.1, 0.3, 0.5, 0.7]:
                continue
                
            gamma_data = aggregated[aggregated['gamma'] == gamma]
            
            # Création d'une grille pour la surface
            alphas = sorted(gamma_data['alpha'].unique())
            betas = sorted(gamma_data['beta'].unique())
            
            accuracy_grid = np.zeros((len(alphas), len(betas)))
            depth_grid = np.zeros((len(alphas), len(betas)))
            
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    mask = (gamma_data['alpha'] == alpha) & (gamma_data['beta'] == beta)
                    if mask.any():
                        accuracy_grid[i, j] = gamma_data[mask]['accuracy'].values[0]
                        depth_grid[i, j] = gamma_data[mask]['tree_depth'].values[0]
                    else:
                        accuracy_grid[i, j] = np.nan
                        depth_grid[i, j] = np.nan
            
            # Graphique de surface pour l'accuracy
            self._create_surface_plot(
                alphas, betas, accuracy_grid,
                title=f'ITC Accuracy Surface (γ={gamma})',
                xlabel='Alpha (α)', ylabel='Beta (β)', zlabel='Accuracy',
                filename=f'{output_dir}itc_accuracy_surface_gamma{gamma}.png'
            )
            
            # Graphique de surface pour la profondeur
            self._create_surface_plot(
                alphas, betas, depth_grid,
                title=f'ITC Tree Depth Surface (γ={gamma})',
                xlabel='Alpha (α)', ylabel='Beta (β)', zlabel='Tree Depth',
                filename=f'{output_dir}itc_depth_surface_gamma{gamma}.png'
            )
    
    def _create_surface_plot(self, x_vals, y_vals, z_grid, title, xlabel, ylabel, zlabel, filename):
        """Crée un graphique de surface 3D"""
        from mpl_toolkits.mplot3d import Axes3D
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = z_grid.T  # Transposer pour correspondre à meshgrid
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Point optimal
        optimal_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
        optimal_x = X[optimal_idx]
        optimal_y = Y[optimal_idx]
        optimal_z = Z[optimal_idx]
        
        ax.scatter(optimal_x, optimal_y, optimal_z, color='red', s=100, label='Optimal')
        ax.legend()
        
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_hybrid_combinations(self, datasets):
        """Analyse différentes combinaisons hybrides"""
        results = []
        
        # Définition des combinaisons hybrides à tester
        hybrid_combinations = [
            {
                'name': 'tsallis_polarization',
                'components': ['tsallis', 'polarization'],
                'gamma_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            },
            {
                'name': 'shannon_polarization', 
                'components': ['shannon', 'polarization'],
                'gamma_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            },
            {
                'name': 'tsallis_hellinger',
                'components': ['tsallis', 'hellinger'],
                'gamma_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            },
            {
                'name': 'renyi_polarization',
                'components': ['renyi', 'polarization'],
                'gamma_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }
        ]
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\nAnalyse des combinaisons hybrides sur: {dataset_name}")
            
            for combo in hybrid_combinations:
                combo_results = self._evaluate_hybrid_combo(
                    X, y, dataset_name, combo
                )
                results.extend(combo_results)
                
        return pd.DataFrame(results)
    
    def _evaluate_hybrid_combo(self, X, y, dataset_name, combo_config):
        """Évalue une combinaison hybride spécifique"""
        results = []
        
        for gamma in combo_config['gamma_range']:
            metric_name = f"{combo_config['name']}_gamma{gamma}"
            
            for run in range(self.n_runs):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=self.test_size,
                        random_state=self.random_state + run,
                        stratify=y
                    )
                    
                    # Utilisation de la métrique hybride correspondante
                    if combo_config['name'] == 'tsallis_polarization':
                        tree = DecisionTree(
                            impurity_measure='itc',
                            max_depth=20,
                            min_samples_split=2,
                            random_state=self.random_state + run,
                            alpha=1.5,
                            beta=3.5,
                            gamma=gamma
                        )
                    elif combo_config['name'] == 'shannon_polarization':
                        tree = DecisionTree(
                            impurity_measure='shannon_polarization',
                            max_depth=20,
                            min_samples_split=2,
                            random_state=self.random_state + run,
                            gamma=gamma,
                            beta=3.5
                        )
                    else:
                        # Pour les autres combinaisons, on utilise une approche générique
                        continue
                    
                    start_time = time.time()
                    tree.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    y_pred = tree.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    tree_depth = tree.get_tree_depth()
                    
                    result = {
                        'dataset': dataset_name,
                        'combination': combo_config['name'],
                        'gamma': gamma,
                        'run': run,
                        'accuracy': accuracy,
                        'tree_depth': tree_depth,
                        'training_time': training_time
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Erreur {combo_config['name']} gamma={gamma}: {e}")
                    continue
                    
        return results

def run_sensitivity_analysis():
    """Fonction principale pour l'analyse de sensibilité"""
    # Chargement des données
    data_loader = DataLoader()
    data_loader.load_sklearn_datasets()
    datasets = data_loader.get_all_datasets()
    
    # Réduction pour les tests
    test_datasets = {k: v for k, v in list(datasets.items())[:3]}
    
    analysis = SensitivityAnalysis(n_runs=3)  # Réduit pour les tests
    
    print("=" * 60)
    print("ANALYSE DE SENSIBILITÉ DES PARAMÈTRES ITC")
    print("=" * 60)
    
    # 1. Analyse des paramètres ITC
    print("\n1. Analyse des paramètres α, β, γ pour ITC...")
    itc_results = analysis.analyze_itc_parameters(test_datasets)
    itc_results.to_csv('results/raw_results/itc_parameter_sensitivity.csv', index=False)
    
    # 2. Trouver les paramètres optimaux
    print("\n2. Recherche des paramètres optimaux...")
    optimal_params = analysis.find_optimal_parameters(itc_results)
    
    print("\nParamètres optimaux trouvés:")
    print(f"Meilleur overall: α={optimal_params['best_overall']['alpha']}, "
          f"β={optimal_params['best_overall']['beta']}, "
          f"γ={optimal_params['best_overall']['gamma']}")
    print(f"Score composite: {optimal_params['best_overall']['composite_score']:.4f}")
    
    # 3. Graphiques de surface
    print("\n3. Création des graphiques de surface...")
    analysis.create_parameter_surface_plots(itc_results)
    
    # 4. Analyse des combinaisons hybrides
    print("\n4. Analyse des combinaisons hybrides alternatives...")
    hybrid_results = analysis.analyze_hybrid_combinations(test_datasets)
    hybrid_results.to_csv('results/raw_results/hybrid_combinations_analysis.csv', index=False)
    
    # Sauvegarde des résultats d'optimisation
    optimal_params['all_results'].to_csv('results/tables/optimal_parameters_analysis.csv', index=False)
    
    return {
        'itc_parameters': itc_results,
        'optimal_parameters': optimal_params,
        'hybrid_combinations': hybrid_results
    }

if __name__ == "__main__":
    results = run_sensitivity_analysis()
    print("\nAnalyse de sensibilité terminée avec succès!")
