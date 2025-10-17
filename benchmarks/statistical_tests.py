import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import itertools
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalysis:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def load_results(self, results_path='results/raw_results/individual_metrics_results.csv'):
        """Charge les résultats du benchmark"""
        return pd.read_csv(results_path)
    
    def wilcoxon_signed_rank_test(self, results_df, control_metric='gini'):
        """
        Test de Wilcoxon signed-rank pour comparer chaque métrique avec la métrique de contrôle
        """
        datasets = results_df['dataset'].unique()
        metrics = results_df['metric'].unique()
        
        test_results = []
        
        for dataset in datasets:
            dataset_data = results_df[results_df['dataset'] == dataset]
            
            # Données de contrôle
            control_data = dataset_data[dataset_data['metric'] == control_metric]['accuracy']
            
            for metric in metrics:
                if metric == control_metric:
                    continue
                    
                metric_data = dataset_data[dataset_data['metric'] == metric]['accuracy']
                
                # Vérification qu'on a des données pour les deux métriques
                if len(control_data) > 0 and len(metric_data) > 0:
                    # Test de Wilcoxon signed-rank
                    statistic, p_value = stats.wilcoxon(control_data, metric_data)
                    
                    # Calcul de la magnitude de l'effet (r de Wilcoxon)
                    n = len(control_data)
                    effect_size = statistic / (n * (n + 1) / 2)
                    
                    test_results.append({
                        'dataset': dataset,
                        'metric': metric,
                        'control_metric': control_metric,
                        'statistic': statistic,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'significant': p_value < self.alpha
                    })
        
        return pd.DataFrame(test_results)
    
    def friedman_test(self, results_df):
        """
        Test de Friedman pour détecter des différences globales entre les métriques
        """
        # Préparation des données pour Friedman
        metrics = results_df['metric'].unique()
        datasets = results_df['dataset'].unique()
        
        # Matrice des rangs pour chaque dataset
        rank_matrix = []
        
        for dataset in datasets:
            dataset_data = results_df[results_df['dataset'] == dataset]
            
            # Calcul des rangs pour ce dataset
            metric_means = dataset_data.groupby('metric')['accuracy'].mean()
            ranks = stats.rankdata(-metric_means)  # Rang 1 pour la meilleure accuracy
            
            rank_matrix.append(ranks)
        
        # Test de Friedman
        statistic, p_value = stats.friedmanchisquare(*rank_matrix)
        
        # Rangs moyens
        mean_ranks = np.mean(rank_matrix, axis=0)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_ranks': dict(zip(metrics, mean_ranks)),
            'rank_matrix': rank_matrix
        }
    
    def nemenyi_posthoc(self, results_df, friedman_result):
        """
        Test post-hoc de Nemenyi après Friedman
        """
        metrics = results_df['metric'].unique()
        n_datasets = len(results_df['dataset'].unique())
        n_metrics = len(metrics)
        
        # Distance critique de Nemenyi
        q_alpha = self._get_nemenyi_critical_value(n_metrics, self.alpha)
        critical_difference = q_alpha * np.sqrt(n_metrics * (n_metrics + 1) / (6 * n_datasets))
        
        # Comparaisons par paires
        comparisons = []
        mean_ranks = friedman_result['mean_ranks']
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i >= j:
                    continue
                    
                rank_diff = abs(mean_ranks[metric1] - mean_ranks[metric2])
                significant = rank_diff > critical_difference
                
                comparisons.append({
                    'metric1': metric1,
                    'metric2': metric2,
                    'rank_difference': rank_diff,
                    'critical_difference': critical_difference,
                    'significant': significant
                })
        
        return pd.DataFrame(comparisons)
    
    def _get_nemenyi_critical_value(self, k, alpha=0.05):
        """Retourne la valeur critique de Nemenyi pour k traitements"""
        # Valeurs approximatives pour alpha=0.05
        nemenyi_table = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
            6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
            10: 3.164, 15: 3.444, 20: 3.615
        }
        
        if k in nemenyi_table:
            return nemenyi_table[k]
        else:
            # Approximation pour k > 20
            return 2.569 + 0.5 * np.sqrt(k - 4)
    
    def effect_size_analysis(self, results_df, control_metric='gini'):
        """
        Analyse de la taille de l'effet pour chaque métrique vs contrôle
        """
        effect_sizes = []
        
        for metric in results_df['metric'].unique():
            if metric == control_metric:
                continue
                
            # Calcul de Cohen's d pour chaque dataset
            dataset_effects = []
            
            for dataset in results_df['dataset'].unique():
                dataset_data = results_df[results_df['dataset'] == dataset]
                
                control_acc = dataset_data[dataset_data['metric'] == control_metric]['accuracy']
                metric_acc = dataset_data[dataset_data['metric'] == metric]['accuracy']
                
                if len(control_acc) > 1 and len(metric_acc) > 1:
                    # Cohen's d
                    mean_diff = np.mean(metric_acc) - np.mean(control_acc)
                    pooled_std = np.sqrt((np.std(control_acc, ddof=1)**2 + np.std(metric_acc, ddof=1)**2) / 2)
                    
                    if pooled_std > 0:
                        cohens_d = mean_diff / pooled_std
                        dataset_effects.append(cohens_d)
            
            if dataset_effects:
                overall_effect = np.mean(dataset_effects)
                effect_sizes.append({
                    'metric': metric,
                    'cohens_d_mean': overall_effect,
                    'cohens_d_std': np.std(dataset_effects),
                    'effect_size': self._interpret_cohens_d(overall_effect),
                    'n_datasets': len(dataset_effects)
                })
        
        return pd.DataFrame(effect_sizes)
    
    def _interpret_cohens_d(self, d):
        """Interprète la taille de l'effet de Cohen"""
        if abs(d) < 0.2:
            return 'Negligible'
        elif abs(d) < 0.5:
            return 'Small'
        elif abs(d) < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def confidence_interval_analysis(self, results_df, confidence=0.95):
        """
        Calcule les intervalles de confiance pour chaque métrique
        """
        ci_results = []
        
        for metric in results_df['metric'].unique():
            metric_data = results_df[results_df['metric'] == metric]['accuracy']
            
            if len(metric_data) > 1:
                mean_acc = np.mean(metric_data)
                std_acc = np.std(metric_data, ddof=1)
                n = len(metric_data)
                
                # Intervalle de confiance t-student
                t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
                margin_error = t_value * (std_acc / np.sqrt(n))
                
                ci_lower = mean_acc - margin_error
                ci_upper = mean_acc + margin_error
                
                ci_results.append({
                    'metric': metric,
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'margin_error': margin_error,
                    'sample_size': n
                })
        
        return pd.DataFrame(ci_results)
    
    def multiple_comparison_correction(self, test_results, method='fdr_bh'):
        """
        Applique une correction pour comparaisons multiples
        """
        p_values = test_results['p_value'].values
        rejected, corrected_p, _, _ = multipletests(p_values, alpha=self.alpha, method=method)
        
        test_results['p_value_corrected'] = corrected_p
        test_results['significant_corrected'] = rejected
        
        return test_results
    
    def create_statistical_summary(self, results_df):
        """
        Crée un résumé statistique complet
        """
        print("=" * 60)
        print("ANALYSE STATISTIQUE COMPLÈTE")
        print("=" * 60)
        
        # 1. Test de Friedman
        print("\n1. TEST DE FRIEDMAN (différences globales)")
        friedman_result = self.friedman_test(results_df)
        print(f"Statistique Friedman: {friedman_result['statistic']:.4f}")
        print(f"P-value: {friedman_result['p_value']:.6f}")
        print(f"Significatif: {friedman_result['significant']}")
        
        if friedman_result['significant']:
            print("\nRangs moyens:")
            for metric, rank in sorted(friedman_result['mean_ranks'].items(), key=lambda x: x[1]):
                print(f"  {metric}: {rank:.3f}")
        
        # 2. Test de Nemenyi post-hoc
        print("\n2. TEST POST-HOC DE NEMENYI")
        nemenyi_results = self.nemenyi_posthoc(results_df, friedman_result)
        significant_pairs = nemenyi_results[nemenyi_results['significant']]
        print(f"Comparaisons significatives: {len(significant_pairs)}")
        
        if len(significant_pairs) > 0:
            print("Paires significatives:")
            for _, row in significant_pairs.iterrows():
                print(f"  {row['metric1']} vs {row['metric2']}: "
                      f"diff={row['rank_difference']:.3f} (CD={row['critical_difference']:.3f})")
        
        # 3. Tests de Wilcoxon vs Gini
        print("\n3. TESTS DE WILCOXON SIGNED-RANK (vs Gini)")
        wilcoxon_results = self.wilcoxon_signed_rank_test(results_df, 'gini')
        
        # Correction pour comparaisons multiples
        wilcoxon_corrected = self.multiple_comparison_correction(wilcoxon_results)
        
        significant_wilcoxon = wilcoxon_corrected[wilcoxon_corrected['significant_corrected']]
        print(f"Métriques significativement différentes de Gini: {len(significant_wilcoxon)}")
        
        for _, row in significant_wilcoxon.iterrows():
            improvement = "✓ AMÉLIORATION" if row['effect_size'] > 0 else "✗ DÉTERIORATION"
            print(f"  {row['metric']}: p={row['p_value_corrected']:.6f}, "
                  f"effet={row['effect_size']:.3f} {improvement}")
        
        # 4. Analyse de la taille de l'effet
        print("\n4. ANALYSE DE LA TAILLE DE L'EFFET (Cohen's d vs Gini)")
        effect_results = self.effect_size_analysis(results_df, 'gini')
        
        for _, row in effect_results.iterrows():
            print(f"  {row['metric']}: d={row['cohens_d_mean']:.3f} ({row['effect_size']})")
        
        # 5. Intervalles de confiance
        print("\n5. INTERVALLES DE CONFIANCE (95%)")
        ci_results = self.confidence_interval_analysis(results_df)
        
        top_metrics = ci_results.nlargest(5, 'mean_accuracy')
        for _, row in top_metrics.iterrows():
            print(f"  {row['metric']}: {row['mean_accuracy']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        
        return {
            'friedman': friedman_result,
            'nemenyi': nemenyi_results,
            'wilcoxon': wilcoxon_corrected,
            'effect_sizes': effect_results,
            'confidence_intervals': ci_results
        }
    
    def create_statistical_plots(self, results_df, output_dir='results/figures/'):
        """Crée des visualisations statistiques"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Graphique des intervalles de confiance
        ci_results = self.confidence_interval_analysis(results_df)
        ci_results = ci_results.sort_values('mean_accuracy', ascending=False)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(ci_results))
        
        plt.errorbar(ci_results['mean_accuracy'], y_pos,
                    xerr=[ci_results['mean_accuracy'] - ci_results['ci_lower'],
                          ci_results['ci_upper'] - ci_results['mean_accuracy']],
                    fmt='o', capsize=5, capthick=2)
        
        plt.yticks(y_pos, ci_results['metric'])
        plt.xlabel('Accuracy')
        plt.title('Intervalles de Confiance 95% par Métrique')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap des comparaisons de Nemenyi
        friedman_result = self.friedman_test(results_df)
        nemenyi_results = self.nemenyi_posthoc(results_df, friedman_result)
        
        metrics = sorted(results_df['metric'].unique())
        comparison_matrix = np.zeros((len(metrics), len(metrics)))
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i == j:
                    comparison_matrix[i, j] = 0  # Diagonale
                else:
                    mask = ((nemenyi_results['metric1'] == metric1) & 
                           (nemenyi_results['metric2'] == metric2)) | \
                           ((nemenyi_results['metric1'] == metric2) & 
                           (nemenyi_results['metric2'] == metric1))
                    
                    if mask.any():
                        comparison_matrix[i, j] = 1 if nemenyi_results[mask]['significant'].values[0] else 0
                    else:
                        comparison_matrix[i, j] = 0
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(comparison_matrix, annot=True, fmt='d',
                   xticklabels=metrics, yticklabels=metrics,
                   cmap=['white', 'red'], cbar=False)
        plt.title('Comparaisons Significatives de Nemenyi\n(1 = significatif, 0 = non significatif)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}nemenyi_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_statistical_analysis():
    """Fonction principale pour l'analyse statistique"""
    analysis = StatisticalAnalysis(alpha=0.05)
    
    # Chargement des résultats
    try:
        results_df = analysis.load_results()
    except FileNotFoundError:
        print("Fichier de résultats non trouvé. Lancez d'abord le benchmark individuel.")
        return None
    
    print("Début de l'analyse statistique...")
    
    # Analyse statistique complète
    statistical_summary = analysis.create_statistical_summary(results_df)
    
    # Création des graphiques
    analysis.create_statistical_plots(results_df)
    
    # Sauvegarde des résultats détaillés
    statistical_summary['wilcoxon'].to_csv('results/tables/wilcoxon_test_results.csv', index=False)
    statistical_summary['effect_sizes'].to_csv('results/tables/effect_size_analysis.csv', index=False)
    statistical_summary['confidence_intervals'].to_csv('results/tables/confidence_intervals.csv', index=False)
    statistical_summary['nemenyi'].to_csv('results/tables/nemenyi_posthoc.csv', index=False)
    
    print("\nAnalyse statistique terminée!")
    return statistical_summary

if __name__ == "__main__":
    results = run_statistical_analysis()
    if results:
        print("Résultats sauvegardés dans results/tables/")
