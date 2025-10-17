import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os

class ResultsVisualizer:
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
        
    def create_performance_comparison_plot(self, results_df, output_path='results/figures/performance_comparison.png'):
        """Crée un graphique comparant les performances des différentes métriques"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Agrégation des résultats
        performance = results_df.groupby('metric').agg({
            'accuracy': ['mean', 'std'],
            'tree_depth': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        performance.columns = ['_'.join(col).strip() for col in performance.columns.values]
        performance = performance.reset_index()
        
        # Tri par accuracy moyenne
        performance = performance.sort_values('accuracy_mean', ascending=False)
        
        # Graphique d'accuracy
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Accuracy moyenne
        metrics = performance['metric']
        y_pos = np.arange(len(metrics))
        
        bars = ax1.barh(y_pos, performance['accuracy_mean'], 
                       xerr=performance['accuracy_std'], 
                       alpha=0.7, color=self.colors[0])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(metrics, fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Performance en Accuracy par Métrique')
        ax1.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=7)
        
        # 2. Profondeur moyenne des arbres
        ax2.barh(y_pos, performance['tree_depth_mean'],
                xerr=performance['tree_depth_std'], alpha=0.7, color=self.colors[1])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(metrics, fontsize=8)
        ax2.set_xlabel('Profondeur moyenne')
        ax2.set_title('Complexité des Arbres (Profondeur)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Temps d'entraînement
        ax3.barh(y_pos, performance['training_time_mean'],
                xerr=performance['training_time_std'], alpha=0.7, color=self.colors[2])
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(metrics, fontsize=8)
        ax3.set_xlabel('Temps (secondes)')
        ax3.set_title('Temps d\'Entraînement')
        ax3.grid(True, alpha=0.3)
        
        # 4. Score composite (accuracy / profondeur)
        performance['composite_score'] = (
            performance['accuracy_mean'] / performance['tree_depth_mean']
        )
        performance = performance.sort_values('composite_score', ascending=False)
        
        ax4.barh(np.arange(len(performance)), performance['composite_score'],
                alpha=0.7, color=self.colors[3])
        ax4.set_yticks(np.arange(len(performance)))
        ax4.set_yticklabels(performance['metric'], fontsize=8)
        ax4.set_xlabel('Score (Accuracy / Profondeur)')
        ax4.set_title('Score Composite: Accuracy par Profondeur')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return performance
    
    def create_metric_category_plot(self, results_df, output_path='results/figures/metric_categories.png'):
        """Crée un graphique groupé par catégories de métriques"""
        # Définition des catégories
        categories = {
            'Classique': ['gini', 'shannon', 'misclassification'],
            'Paramétrique': ['renyi_0.5', 'renyi_2.0', 'tsallis_0.5', 'tsallis_1.3', 'tsallis_2.0', 'kumaraswamy'],
            'Probabiliste': ['cross_entropy', 'kl_divergence', 'js_divergence'],
            'Distance': ['hellinger', 'energy', 'polarization_3.5'],
            'Théorique': ['bregman_squared', 'bregman_entropy'],
            'Hybride': ['itc', 'itc_alpha1.3', 'itc_alpha1.7', 'shannon_polarization', 'tsallis_hellinger']
        }
        
        # Ajout de la catégorie aux résultats
        results_with_cat = results_df.copy()
        results_with_cat['category'] = 'Autre'
        
        for category, metrics in categories.items():
            for metric in metrics:
                results_with_cat.loc[results_with_cat['metric'] == metric, 'category'] = category
        
        # Agrégation par catégorie
        category_performance = results_with_cat.groupby('category').agg({
            'accuracy': ['mean', 'std'],
            'tree_depth': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        category_performance.columns = ['_'.join(col).strip() for col in category_performance.columns.values]
        category_performance = category_performance.reset_index()
        
        # Graphique
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy par catégorie
        categories_sorted = category_performance.sort_values('accuracy_mean', ascending=False)
        bars1 = ax1.bar(range(len(categories_sorted)), categories_sorted['accuracy_mean'],
                       yerr=categories_sorted['accuracy_std'], capsize=5,
                       color=self.colors[:len(categories_sorted)], alpha=0.7)
        ax1.set_xticks(range(len(categories_sorted)))
        ax1.set_xticklabels(categories_sorted['category'], rotation=45, ha='right')
        ax1.set_ylabel('Accuracy moyenne')
        ax1.set_title('Performance par Catégorie de Métrique')
        ax1.grid(True, alpha=0.3)
        
        # Profondeur par catégorie
        categories_sorted_depth = category_performance.sort_values('tree_depth_mean')
        bars2 = ax2.bar(range(len(categories_sorted_depth)), categories_sorted_depth['tree_depth_mean'],
                       yerr=categories_sorted_depth['tree_depth_std'], capsize=5,
                       color=self.colors[:len(categories_sorted_depth)], alpha=0.7)
        ax2.set_xticks(range(len(categories_sorted_depth)))
        ax2.set_xticklabels(categories_sorted_depth['category'], rotation=45, ha='right')
        ax2.set_ylabel('Profondeur moyenne')
        ax2.set_title('Complexité par Catégorie de Métrique')
        ax2.grid(True, alpha=0.3)
        
        # Temps par catégorie
        categories_sorted_time = category_performance.sort_values('training_time_mean')
        bars3 = ax3.bar(range(len(categories_sorted_time)), categories_sorted_time['training_time_mean'],
                       yerr=categories_sorted_time['training_time_std'], capsize=5,
                       color=self.colors[:len(categories_sorted_time)], alpha=0.7)
        ax3.set_xticks(range(len(categories_sorted_time)))
        ax3.set_xticklabels(categories_sorted_time['category'], rotation=45, ha='right')
        ax3.set_ylabel('Temps moyen (s)')
        ax3.set_title('Efficacité par Catégorie de Métrique')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return category_performance
    
    def create_hybrid_comparison_plot(self, hybrid_results, output_path='results/figures/hybrid_comparison.png'):
        """Crée un graphique comparant les méthodes hybrides"""
        # Agrégation des résultats hybrides
        hybrid_performance = hybrid_results.groupby(['combination', 'gamma']).agg({
            'accuracy': 'mean',
            'tree_depth': 'mean'
        }).reset_index()
        
        # Graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs gamma pour chaque combinaison
        combinations = hybrid_performance['combination'].unique()
        
        for i, combo in enumerate(combinations):
            combo_data = hybrid_performance[hybrid_performance['combination'] == combo]
            ax1.plot(combo_data['gamma'], combo_data['accuracy'], 
                    marker='o', linewidth=2, label=combo, color=self.colors[i])
        
        ax1.set_xlabel('Paramètre γ')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Impact du Paramètre γ sur l\'Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Profondeur vs gamma pour chaque combinaison
        for i, combo in enumerate(combinations):
            combo_data = hybrid_performance[hybrid_performance['combination'] == combo]
            ax2.plot(combo_data['gamma'], combo_data['tree_depth'], 
                    marker='s', linewidth=2, label=combo, color=self.colors[i])
        
        ax2.set_xlabel('Paramètre γ')
        ax2.set_ylabel('Profondeur')
        ax2.set_title('Impact du Paramètre γ sur la Profondeur')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return hybrid_performance
    
    def create_correlation_heatmap(self, results_df, output_path='results/figures/correlation_heatmap.png'):
        """Crée une heatmap de corrélation entre les métriques de performance"""
        # Pivot des résultats pour avoir une ligne par métrique
        pivot_df = results_df.groupby('metric').agg({
            'accuracy': 'mean',
            'tree_depth': 'mean', 
            'tree_size': 'mean',
            'training_time': 'mean'
        }).reset_index()
        
        # Matrice de corrélation
        corr_matrix = pivot_df[['accuracy', 'tree_depth', 'tree_size', 'training_time']].corr()
        
        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Corrélations entre Métriques de Performance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix
    
    def create_radar_chart(self, results_df, top_n=8, output_path='results/figures/radar_chart.png'):
        """Crée un graphique radar pour les meilleures métriques"""
        from math import pi
        
        # Sélection des top N métriques
        top_metrics = results_df.groupby('metric')['accuracy'].mean().nlargest(top_n).index
        top_data = results_df[results_df['metric'].isin(top_metrics)]
        
        # Normalisation des métriques
        metrics_agg = top_data.groupby('metric').agg({
            'accuracy': 'mean',
            'tree_depth': 'mean',
            'training_time': 'mean'
        })
        
        # Normalisation (0-1, 1 étant le meilleur)
        metrics_normalized = metrics_agg.copy()
        metrics_normalized['accuracy'] = metrics_agg['accuracy'] / metrics_agg['accuracy'].max()
        metrics_normalized['tree_depth'] = 1 - (metrics_agg['tree_depth'] / metrics_agg['tree_depth'].max())
        metrics_normalized['training_time'] = 1 - (metrics_agg['training_time'] / metrics_agg['training_time'].max())
        
        # Préparation pour le radar chart
        categories = ['Accuracy', 'Profondeur\n(inverse)', 'Temps\n(inverse)']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Fermer le cercle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, (metric, row) in enumerate(metrics_normalized.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # Fermer le cercle
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=metric, color=self.colors[i])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title('Comparaison Radar des Top Métriques\n(Plus grand = meilleur)', size=14, y=1.08)
        plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics_normalized

def create_all_visualizations():
    """Crée toutes les visualisations principales"""
    visualizer = ResultsVisualizer()
    
    try:
        # Chargement des résultats
        individual_results = pd.read_csv('results/raw_results/individual_metrics_results.csv')
        hybrid_results = pd.read_csv('results/raw_results/hybrid_comparison_results.csv')
        
        print("Création des visualisations...")
        
        # 1. Comparaison générale des performances
        print("1. Graphique de comparaison des performances...")
        performance_df = visualizer.create_performance_comparison_plot(individual_results)
        
        # 2. Analyse par catégories
        print("2. Graphique par catégories...")
        category_df = visualizer.create_metric_category_plot(individual_results)
        
        # 3. Comparaison des hybrides
        print("3. Graphique des méthodes hybrides...")
        hybrid_df = visualizer.create_hybrid_comparison_plot(hybrid_results)
        
        # 4. Heatmap de corrélation
        print("4. Heatmap de corrélation...")
        corr_matrix = visualizer.create_correlation_heatmap(individual_results)
        
        # 5. Radar chart
        print("5. Graphique radar...")
        radar_df = visualizer.create_radar_chart(individual_results)
        
        print("Toutes les visualisations ont été créées avec succès!")
        
        return {
            'performance': performance_df,
            'categories': category_df,
            'hybrids': hybrid_df,
            'correlation': corr_matrix,
            'radar': radar_df
        }
        
    except FileNotFoundError as e:
        print(f"Fichier de résultats non trouvé: {e}")
        print("Veuillez d'abord exécuter les benchmarks.")
        return None

if __name__ == "__main__":
    results = create_all_visualizations()
    if results:
        print("Visualisations sauvegardées dans results/figures/")
