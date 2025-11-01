#!/usr/bin/env python3
"""
Script principal pour exécuter tous les benchmarks
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))
import pandas as pd
from datetime import datetime
from benchmarks.__init__ import *

def create_directories():
    """Crée la structure de dossiers nécessaire"""
    directories = [
        'results/raw_results',
        'results/tables', 
        'results/figures',
        'data/datasets'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Dossier créé: {directory}")

def main():
    print("=" * 60)
    print("BENCHMARK COMPLET DES MÉTRIQUES D'IMPURETÉ")
    print("=" * 60)
    
    # Création des dossiers
    create_directories()
    
    start_time = datetime.now()
    print(f"\nDébut du benchmark: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Benchmark des métriques individuelles
        print("\n" + "=" * 50)
        print("1. BENCHMARK DES MÉTRIQUES INDIVIDUELLES")
        print("=" * 50)
        individual_results, individual_agg = run_individual_benchmarks()
        print("✓ Benchmark individuel terminé")
        
        # 2. Comparaison des métriques hybrides
        print("\n" + "=" * 50)
        print("2. COMPARAISON DES MÉTRIQUES HYBRIDES")
        print("=" * 50)
        hybrid_results, hybrid_analysis = run_hybrid_comparison()
        print("✓ Comparaison hybride terminée")
        
        # 3. Analyse de sensibilité
        print("\n" + "=" * 50)
        print("3. ANALYSE DE SENSIBILITÉ")
        print("=" * 50)
        sensitivity_results = run_sensitivity_analysis()
        print("✓ Analyse de sensibilité terminée")
        
        # 4. Tests statistiques
        print("\n" + "=" * 50)
        print("4. TESTS STATISTIQUES")
        print("=" * 50)
        statistical_results = run_statistical_analysis()
        print("✓ Tests statistiques terminés")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("BENCHMARK TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"Durée totale: {duration}")
        print(f"Début: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Résumé des résultats
        print("\nRÉSUMÉ DES RÉSULTATS:")
        print(f"- Métriques individuelles évaluées: {len(individual_agg['metric'].unique())}")
        print(f"- Datasets utilisés: {len(individual_agg['dataset'].unique())}")
        print(f"- Méthodes hybrides comparées: {len(hybrid_analysis[hybrid_analysis['category'] == 'hybrid']['metric'].unique())}")
        
        # Top 5 des métriques par accuracy moyenne
        overall_performance = individual_agg.groupby('metric')['accuracy_mean'].mean().sort_values(ascending=False)
        print(f"\nTOP 5 DES MÉTRIQUES (Accuracy moyenne):")
        for i, (metric, acc) in enumerate(overall_performance.head().items(), 1):
            print(f"  {i}. {metric}: {acc:.4f}")
            
    except Exception as e:
        print(f"\n❌ ERREUR lors du benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
