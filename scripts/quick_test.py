#!/usr/bin/env python3
"""
Script de test rapide pour vérifier que tout fonctionne
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Teste que tous les modules peuvent être importés"""
    print("Test des imports...")
    
    try:
        from metrics import ALL_METRICS, METRICS_BY_CATEGORY
        from tree import DecisionTree
        from data import DataLoader
        from utils import ResultsVisualizer
        
        print("✓ Tous les imports fonctionnent")
        return True
        
    except Exception as e:
        print(f"✗ Erreur d'import: {e}")
        return False

def test_metrics():
    """Teste le calcul des métriques d'impureté"""
    print("\nTest des métriques d'impureté...")
    
    try:
        from metrics import ALL_METRICS
        
        # Distribution test
        proportions = [0.7, 0.3]
        
        for name, metric_func in list(ALL_METRICS.items())[:5]:  # Test des 5 premières
            try:
                result = metric_func(proportions)
                print(f"✓ {name}: {result:.4f}")
            except Exception as e:
                print(f"✗ {name}: {e}")
                
        return True
        
    except Exception as e:
        print(f"✗ Erreur avec les métriques: {e}")
        return False

def test_decision_tree():
    """Teste l'arbre de décision avec une petite donnée"""
    print("\nTest de l'arbre de décision...")
    
    try:
        import numpy as np
        from tree import DecisionTree
        
        # Données simples
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(impurity_measure='gini', max_depth=3)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"✓ Arbre entraîné avec succès")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Profondeur: {tree.get_tree_depth()}")
        print(f"  Taille: {tree.get_tree_size()} nœuds")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur avec l'arbre: {e}")
        return False

def test_data_loader():
    """Teste le chargement des données"""
    print("\nTest du DataLoader...")
    
    try:
        from data import DataLoader
        
        loader = DataLoader()
        loader.load_sklearn_datasets()
        
        datasets = loader.get_all_datasets()
        print(f"✓ {len(datasets)} datasets chargés")
        
        for name, (X, y) in datasets.items():
            print(f"  {name}: {X.shape[0]} instances, {X.shape[1]} features, {len(np.unique(y))} classes")
            
        return True
        
    except Exception as e:
        print(f"✗ Erreur avec DataLoader: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("=" * 50)
    print("TEST RAPIDE DU SYSTÈME ITC")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
        
    if test_metrics():
        tests_passed += 1
        
    if test_decision_tree():
        tests_passed += 1
        
    if test_data_loader():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"RÉSULTAT: {tests_passed}/{total_tests} tests passés")
    
    if tests_passed == total_tests:
        print("✓ Le système est prêt pour le benchmarking!")
        print("\nProchaines étapes:")
        print("1. python scripts/download_datasets.py")
        print("2. python run_all_benchmarks.py")
    else:
        print("✗ Certains tests ont échoué. Vérifiez l'installation.")
        
    return tests_passed == total_tests

if __name__ == "__main__":
    import numpy as np  # Pour test_data_loader
    success = main()
    sys.exit(0 if success else 1)
