#!/usr/bin/env python3
"""
Script pour télécharger les datasets utilisés dans le benchmark
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml
from data.data_loader import DataLoader

def download_uci_datasets():
    """Télécharge les datasets UCI utilisés dans l'étude"""
    datasets_info = {
        'balance-scale': {
            'openml_id': 12,
            'target_column': 'class'
        },
        'car-evaluation': {
            'openml_id': 40975,
            'target_column': 'class'
        },
        'nursery': {
            'openml_id': 40984,
            'target_column': 'class'
        },
        'chess-krvk': {
            'openml_id': 40968,
            'target_column': 'class'
        }
    }
    
    data_dir = 'data/datasets'
    os.makedirs(data_dir, exist_ok=True)
    
    for name, info in datasets_info.items():
        print(f"Téléchargement de {name}...")
        try:
            # Téléchargement depuis OpenML
            dataset = fetch_openml(data_id=info['openml_id'], as_frame=True, parser='pandas')
            df = dataset.frame
            
            # Sauvegarde en CSV
            filename = os.path.join(data_dir, f"{name}.csv")
            df.to_csv(filename, index=False)
            print(f"✓ {name} sauvegardé ({len(df)} instances, {len(df.columns)} features)")
            
        except Exception as e:
            print(f"✗ Erreur avec {name}: {e}")
    
    print("\nTous les datasets ont été téléchargés!")

def create_sample_datasets():
    """Crée des datasets d'exemple si les vrais ne sont pas disponibles"""
    data_dir = 'data/datasets'
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset d'exemple balance-scale
    balance_data = {
        'feature1': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'feature2': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'feature3': [1, 2, 1, 2, 1, 2, 1, 2, 1],
        'class': ['L', 'B', 'R', 'L', 'B', 'R', 'L', 'B', 'R']
    }
    
    df = pd.DataFrame(balance_data)
    df.to_csv(os.path.join(data_dir, 'balance-scale.csv'), index=False)
    print("✓ Dataset d'exemple balance-scale créé")
    
    print("\\nNote: Ce sont des datasets d'exemple. Pour les vrais résultats,")
    print("téléchargez les datasets complets avec download_uci_datasets()")

if __name__ == "__main__":
    print("Téléchargement des datasets UCI...")
    try:
        download_uci_datasets()
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        print("Création de datasets d'exemple...")
        create_sample_datasets()
