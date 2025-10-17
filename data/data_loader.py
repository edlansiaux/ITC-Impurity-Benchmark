import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self):
        self.datasets = {}
        
    def load_sklearn_datasets(self):
        """Charge les datasets de sklearn"""
        self.datasets['iris'] = load_iris()
        self.datasets['wine'] = load_wine()
        self.datasets['breast_cancer'] = load_breast_cancer()
        self.datasets['digits'] = load_digits()
        
    def load_uci_datasets(self, data_path='data/datasets/'):
        """Charge les datasets UCI depuis des fichiers CSV"""
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Veuillez placer les datasets UCI dans le dossier {data_path}")
            return
            
        uci_files = {
            'balance_scale': 'balance-scale.csv',
            'car_evaluation': 'car.csv',
            'chess': 'chess.csv',
            'nursery': 'nursery.csv'
        }
        
        for name, filename in uci_files.items():
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # Supposons que la dernière colonne est la target
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values
                    # Encoder les labels si nécessaire
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    self.datasets[name] = {'data': X, 'target': y}
                except Exception as e:
                    print(f"Erreur lors du chargement de {filename}: {e}")
    
    def get_dataset(self, name):
        """Retourne un dataset spécifique"""
        if name in self.datasets:
            data = self.datasets[name]
            if hasattr(data, 'data'):
                return data.data, data.target
            else:
                return data['data'], data['target']
        else:
            raise ValueError(f"Dataset {name} non trouvé")
    
    def get_all_datasets(self):
        """Retourne tous les datasets"""
        datasets = {}
        for name in self.datasets.keys():
            X, y = self.get_dataset(name)
            datasets[name] = (X, y)
        return datasets

def create_synthetic_datasets():
    """Crée des datasets synthétiques pour le testing"""
    from sklearn.datasets import make_classification
    
    synthetic_datasets = {}
    
    # Dataset binaire équilibré
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, n_repeated=0, n_classes=2,
                              n_clusters_per_class=1, weights=None, random_state=42)
    synthetic_datasets['binary_balanced'] = (X, y)
    
    # Dataset binaire déséquilibré
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_classes=2, weights=[0.9, 0.1], random_state=42)
    synthetic_datasets['binary_imbalanced'] = (X, y)
    
    # Dataset multi-classes
    X, y = make_classification(n_samples=1500, n_features=12, n_informative=6,
                              n_classes=4, n_clusters_per_class=1, random_state=42)
    synthetic_datasets['multiclass_4'] = (X, y)
    
    return synthetic_datasets
