import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification


class DataLoader:
    def __init__(self):
        self.datasets = {}
        
    def load_sklearn_datasets(self):
        """Charge les datasets de sklearn"""
        self.datasets['iris'] = load_iris()
        self.datasets['wine'] = load_wine()
        self.datasets['breast_cancer'] = load_breast_cancer()
        self.datasets['digits'] = load_digits()
        print("Datasets sklearn chargés avec succès!")
        
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
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    self.datasets[name] = {'data': X, 'target': y}
                    print(f"Dataset UCI {name} chargé avec succès!")
                except Exception as e:
                    print(f"Erreur lors du chargement de {filename}: {e}")
    
    def create_synthetic_datasets(self):
        """
        Crée des datasets synthétiques pour le testing
        RETOURNE un dictionnaire de tuples (X, y) pour compatibilité avec le code existant
        """
        synthetic_datasets = {}
        
        print("Création des datasets synthétiques...")
        
        # Dataset binaire équilibré
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                  n_redundant=2, n_repeated=0, n_classes=2,
                                  n_clusters_per_class=1, weights=None, random_state=42)
        synthetic_datasets['binary_balanced'] = (X, y)
        self.datasets['binary_balanced'] = {'data': X, 'target': y}
        
        # Dataset binaire déséquilibré
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                  n_classes=2, weights=[0.9, 0.1], random_state=42)
        synthetic_datasets['binary_imbalanced'] = (X, y)
        self.datasets['binary_imbalanced'] = {'data': X, 'target': y}
        
        # Dataset multi-classes
        X, y = make_classification(n_samples=1500, n_features=12, n_informative=6,
                                  n_classes=4, n_clusters_per_class=1, random_state=42)
        synthetic_datasets['multiclass_4'] = (X, y)
        self.datasets['multiclass_4'] = {'data': X, 'target': y}
        
        print("✓ Datasets synthétiques créés avec succès!")
        return synthetic_datasets
    
    def get_dataset(self, name):
        """Retourne un dataset spécifique"""
        if name in self.datasets:
            data = self.datasets[name]
            if hasattr(data, 'data'):  # Pour les datasets sklearn
                return data.data, data.target
            else:  # Pour les datasets UCI et synthétiques
                return data['data'], data['target']
        else:
            raise ValueError(f"Dataset {name} non trouvé")
    
    def get_all_datasets(self):
        """Retourne tous les datasets sous forme de tuples (X, y)"""
        datasets = {}
        for name in self.datasets.keys():
            X, y = self.get_dataset(name)
            datasets[name] = (X, y)
        return datasets
    
    def get_dataset_info(self):
        """Affiche des informations sur tous les datasets chargés"""
        print("=== INFORMATIONS SUR LES DATASETS ===")
        for name in self.datasets.keys():
            try:
                X, y = self.get_dataset(name)
                n_samples, n_features = X.shape
                n_classes = len(np.unique(y))
                print(f"✓ {name}: {n_samples} échantillons, {n_features} features, {n_classes} classes")
            except Exception as e:
                print(f"✗ Erreur avec le dataset {name}: {e}")
    
    def prepare_train_test_splits(self, test_size=0.2, random_state=42):
        """Prépare des splits train/test pour tous les datasets"""
        splits = {}
        for name in self.datasets.keys():
            X, y = self.get_dataset(name)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            splits[name] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
        return splits

# =============================================================================
# FONCTIONS STANDALONE (en dehors de la classe)
# =============================================================================

def create_synthetic_datasets():
    """
    Fonction standalone pour créer des datasets synthétiques
    RETOURNE: dict avec les datasets sous forme de tuples (X, y)
    """
    synthetic_datasets = {}
    
    print("Création des datasets synthétiques (fonction standalone)...")
    
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
    
    print("✓ Datasets synthétiques créés avec succès (fonction standalone)!")
    return synthetic_datasets

def load_all_datasets(include_sklearn=True, include_uci=True, include_synthetic=True, uci_path='data/datasets/'):
    """
    Fonction utilitaire pour charger tous les types de datasets rapidement
    
    Args:
        include_sklearn (bool): Inclure les datasets sklearn
        include_uci (bool): Inclure les datasets UCI
        include_synthetic (bool): Inclure les datasets synthétiques
        uci_path (str): Chemin vers les datasets UCI
    
    Returns:
        DataLoader: Instance avec tous les datasets chargés
    """
    loader = DataLoader()
    
    if include_sklearn:
        loader.load_sklearn_datasets()
    
    if include_uci:
        loader.load_uci_datasets(uci_path)
    
    if include_synthetic:
        loader.create_synthetic_datasets()
    
    return loader

def get_available_datasets():
    """
    Retourne la liste de tous les datasets disponibles
    """
    base_datasets = ['iris', 'wine', 'breast_cancer', 'digits']
    uci_datasets = ['balance_scale', 'car_evaluation', 'chess', 'nursery']
    synthetic_datasets = ['binary_balanced', 'binary_imbalanced', 'multiclass_4']
    
    return {
        'sklearn': base_datasets,
        'uci': uci_datasets,
        'synthetic': synthetic_datasets,
        'all': base_datasets + uci_datasets + synthetic_datasets
    }

# =============================================================================
# FONCTION DE TEST
# =============================================================================

def test_data_loader():
    """Teste le DataLoader"""
    print("🧪 Test du DataLoader...")
    
    # Test 1: DataLoader classique
    print("\n1. Test avec DataLoader classique:")
    loader = DataLoader()
    loader.load_sklearn_datasets()
    synthetic_data = loader.create_synthetic_datasets()  # Doit retourner un dict
    print(f"Type de retour: {type(synthetic_data)}")
    print(f"Clés: {list(synthetic_data.keys())}")
    
    # Test 2: Fonction standalone
    print("\n2. Test avec fonction standalone:")
    standalone_data = create_synthetic_datasets()
    print(f"Type de retour: {type(standalone_data)}")
    print(f"Clés: {list(standalone_data.keys())}")
    
    # Test 3: Loader complet
    print("\n3. Test avec load_all_datasets:")
    full_loader = load_all_datasets()
    full_loader.get_dataset_info()
    
    print("\n✅ Tous les tests passés avec succès!")

if __name__ == "__main__":
    test_data_loader()