import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import time

class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, impurity_measure='gini', max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0, random_state=None, **impurity_params):
        self.impurity_measure = impurity_measure
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.impurity_params = impurity_params
        self.tree_ = None
        self.feature_importances_ = None
        self.training_time_ = None
        
    def fit(self, X, y):
        start_time = time.time()
        X, y = check_X_y(X, y)
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.tree_ = self._build_tree(X, y, depth=0)
        self.training_time_ = time.time() - start_time
        self._compute_feature_importances()
        return self
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Calcul de l'impureté courante
        current_impurity = self._calculate_impurity(y)
        
        # Critères d'arrêt
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1 or current_impurity == 0):
            return self._make_leaf_node(y)
        
        best_impurity_reduction = -np.inf
        best_split = None
        
        # Recherche de la meilleure split
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                impurity_reduction = self._calculate_impurity_reduction(
                    y, left_mask, right_mask, current_impurity
                )
                
                if (impurity_reduction > best_impurity_reduction and 
                    impurity_reduction >= self.min_impurity_decrease):
                    best_impurity_reduction = impurity_reduction
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        if best_split is None:
            return self._make_leaf_node(y)
            
        # Construction récursive des sous-arbres
        left_tree = self._build_tree(
            X[best_split['left_mask']], y[best_split['left_mask']], depth + 1
        )
        right_tree = self._build_tree(
            X[best_split['right_mask']], y[best_split['right_mask']], depth + 1
        )
        
        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree,
            'impurity_reduction': best_impurity_reduction
        }
    
    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0
            
        proportions = np.bincount(y) / len(y)
        
        # Sélection de la métrique d'impureté
        from metrics import ALL_METRICS
        if self.impurity_measure in ALL_METRICS:
            return ALL_METRICS[self.impurity_measure](proportions, **self.impurity_params)
        else:
            raise ValueError(f"Métrique d'impureté non reconnue: {self.impurity_measure}")
    
    def _calculate_impurity_reduction(self, y, left_mask, right_mask, current_impurity):
        n_total = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        impurity_left = self._calculate_impurity(y[left_mask])
        impurity_right = self._calculate_impurity(y[right_mask])
        
        weighted_impurity = (n_left / n_total) * impurity_left + (n_right / n_total) * impurity_right
        return current_impurity - weighted_impurity
    
    def _make_leaf_node(self, y):
        if len(y) == 0:
            return {'class': 0, 'samples': 0}
        
        class_counts = np.bincount(y)
        predicted_class = np.argmax(class_counts)
        
        return {
            'class': predicted_class,
            'samples': len(y),
            'class_distribution': class_counts / len(y)
        }
    
    def predict(self, X):
        X = check_array(X)
        return np.array([self._predict_single(x, self.tree_) for x in X])
    
    def _predict_single(self, x, node):
        if 'class' in node:
            return node['class']
            
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def _compute_feature_importances(self):
        self.feature_importances_ = np.zeros(self.n_features_)
        self._accumulate_feature_importance(self.tree_)
        
        # Normalisation
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
    
    def _accumulate_feature_importance(self, node):
        if 'feature_idx' in node:
            feature_idx = node['feature_idx']
            self.feature_importances_[feature_idx] += node['impurity_reduction']
            self._accumulate_feature_importance(node['left'])
            self._accumulate_feature_importance(node['right'])
    
    def get_tree_depth(self, node=None):
        if node is None:
            node = self.tree_
            
        if 'class' in node:
            return 0
            
        left_depth = self.get_tree_depth(node['left'])
        right_depth = self.get_tree_depth(node['right'])
        
        return 1 + max(left_depth, right_depth)
    
    def get_tree_size(self, node=None):
        if node is None:
            node = self.tree_
            
        if 'class' in node:
            return 1
            
        return 1 + self.get_tree_size(node['left']) + self.get_tree_size(node['right'])
