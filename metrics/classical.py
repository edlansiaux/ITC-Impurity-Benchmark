import numpy as np

def gini_impurity(proportions):
    """
    Impureté de Gini classique
    """
    return 1 - np.sum(proportions**2)

def shannon_entropy(proportions, epsilon=1e-15):
    """
    Entropie de Shannon
    """
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    return -np.sum(proportions * np.log2(proportions))

def misclassification_rate(proportions):
    """
    Taux d'erreur de classification
    """
    return 1 - np.max(proportions)

# Dictionnaire des métriques classiques
CLASSICAL_METRICS = {
    'gini': gini_impurity,
    'shannon': shannon_entropy,
    'misclassification': misclassification_rate
}
