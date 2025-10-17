import numpy as np

def cross_entropy_impurity(proportions, epsilon=1e-15):
    """
    Perte d'entropie croisée par rapport à la distribution uniforme
    """
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    uniform = np.ones_like(proportions) / len(proportions)
    return -np.sum(uniform * np.log2(proportions))

def kl_divergence(proportions, epsilon=1e-15):
    """
    Divergence Kullback-Leibler par rapport à l'uniforme
    """
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    uniform = np.ones_like(proportions) / len(proportions)
    return np.sum(proportions * np.log2(proportions / uniform))

def jensen_shannon_divergence(proportions, epsilon=1e-15):
    """
    Divergence de Jensen-Shannon
    """
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    uniform = np.ones_like(proportions) / len(proportions)
    M = 0.5 * (proportions + uniform)
    return 0.5 * (np.sum(proportions * np.log2(proportions / M)) + 
                 np.sum(uniform * np.log2(uniform / M)))

# Dictionnaire des métriques probabilistes
PROBABILISTIC_METRICS = {
    'cross_entropy': cross_entropy_impurity,
    'kl_divergence': kl_divergence,
    'js_divergence': jensen_shannon_divergence
}
