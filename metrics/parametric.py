import numpy as np

def renyi_entropy(proportions, alpha=0.5, epsilon=1e-15):
    """
    Entropie de Rényi d'ordre alpha
    """
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    if alpha == 1:
        return -np.sum(proportions * np.log2(proportions))
    else:
        return (1/(1-alpha)) * np.log2(np.sum(proportions**alpha))

def tsallis_entropy(proportions, alpha=1.5, epsilon=1e-15):
    """
    Entropie de Tsallis d'ordre alpha
    """
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    if alpha == 1:
        return -np.sum(proportions * np.log2(proportions))
    else:
        return (1 - np.sum(proportions**alpha)) / (alpha - 1)

def normalized_tsallis(proportions, alpha=1.5, epsilon=1e-15):
    """
    Entropie de Tsallis normalisée
    """
    K = len(proportions)
    if K <= 1:
        return 0.0
    
    tsallis_val = tsallis_entropy(proportions, alpha, epsilon)
    tsallis_max = (1 - K**(1-alpha)) / (alpha - 1) if alpha != 1 else np.log2(K)
    
    return tsallis_val / tsallis_max

def kumaraswamy_charlier(proportions, a=2.0, b=2.0):
    """
    Métrique basée sur la distribution de Kumaraswamy
    """
    return 1 - np.sum(proportions * (1 - (1 - proportions**a)**b))

# Dictionnaire des métriques paramétriques
PARAMETRIC_METRICS = {
    'renyi_0.5': lambda p: renyi_entropy(p, alpha=0.5),
    'renyi_2.0': lambda p: renyi_entropy(p, alpha=2.0),
    'tsallis_0.5': lambda p: tsallis_entropy(p, alpha=0.5),
    'tsallis_1.3': lambda p: tsallis_entropy(p, alpha=1.3),
    'tsallis_2.0': lambda p: tsallis_entropy(p, alpha=2.0),
    'norm_tsallis_1.3': lambda p: normalized_tsallis(p, alpha=1.3),
    'kumaraswamy': kumaraswamy_charlier
}
