import numpy as np
from .parametric import normalized_tsallis
from .distance_based import polarization_index

def itc_impurity(proportions, alpha=1.5, beta=3.5, gamma=0.3):
    """
    Integrated Tsallis Combination (ITC) - Notre métrique hybride
    """
    proportions = np.asarray(proportions)
    K = len(proportions)
    if K <= 1:
        return 0.0
    
    # Composante Tsallis normalisée
    tsallis_norm = normalized_tsallis(proportions, alpha=alpha)
    
    # Composante Polarisation
    polarization_norm = polarization_index(proportions, beta=beta)
    
    # Combinaison convexe
    return gamma * tsallis_norm + (1 - gamma) * polarization_norm

def shannon_polarization_hybrid(proportions, gamma=0.5, beta=3.5, epsilon=1e-15):
    """
    Hybridation Shannon + Polarisation
    """
    from .classical import shannon_entropy
    from .distance_based import polarization_index
    
    proportions = np.asarray(proportions)
    K = len(proportions)
    if K <= 1:
        return 0.0
    
    # Normalisation de Shannon
    shannon_val = shannon_entropy(proportions, epsilon)
    shannon_max = np.log2(K)
    shannon_norm = shannon_val / shannon_max
    
    # Composante Polarisation
    polarization_norm = polarization_index(proportions, beta=beta)
    
    return gamma * shannon_norm + (1 - gamma) * polarization_norm

def tsallis_hellinger_hybrid(proportions, alpha=1.5, gamma=0.5):
    """
    Hybridation Tsallis + Hellinger
    """
    from .parametric import normalized_tsallis
    from .distance_based import hellinger_distance
    
    proportions = np.asarray(proportions)
    tsallis_norm = normalized_tsallis(proportions, alpha=alpha)
    hellinger_val = hellinger_distance(proportions)
    
    return gamma * tsallis_norm + (1 - gamma) * (1 - hellinger_val)

# Dictionnaire des métriques hybrides
HYBRID_METRICS = {
    'itc': itc_impurity,
    'itc_alpha1.3': lambda p: itc_impurity(p, alpha=1.3, beta=3.5, gamma=0.3),
    'itc_alpha1.7': lambda p: itc_impurity(p, alpha=1.7, beta=3.5, gamma=0.3),
    'shannon_polarization': lambda p: shannon_polarization_hybrid(p, gamma=0.5, beta=3.5),
    'tsallis_hellinger': lambda p: tsallis_hellinger_hybrid(p, alpha=1.5, gamma=0.5)
}
