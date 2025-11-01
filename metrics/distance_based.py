import numpy as np

def hellinger_distance(proportions, epsilon=1e-15):
    """
    Distance de Hellinger par rapport à l'uniforme
    """
    proportions = np.asarray(proportions)
    proportions = np.clip(proportions, epsilon, 1 - epsilon)
    uniform = np.ones_like(proportions) / len(proportions)
    return np.sqrt(0.5 * np.sum((np.sqrt(proportions) - np.sqrt(uniform))**2))

def energy_distance(proportions):
    """
    Distance énergétique par rapport à l'uniforme
    """
    proportions = np.asarray(proportions)
    n = len(proportions)
    if n <= 1:
        return 0
    # Implémentation simplifiée pour les distributions discrètes
    sorted_prop = np.sort(proportions)
    uniform = np.ones(n) / n
    return np.sqrt(np.sum((sorted_prop - uniform)**2))

def polarization_index(proportions, beta=3.5):
    """
    Indice de polarisation avec décroissance exponentielle
    """
    proportions = np.asarray(proportions)
    K = len(proportions)
    if K <= 1:
        return 0.0
    
    p_bar = 1.0 / K
    polarization = np.sum(np.abs(proportions - p_bar) * np.exp(-beta * np.abs(proportions - p_bar)))
    polarization_max = (2 * (K - 1) / K) * (1 - np.exp(-beta))
    
    return 1 - (polarization / polarization_max)

# Dictionnaire des métriques basées sur la distance
DISTANCE_METRICS = {
    'hellinger': hellinger_distance,
    'energy': energy_distance,
    'polarization_3.5': lambda p: polarization_index(p, beta=3.5)
}
