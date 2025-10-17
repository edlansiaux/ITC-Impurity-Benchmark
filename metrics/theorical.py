import numpy as np

def bregman_divergence(proportions, f_type='squared', epsilon=1e-15):
    """
    Divergence de Bregman avec différentes fonctions génératrices
    """
    uniform = np.ones_like(proportions) / len(proportions)
    
    if f_type == 'squared':
        # f(x) = x²
        f = lambda x: np.sum(x**2)
        f_grad = lambda x: 2*x
    elif f_type == 'entropy':
        # f(x) = x log x
        f = lambda x: np.sum(x * np.log2(np.clip(x, epsilon, 1)))
        f_grad = lambda x: np.log2(np.clip(x, epsilon, 1)) + 1/np.log(2)
    elif f_type == 'exponential':
        # f(x) = exp(x)
        f = lambda x: np.sum(np.exp(x))
        f_grad = lambda x: np.exp(x)
    else:
        raise ValueError("Type de fonction non supporté")
    
    return f(proportions) - f(uniform) - np.dot(f_grad(uniform), proportions - uniform)

# Dictionnaire des métriques théoriques
THEORETICAL_METRICS = {
    'bregman_squared': lambda p: bregman_divergence(p, f_type='squared'),
    'bregman_entropy': lambda p: bregman_divergence(p, f_type='entropy')
}
