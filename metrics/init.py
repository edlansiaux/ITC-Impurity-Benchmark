from .classical import CLASSICAL_METRICS
from .parametric import PARAMETRIC_METRICS
from .probabilistic import PROBABILISTIC_METRICS
from .distance_based import DISTANCE_METRICS
from .theoretical import THEORETICAL_METRICS
from .hybrid import HYBRID_METRICS

# Combinaison de toutes les métriques
ALL_METRICS = {
    **CLASSICAL_METRICS,
    **PARAMETRIC_METRICS,
    **PROBABILISTIC_METRICS,
    **DISTANCE_METRICS,
    **THEORETICAL_METRICS,
    **HYBRID_METRICS
}

# Métriques par catégorie pour l'analyse
METRICS_BY_CATEGORY = {
    'classical': CLASSICAL_METRICS,
    'parametric': PARAMETRIC_METRICS,
    'probabilistic': PROBABILISTIC_METRICS,
    'distance': DISTANCE_METRICS,
    'theoretical': THEORETICAL_METRICS,
    'hybrid': HYBRID_METRICS
}

__all__ = ['ALL_METRICS', 'METRICS_BY_CATEGORY']
