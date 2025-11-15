# attacks/__init__.py
from .byzantine import apply_byzantine
from .poison import PoisonController
from .aggregation import aggregate_with_byzantine, softmax_weights
from .robustness_metrics import byzantine_accuracy_degradation, dpa_eodd
