# attacks/robustness_metrics.py
def byzantine_accuracy_degradation(mAP_clean: float, mAP_attacked: float) -> float:
    return max(0.0, (mAP_clean - mAP_attacked) / max(mAP_clean, 1e-12))

def dpa_eodd(eodd_clean: float, eodd_poisoned: float) -> float:
    return max(0.0, eodd_poisoned - eodd_clean)
