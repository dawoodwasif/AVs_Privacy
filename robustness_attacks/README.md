# Robustness Attacks (Byzantine & Poisoning Attacks)


Robustness attacks for all FL baselines:
- **Byzantine (server-side)**: `sign_flip` (scale γ), `l2` (fixed L2 ε).
- **Data poisoning (client-side)**: `patch` backdoor (square patch on images), `dropout` (remove GT boxes for a target MST group).
- **Metrics**: BA AD (Byzantine Accuracy Degradation) and DPA EODD (Δ Equalized Odds under Poisoning).

## Files
- `byzantine.py` — sign-flip and L2-bounded update attacks.
- `poison.py` — patch backdoor + box dropout; `PoisonController` entry point.
- `aggregation.py` — UFM softmax weights and delta aggregation with optional Byzantine.
- `robustness_metrics.py` — BA AD and DPA EODD helpers.
- `attack_config.yaml` — example config.

## Quick Start
1) **Place the folder**
```
your_repo/
  attacks/
    __init__.py
    byzantine.py
    poison.py
    aggregation.py
    robustness_metrics.py
    attack_config.yaml
```
2) **Client-side poisoning** (Ultralytics trainer)
```python
# in your custom trainer __init__
from attacks.poison import PoisonController
self.poison_ctrl = PoisonController.from_yaml("attacks/attack_config.yaml")

# in the training loop after preprocess
batch = self.preprocess_batch(batch)
if self.poison_ctrl.enabled:
    batch = self.poison_ctrl.apply(batch)
```
3) **Server-side Byzantine** (during aggregation)
```python
from attacks.aggregation import aggregate_with_byzantine
global_sd = model.model.state_dict()
client_sds = [m.model.state_dict() for m in client_models]
client_ufms = fairness_metrics  # one scalar per client
new_global = aggregate_with_byzantine(
    global_sd, client_sds, client_ufms,
    beta=2.0, attacker_frac=0.25, byz_mode="sign_flip", gamma=3.0
)
model.model.load_state_dict(new_global)
```
4) **Config** (`attacks/attack_config.yaml`)
```yaml
byzantine:
  enabled: true
  mode: sign_flip      # sign_flip | l2
  attacker_frac: 0.25
  gamma: 3.0           # sign_flip
  epsilon: 2.0         # l2

poison:
  enabled: true
  kind: patch          # patch | dropout
  frac: 0.20           # images (patch) or boxes (dropout)
  size: 24             # patch size
  target_group: 9      # MST group for dropout
```
5) **Robustness metrics**
```python
from attacks.robustness_metrics import byzantine_accuracy_degradation, dpa_eodd
ba_ad = byzantine_accuracy_degradation(mAP_clean, mAP_attacked)
dpa   = dpa_eodd(eodd_clean, eodd_poisoned)
```

## Configurations

```python
# enable or disable quickly
byzantine: {enabled: true,  mode: sign_flip, attacker_frac: 0.25, gamma: 3.0}   # use 'l2' with epsilon
poison:    {enabled: true,  kind: patch,     frac: 0.20,     size: 24, target_group: 9}

# examples:
# byzantine: {enabled: true,  mode: l2,        attacker_frac: 0.25, epsilon: 2.0}
# poison:    {enabled: true,  kind: dropout,   frac: 0.30, target_group: 9}
# to disable: set enabled: false
```

