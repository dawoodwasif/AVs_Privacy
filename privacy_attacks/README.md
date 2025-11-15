# Privacy Attacks (MIA & PIA)

A tiny module to evaluate privacy leakage in your FL runs. Implements:
- **MIA** (membership inference): score-based probe on model outputs. Reports Success Rate and **TPR@FPR** at {1%, 5%, 10%}.
- **PIA** (property inference): small MLP probe on client update features.

## Files
```
privacy_attacks/
  __init__.py
  mia.py              # MIARunner, MIAResult
  pia.py              # PIARunner, PIAResult
  utils.py            # feature extractors and TPR@FPR helper
  mia_pia_config.yaml # optional defaults
```
Import path assumes `privacy_attacks/` is on `PYTHONPATH` (or inside your repo).

## Quick start

### MIA (membership inference)
```python
from privacy_attacks import MIARunner

# model: trained global model (eval mode is set inside)
# member_loader: samples from train set
# nonmember_loader: disjoint holdout
mia = MIARunner(model, preprocess_fn=trainer.preprocess_batch)
res = mia.run(member_loader, nonmember_loader, fpr_points=(0.01,0.05,0.10), train_frac=0.5, seed=0)

print("MIA Success Rate:", res.success_rate)     # balanced accuracy
print("MIA TPR@FPR:", res.tpr_at_fpr)            # dict {0.01: TPR, 0.05: TPR, 0.10: TPR}
```

Features used by default:
- loss (if available), max and mean detection confidence, number of detections
- evidential alpha0 mean and variance when the model outputs Dirichlet alphas

### PIA (property inference)
```python
import numpy as np
from privacy_attacks import PIARunner

# client_updates: list of state_dict deltas from a single round
# property_labels: np.array of ints per client (e.g., 0/1 for thresholded group mix)
pia = PIARunner()
res = pia.run(client_updates, property_labels, train_frac=0.6, seed=0)

print("PIA Acc:", res.acc)
print("PIA Macro-F1:", res.macro_f1)
```

Update features per layer:
- L2 norm, sign ratio, top-k index histogram (k=16) over 16 bins
- Concatenated across layers and padded to the same length

## Integration tips
- Reuse your trainer’s `preprocess_batch` so the model sees identical tensors.
- For MIA, keep member and non-member sets balanced. Do not reuse samples used to fit the attack probe.
- For PIA, form deltas as `client_sd - global_sd` right after local training and before aggregation.
- Store round, seed, and attack config with results for reproducibility.

## Outputs to report
- **MIA:** Success Rate and TPR at FPR ∈ {1%, 5%, 10%}.
- **PIA:** Accuracy and Macro-F1.

## Configurations
See `privacy_attacks/mia_pia_config.yaml` for default seeds, FPR points, and probe settings.
