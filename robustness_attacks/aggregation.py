# attacks/aggregation.py
from typing import Dict, List
import torch
from .byzantine import apply_byzantine

StateDict = Dict[str, torch.Tensor]

def state_dict_delta(global_sd: StateDict, client_sd: StateDict) -> StateDict:
    return {k: client_sd[k] - global_sd[k] for k in global_sd.keys()}

def add_inplace(dst: StateDict, src: StateDict, scale: float = 1.0):
    for k in dst:
        dst[k] += scale * src[k]

def zeros_like(sd: StateDict) -> StateDict:
    return {k: torch.zeros_like(v) for k, v in sd.items()}

def softmax_weights(ufms, beta: float):
    t = torch.tensor(ufms, dtype=torch.float32)
    w = torch.exp(-beta * t)
    w = w / (w.sum() + 1e-12)
    return w.tolist()

def aggregate_with_byzantine(
    global_sd: StateDict,
    client_sds: List[StateDict],
    client_ufms: List[float],
    beta: float,
    attacker_frac: float = 0.25,
    byz_mode: str = "sign_flip",
    **byz_kwargs
) -> StateDict:
    """
    1) build deltas
    2) choose attackers
    3) apply byzantine
    4) softmax(-beta * UFM) weighting
    5) global update
    """
    N = len(client_sds)
    deltas = [state_dict_delta(global_sd, sd) for sd in client_sds]
    # attackers
    num_attackers = max(1, int(attacker_frac * N))
    attacker_ids = list(range(num_attackers))  # deterministic for reproducibility; you can randomize
    deltas = apply_byzantine(deltas, attacker_ids, mode=byz_mode, **byz_kwargs)

    w = softmax_weights(client_ufms, beta)
    agg = zeros_like(global_sd)
    for i in range(N):
        add_inplace(agg, deltas[i], scale=w[i])

    new_global = {k: global_sd[k] + agg[k] for k in global_sd.keys()}
    return new_global
