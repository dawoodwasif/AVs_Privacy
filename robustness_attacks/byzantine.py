# attacks/byzantine.py
from typing import Dict, List
import torch

StateDict = Dict[str, torch.Tensor]

def _clone_delta(delta: StateDict) -> StateDict:
    return {k: v.clone() for k, v in delta.items()}

def sign_flip(delta: StateDict, gamma: float = 1.0) -> StateDict:
    out = _clone_delta(delta)
    for k in out:
        out[k] = -gamma * out[k]
    return out

def l2_bounded(delta: StateDict, epsilon: float = 1.0) -> StateDict:
    out = _clone_delta(delta)
    flat = torch.cat([p.reshape(-1) for p in out.values()])
    # random direction with fixed L2 epsilon
    r = torch.randn_like(flat)
    r = epsilon * r / (torch.norm(r, p=2) + 1e-12)
    idx = 0
    for k, p in out.items():
        n = p.numel()
        out[k] = p + r[idx:idx+n].reshape_as(p)
        idx += n
    return out

def apply_byzantine(
    client_deltas: List[StateDict],
    attacker_ids: List[int],
    mode: str = "sign_flip",
    **kwargs
) -> List[StateDict]:
    out = [ _clone_delta(d) for d in client_deltas ]
    for i in attacker_ids:
        if mode == "sign_flip":
            out[i] = sign_flip(out[i], gamma=kwargs.get("gamma", 3.0))
        elif mode == "l2":
            out[i] = l2_bounded(out[i], epsilon=kwargs.get("epsilon", 2.0))
        else:
            raise ValueError("Unknown byzantine mode. Use 'sign_flip' or 'l2'.")
    return out
