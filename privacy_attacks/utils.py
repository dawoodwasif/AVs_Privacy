# privacy_attacks/utils.py
from typing import Dict, List, Tuple, Iterable, Optional
import torch
import numpy as np

@torch.no_grad()
def extract_output_features(
    model: torch.nn.Module,
    dataloader: Iterable,
    preprocess_fn=None,
    evidential_key: Optional[str] = "alpha",   # if your preds contain Dirichlet alphas
) -> np.ndarray:
    """
    Build per-example features from model outputs for MIA.
    Works with Ultralytics-style batches: dict with "img" and labels.
    Assumes model(batch) -> (loss, preds) or preds. We try to be permissive.
    Features:
      - loss (scalar; if available)
      - max_conf, mean_conf, num_dets (from predictions)
      - evidential stats: mean(alpha0), var(alpha0) if available
    Returns: [N, D] numpy array.
    """
    feats = []
    for batch in dataloader:
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)
        out = model(batch)
        loss_val = None
        preds = None
        if isinstance(out, tuple) and len(out) >= 2:
            loss_val = out[0]
            preds = out[1]
        else:
            preds = out

        # try to turn loss into per-batch scalar
        if loss_val is not None:
            try:
                loss_scalar = float(loss_val.detach().mean().cpu())
            except Exception:
                loss_scalar = 0.0
        else:
            loss_scalar = 0.0

        # prediction confidences
        # expect preds to have "conf" or a list of per-image dets with confidence scores
        max_conf, mean_conf, num_dets = 0.0, 0.0, 0.0
        alpha0_mean, alpha0_var = 0.0, 0.0

        # handle common Ultralytics predictor outputs
        # - if preds is a dict with "conf" tensor per image
        confs = []
        alphas0 = []

        if isinstance(preds, dict):
            if "conf" in preds:
                c = preds["conf"]
                if isinstance(c, torch.Tensor):
                    confs = c.detach().flatten().cpu().tolist()
            if evidential_key in preds:
                alpha = preds[evidential_key]  # shape [..., C] or [N, C]
                if isinstance(alpha, torch.Tensor):
                    a = alpha.detach().cpu()
                    if a.ndim >= 2:
                        a0 = a.sum(dim=-1).numpy().flatten()
                        alphas0 = a0.tolist()

        elif isinstance(preds, (list, tuple)):
            # list of per-image dicts or tensors
            for p in preds:
                if isinstance(p, dict) and "conf" in p and isinstance(p["conf"], torch.Tensor):
                    confs.extend(p["conf"].detach().cpu().tolist())
                if isinstance(p, dict) and evidential_key in p:
                    alpha = p[evidential_key]
                    if isinstance(alpha, torch.Tensor) and alpha.ndim >= 2:
                        alphas0.extend(alpha.sum(dim=-1).detach().cpu().numpy().flatten().tolist())

        # aggregate stats for this batch
        if len(confs) > 0:
            max_conf = float(np.max(confs))
            mean_conf = float(np.mean(confs))
            num_dets = float(len(confs))
        if len(alphas0) > 0:
            alpha0_mean = float(np.mean(alphas0))
            alpha0_var = float(np.var(alphas0))

        feats.append([loss_scalar, max_conf, mean_conf, num_dets, alpha0_mean, alpha0_var])

    if len(feats) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)

def extract_update_features(update_sd: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    Build a fixed-length vector from a client update state_dict for PIA.
    Features per layer:
      - l2 norm
      - sign ratio (fraction positive)
      - top-k index histogram (k=16, binned into 16 buckets)
    Concatenate across layers. Returns [D] numpy array.
    """
    feats: List[np.ndarray] = []
    for k, v in update_sd.items():
        t = v.detach().flatten()
        if t.numel() == 0:
            continue
        l2 = torch.norm(t, p=2).item()
        sign_ratio = float((t > 0).float().mean().item())

        # top-k indices histogram over 16 bins
        k_sel = min(16, t.numel())
        topk = torch.topk(t.abs(), k_sel, largest=True).indices.cpu().numpy()
        # map indices into 16 equal segments
        bins = np.linspace(0, t.numel(), 17)
        hist, _ = np.histogram(topk, bins=bins)
        hist = hist.astype(np.float32) / (hist.sum() + 1e-12)

        feats.append(np.concatenate([[l2, sign_ratio], hist], axis=0))
    if len(feats) == 0:
        return np.zeros((18,), dtype=np.float32)
    return np.concatenate(feats, axis=0)

def compute_tpr_at_fpr_points(scores: np.ndarray, labels: np.ndarray, fprs=(0.01, 0.05, 0.10)) -> Dict[float, float]:
    """
    Given anomaly-like 'member' scores (higher -> more likely member) and binary labels {0,1} with 1=member,
    compute TPR at fixed FPR operating points.
    """
    order = np.argsort(-scores)  # descending scores
    y = labels[order]
    pos = (y == 1).astype(np.int32)
    neg = (y == 0).astype(np.int32)

    tp_cum = np.cumsum(pos)
    fp_cum = np.cumsum(neg)
    P = pos.sum()
    N = neg.sum()

    # FPR curve: fp_cum / N
    # For each desired FPR, find index where FPR crosses it, then read TPR = tp_cum/P
    out = {}
    for fpr in fprs:
        threshold_idx = np.searchsorted(fp_cum / max(N, 1), fpr, side="right")
        if threshold_idx >= len(y):
            threshold_idx = len(y) - 1
        tpr = tp_cum[threshold_idx] / max(P, 1)
        out[float(fpr)] = float(tpr)
    return out
