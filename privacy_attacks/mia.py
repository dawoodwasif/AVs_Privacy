# privacy_attacks/mia.py
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import extract_output_features, compute_tpr_at_fpr_points

@dataclass
class MIAResult:
    success_rate: float
    tpr_at_fpr: Dict[float, float]

class _LogReg(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.lin(x)

def _train_probe(Xtr, ytr, Xva, yva, epochs=100, lr=1e-2, wd=1e-4, device="cpu"):
    model = _LogReg(Xtr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr.reshape(-1,1), dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    yv = torch.tensor(yva.reshape(-1,1), dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt)
        loss = bce(out, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xv).cpu().numpy().ravel()
    return model, logits

class MIARunner:
    """
    Score-based MIA with a simple logistic probe on output features.
    Usage:
        mia = MIARunner(model, preprocess_fn=your_preprocess)
        res = mia.run(member_loader, nonmember_loader)
    """
    def __init__(self, model: torch.nn.Module, preprocess_fn=None, device: Optional[str] = None):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def run(
        self,
        member_loader: Iterable,
        nonmember_loader: Iterable,
        fpr_points=(0.01, 0.05, 0.10),
        train_frac=0.5,
        seed: int = 0,
    ) -> MIAResult:
        np.random.seed(seed)
        torch.manual_seed(seed)

        X_m = extract_output_features(self.model, member_loader, self.preprocess_fn)
        X_n = extract_output_features(self.model, nonmember_loader, self.preprocess_fn)

        # balance and split
        n = min(len(X_m), len(X_n))
        X_m = X_m[:n]; X_n = X_n[:n]
        X = np.concatenate([X_m, X_n], axis=0)
        y = np.concatenate([np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32)], axis=0)

        idx = np.arange(2*n); np.random.shuffle(idx)
        tr_cut = int(train_frac * 2*n)
        tr_idx, va_idx = idx[:tr_cut], idx[tr_cut:]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        _, logits = _train_probe(Xtr, ytr, Xva, yva, device=self.device)
        # balanced accuracy
        scores = 1/(1+np.exp(-logits))
        yhat = (scores >= 0.5).astype(np.int32)
        tp = ((yhat==1) & (yva==1)).sum()
        tn = ((yhat==0) & (yva==0)).sum()
        fp = ((yhat==1) & (yva==0)).sum()
        fn = ((yhat==0) & (yva==1)).sum()
        succ = 0.5*((tp/(tp+fn+1e-12)) + (tn/(tn+fp+1e-12)))

        tpr_map = compute_tpr_at_fpr_points(scores, yva, fprs=fpr_points)
        return MIAResult(success_rate=float(succ), tpr_at_fpr=tpr_map)
