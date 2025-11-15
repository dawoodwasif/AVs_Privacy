# privacy_attacks/pia.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import extract_update_features

@dataclass
class PIAResult:
    acc: float
    macro_f1: float

class _MLPProbe(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 128, n_out: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Linear(d_hid, n_out)
        )

    def forward(self, x):
        return self.net(x)

def _f1_macro(y_true, y_pred, n_classes=2):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred==c) & (y_true==c))
        fp = np.sum((y_pred==c) & (y_true!=c))
        fn = np.sum((y_pred!=c) & (y_true==c))
        prec = tp / (tp+fp+1e-12)
        rec = tp / (tp+fn+1e-12)
        f1 = 2*prec*rec / (prec+rec+1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

class PIARunner:
    """
    Property inference from client updates.
    You provide:
      - a list of client update state_dicts (same round)
      - a vector of property labels per client (e.g., 0/1 for "dark-skin proportion > τ")
    We train a small MLP probe on a split and evaluate accuracy and macro-F1.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        property_labels: np.ndarray,
        train_frac: float = 0.6,
        seed: int = 0,
        epochs: int = 100,
        lr: float = 1e-3,
        wd: float = 1e-4,
        hidden: int = 128,
    ) -> PIAResult:
        np.random.seed(seed)
        torch.manual_seed(seed)

        X_list = [extract_update_features(sd) for sd in client_updates]
        # pad to same length
        dmax = max(x.shape[0] for x in X_list)
        X_pad = np.stack([np.pad(x, (0, dmax - x.shape[0])) for x in X_list], axis=0).astype(np.float32)
        y = property_labels.astype(np.int64)
        n = len(y)
        idx = np.arange(n); np.random.shuffle(idx)
        cut = int(train_frac * n)
        tr, va = idx[:cut], idx[cut:]
        Xtr, ytr = X_pad[tr], y[tr]
        Xva, yva = X_pad[va], y[va]

        model = _MLPProbe(d_in=dmax, d_hid=hidden, n_out=int(y.max()+1)).to(self.device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        ce = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=self.device)
        yt = torch.tensor(ytr, dtype=torch.long, device=self.device)
        Xv = torch.tensor(Xva, dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = ce(model(Xt), yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xv).cpu().numpy()
        yhat = logits.argmax(axis=1)
        acc = float(np.mean(yhat == yva))
        macro_f1 = _f1_macro(yva, yhat, n_classes=int(y.max()+1))
        return PIAResult(acc=acc, macro_f1=macro_f1)
