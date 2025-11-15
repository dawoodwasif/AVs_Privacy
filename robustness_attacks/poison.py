# attacks/poison.py
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import yaml

@dataclass
class PoisonCfg:
    enabled: bool = False
    kind: str = "patch"      # "patch" or "dropout"
    frac: float = 0.2        # fraction of images or boxes
    size: int = 24           # patch size for "patch"
    target_group: int = 9    # MST group id for targeting in "dropout"

class PoisonController:
    def __init__(self, cfg: PoisonCfg):
        self.cfg = cfg

    @staticmethod
    def from_yaml(path: str) -> "PoisonController":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = PoisonCfg(**raw.get("poison", {}))
        return PoisonController(cfg)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, batch: Dict, metadata: Optional[Dict] = None) -> Dict:
        if not self.enabled:
            return batch
        if self.cfg.kind == "patch":
            return _patch_backdoor(batch, frac=self.cfg.frac, size=self.cfg.size)
        if self.cfg.kind == "dropout":
            return _box_dropout(batch, target_group=self.cfg.target_group, drop_frac=self.cfg.frac)
        return batch

def _patch_backdoor(batch: Dict, frac: float = 0.2, size: int = 24) -> Dict:
    """
    Add a small white square patch to a fraction of images.
    Assumes batch["img"] is [B,C,H,W] float in [0,1].
    """
    imgs = batch.get("img", None)
    if imgs is None:
        return batch
    B, C, H, W = imgs.shape
    sel = torch.rand(B, device=imgs.device) < frac
    ys = torch.randint(low=0, high=max(1, H - size), size=(B,), device=imgs.device)
    xs = torch.randint(low=0, high=max(1, W - size), size=(B,), device=imgs.device)
    for i in range(B):
        if sel[i]:
            y0 = int(ys[i].item()); x0 = int(xs[i].item())
            y1 = min(H, y0 + size); x1 = min(W, x0 + size)
            imgs[i, :, y0:y1, x0:x1] = 1.0
    batch["img"] = imgs
    return batch

def _box_dropout(batch: Dict, target_group: int, drop_frac: float = 0.3) -> Dict:
    """
    Remove a fraction of GT boxes for a target MST group.
    Requires batch["msts"] aligned with batch["bboxes"] per image.
    """
    if "msts" not in batch or "bboxes" not in batch:
        return batch
    # Ultralytics packs labels per image into lists of tensors
    msts_list = batch["msts"]
    bboxes_list = batch["bboxes"]
    cls_list = batch.get("cls", None)

    for i in range(len(msts_list)):
        msts = msts_list[i]
        if msts.numel() == 0:
            continue
        tgt = msts.eq(target_group)
        rnd = torch.rand_like(msts.float()) < drop_frac
        drop_mask = tgt & rnd
        keep = ~drop_mask
        bboxes_list[i] = bboxes_list[i][keep]
        if cls_list is not None:
            cls_list[i] = cls_list[i][keep]
        msts_list[i] = msts_list[i][keep]

    batch["msts"] = msts_list
    batch["bboxes"] = bboxes_list
    if cls_list is not None:
        batch["cls"] = cls_list
    return batch
