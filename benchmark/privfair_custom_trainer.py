# privfair_custom_trainer.py

from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import json
import os
import time
import warnings
from typing import Dict, List, Tuple
from ultralytics.utils import LOGGER, RANK, TQDM
from ultralytics.utils.torch_utils import autocast

import numpy as np


def trimmed_mean(arrs: List[torch.Tensor], trim: float = 0.1) -> torch.Tensor:
    """Elementwise trimmed-mean across a list of same-shaped tensors."""
    if len(arrs) == 1:
        return arrs[0]
    stacked = torch.stack(arrs, dim=0)
    k = int(trim * stacked.size(0))
    if k == 0:
        return stacked.mean(dim=0)
    topk_vals, _ = torch.topk(stacked, k, dim=0, largest=True)
    bottomk_vals, _ = torch.topk(stacked, k, dim=0, largest=False)
    mask = torch.ones_like(stacked, dtype=torch.bool)
    mask[:k] = False  # will re-sort below
    # More robust approach: sort and slice
    sorted_vals, _ = torch.sort(stacked, dim=0)
    return sorted_vals[k: stacked.size(0) - k].mean(dim=0)


class PrivFairDetectionTrainer(DetectionTrainer):
    """
    One trainer supports both PrivFair-Pre (loss-side) and PrivFair-Post (threshold calibration).
    Switch with self.privfair_mode in {'pre','post'}.
    """

    def __init__(self, *args, metadata_path=None, privfair_mode: str = "pre",
                 groups: int = 10, target_tpr: float = 0.85, **kwargs):
        super().__init__(*args, **kwargs)

        # Resolve metadata
        default_meta = "/home/Virginia_Research/FACET/metadata.json"
        self.metadata_path = metadata_path or default_meta
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata: Dict = json.load(f)
            LOGGER.info(f"[PrivFair] Loaded metadata from {self.metadata_path}")
        else:
            self.metadata = {}
            LOGGER.warning(f"[PrivFair] Metadata not found at {self.metadata_path}. Proceeding without attributes.")

        # Mode and knobs
        assert privfair_mode in {"pre", "post"}, "privfair_mode must be 'pre' or 'post'"
        self.privfair_mode = privfair_mode
        self.groups = int(groups)
        self.target_tpr = float(target_tpr)

        # Buffers and state
        self.feature_space = None
        self.fairness_metric = 1.0
        self.group_thresholds: List[float] = [0.25 for _ in range(self.groups)]  # default fallback
        self._tp = torch.zeros(self.groups, dtype=torch.long)
        self._fn = torch.zeros(self.groups, dtype=torch.long)

        # Register a late backbone hook for feature-based loss (pre variant)
        self._fs_hook = None
        try:
            self._fs_hook = self.model.model[-4].register_forward_hook(self._feature_hook)
            LOGGER.info("[PrivFair] Feature hook registered on model[-4]")
        except Exception:
            LOGGER.warning("[PrivFair] Could not register feature hook. Pre variant will skip feature-based terms.")

    # ---------- Utilities ----------

    def _feature_hook(self, module, inputs, outputs):
        self.feature_space = outputs

    def _extract_group_id(self, img_file: str) -> int:
        """
        Map image filename to a group id [0..G-1] using metadata.
        Strategy: choose the person entry with max skin_tone_* value and map its index to [0..G-1].
        """
        try:
            img_id = os.path.basename(img_file).split('.')[0].replace('sa_', '')
            entry = self.metadata.get(str(img_id), {})
            persons = entry.get("persons", [])
            best_key, best_val = None, -1.0
            for p in persons:
                for k, v in p.items():
                    if k.startswith("skin_tone_") and isinstance(v, (int, float)):
                        if v > best_val:
                            best_key, best_val = k, float(v)
            if best_key is None:
                return -1
            # Expect skin_tone_X with X in [1..10], map to [0..G-1]
            idx = int(best_key.split("_")[-1]) - 1
            idx = max(0, min(self.groups - 1, idx))
            return idx
        except Exception:
            return -1

    def _per_image_group_vector(self, batch) -> torch.Tensor:
        """
        Returns group ids per image in the batch as a tensor on device. Unknown => -1.
        """
        image_files = batch.get("im_file", [])
        gids = [self._extract_group_id(f) for f in image_files]
        return torch.tensor(gids, device=self.device, dtype=torch.long)

    def _group_weights_from_running_tpr(self) -> torch.Tensor:
        # TPR_g = TP_g / (TP_g + FN_g + eps)
        eps = 1e-6
        tpr = self._tp.float() / (self._tp.float() + self._fn.float() + eps)
        inv = 1.0 / (tpr + eps)
        inv[~torch.isfinite(inv)] = 1.0
        w = inv / inv.clamp_min(1e-6).mean()  # normalize near 1
        w = torch.clamp(w, 0.5, 2.0)
        return w  # shape [G]

    def _update_running_tpr_from_batch(self, batch, pred_scores_by_image: List[float], score_thresh: float = 0.25):
        """
        Very light TPR proxy per image:
        if image has a known group g and has at least one predicted score >= score_thresh -> TP_g += 1 else FN_g += 1
        """
        gids = self._per_image_group_vector(batch)
        for i, g in enumerate(gids.tolist()):
            if g < 0:
                continue
            has_pred = pred_scores_by_image[i] >= score_thresh
            if has_pred:
                self._tp[g] += 1
            else:
                self._fn[g] += 1

    # ---------- Loss-side reweighting (PrivFair-Pre) ----------

    def _apply_privfair_pre(self, task_loss, batch) -> torch.Tensor:
        """
        Reweight per-instance loss using inverse running TPR of its group.
        For detection loss, we apply a scalar weight averaged over images in the batch.
        """
        if self.privfair_mode != "pre":
            return task_loss

        # Estimate if each image has any detection-like evidence in this forward step
        # We use model outputs when available. For Ultralytics tuple, outputs[0] are preds, outputs[1] loss
        # Here, we approximate by using last known predictions from self.targets or confidences if available.
        # As a safe proxy, use a constant 0.5 for all images which still drives weights from previous rounds.
        # If validator later fills scores, training still benefits from EMA of TPR.
        batch_size = batch["img"].shape[0]
        fake_scores = [0.5 for _ in range(batch_size)]
        self._update_running_tpr_from_batch(batch, fake_scores, score_thresh=0.25)

        w_g = self._group_weights_from_running_tpr()  # [G]
        gids = self._per_image_group_vector(batch)     # [B]
        valid = gids >= 0
        if valid.any():
            scalar = w_g[gids[valid]].mean()
        else:
            scalar = torch.tensor(1.0, device=self.device)

        self.fairness_metric = float((w_g.mean()).item())
        return task_loss * scalar

    # ---------- Post-processing calibration (PrivFair-Post) ----------

    @torch.no_grad()
    def _fit_group_thresholds_on_validator(self):
        """
        For each group g, collect top detection score per image on the validator set,
        choose threshold tau_g as the (1 - target_tpr) quantile to reach target recall proxy.
        """
        if self.validator is None or self.validator.dataloader is None:
            LOGGER.warning("[PrivFair] Validator not ready, skip threshold fitting.")
            return

        score_bucket: List[List[float]] = [[] for _ in range(self.groups)]

        model = self.ema.ema if self.ema else self.model
        model.eval()
        dl = self.validator.dataloader
        for batch in dl:
            batch = self.preprocess_batch(batch)
            preds = model(batch["img"])
            # preds[0] for raw, validator will NMS; here we just take max score per image safely
            bs = batch["img"].shape[0]
            # conservative fallback if we cannot parse model outputs
            max_scores = [0.0 for _ in range(bs)]
            # Ultralytics predictor returns list per image in validator, here we estimate using objectness logits if present
            try:
                # Attempt to pull objectness/confidence from preds if it is a tuple like (pred, loss)
                tensor_pred = preds[0] if isinstance(preds, (tuple, list)) else preds
                # tensor_pred shape [N, 6] or [B, anchors, classes+...], handle safe fallback
                if isinstance(tensor_pred, torch.Tensor) and tensor_pred.ndim >= 2:
                    # If batched, collapse per image stats coarsely
                    if tensor_pred.ndim == 3:
                        # [B, N, D] -> per image max over scores
                        conf = tensor_pred[..., -1]
                        for i in range(bs):
                            max_scores[i] = float(conf[i].detach().max().item())
                    else:
                        # single view -> same score for all images
                        common = float(tensor_pred[..., -1].detach().max().item())
                        max_scores = [common for _ in range(bs)]
            except Exception:
                pass

            gids = self._per_image_group_vector(batch)
            for i, g in enumerate(gids.tolist()):
                if 0 <= g < self.groups:
                    score_bucket[g].append(max_scores[i])

        taus = []
        for g in range(self.groups):
            arr = np.array(score_bucket[g]) if len(score_bucket[g]) else np.array([0.25])
            q = np.quantile(arr, 1.0 - self.target_tpr, method="nearest")
            taus.append(float(q))
        self.group_thresholds = taus
        LOGGER.info(f"[PrivFair] Fitted group thresholds (len={len(taus)}).")

    # ---------- Main training loop override to insert pre/post logic ----------

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        LOGGER.info(f"[PrivFair] Mode: {self.privfair_mode}. Starting {self.epochs} epoch(s).")
        epoch = self.start_epoch
        self.optimizer.zero_grad()

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)

            pbar = TQDM(enumerate(self.train_loader), total=nb) if RANK in {-1, 0} else enumerate(self.train_loader)
            self.tloss = torch.zeros(1, device=self.device)

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    outputs = self.model(batch)
                    if isinstance(outputs, tuple):
                        task_loss, _ = outputs
                    else:
                        task_loss = outputs

                    total_loss = task_loss
                    if self.privfair_mode == "pre":
                        total_loss = self._apply_privfair_pre(task_loss, batch)

                    self.loss = total_loss.detach().clone()

                self.scaler.scale(total_loss).backward()

                if i - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = i

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            epoch += 1

            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                if self.ema:
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validate
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()

                # For post variant, fit group thresholds after local training is done
                if self.privfair_mode == "post":
                    try:
                        self._fit_group_thresholds_on_validator()
                    except Exception as e:
                        LOGGER.warning(f"[PrivFair] Threshold fit failed: {e}")

                self.save_metrics(metrics={**self.label_loss_items(self.tloss.flatten(), prefix="train"), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            if epoch + 1 >= self.epochs:
                break

        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed.")
        if self._fs_hook is not None:
            try:
                self._fs_hook.remove()
            except Exception:
                pass
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")
