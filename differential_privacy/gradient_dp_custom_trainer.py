# custom_trainer.py
from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import numpy as np

# NEW: Opacus for example-level DP-SGD
try:
    from opacus import PrivacyEngine
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False


# Optional: fallback defaults if not passed via args/overrides
_DEFAULT_DP_NOISE_MULTIPLIER = 1.25  # tune per your PRV accounting sweep
_DEFAULT_DP_MAX_GRAD_NORM   = 1.0    # per-example clipping bound


class CustomDetectionTrainer(DetectionTrainer):
    """
    Minimal modification of DetectionTrainer to enable example-level DP-SGD via Opacus.
    - Lazily attaches a PrivacyEngine on the first optimizer step.
    - Uses per-example clipping (max_grad_norm) and Gaussian noise (noise_multiplier).
    - Preserves AMP gradient scaling and EMA.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # Internal flag so we only attach the PrivacyEngine once
        self._dp_initialized = False

        # Read DP knobs from args if provided; else use safe defaults
        self.dp_noise_multiplier = getattr(self.args, "dp_noise_multiplier", _DEFAULT_DP_NOISE_MULTIPLIER)
        self.dp_max_grad_norm    = getattr(self.args, "dp_max_grad_norm", _DEFAULT_DP_MAX_GRAD_NORM)
        self.enable_dp_sgd       = bool(getattr(self.args, "enable_dp_sgd", True))  # turn off by setting to False

        # Sanity if Opacus is not present
        if self.enable_dp_sgd and not _HAS_OPACUS:
            raise ImportError(
                "Opacus is required for example-level DP-SGD. Please install with `pip install opacus` "
                "or set --enable_dp_sgd False."
            )

    def _maybe_init_privacy_engine(self):
        """
        Initialize Opacus PrivacyEngine once, binding it to (model, optimizer, train_loader).
        Uses Poisson subsampling via the DataLoader and enforces example-level clipping.
        """
        if self._dp_initialized or not self.enable_dp_sgd:
            return

        # Opacus requires the training DataLoader to compute sample rate;
        # in Ultralytics trainers it's available as self.train_loader after setup.
        if not hasattr(self, "train_loader") or self.train_loader is None:
            # If called too early, skip; will retry next step
            return

        # Attach PrivacyEngine with example-level DP
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=float(self.dp_noise_multiplier),
            max_grad_norm=float(self.dp_max_grad_norm),
        )
        self._dp_initialized = True

        # Helpful log
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(
                f"[DP-SGD] Enabled example-level DP via Opacus "
                f"(noise_multiplier={self.dp_noise_multiplier}, max_grad_norm={self.dp_max_grad_norm})."
            )
        except Exception:
            pass

    def optimizer_step(self):
        """
        Perform one optimizer step with AMP, EMA, and (optionally) example-level DP-SGD.
        - With DP enabled: Opacus intercepts step() to do per-example clipping + Gaussian noise.
        - Without DP: behavior falls back to Ultralytics default.
        """
        # Initialize DP engine once, when dataloader is ready
        self._maybe_init_privacy_engine()

        # Unscale before step for AMP + DP compatibility
        if hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        
        # Opacus will clip per-example grads and add noise inside optimizer.step().

        if hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Zero grads after step
        self.optimizer.zero_grad(set_to_none=True)

        # EMA update if configured
        if getattr(self, "ema", None):
            self.ema.update(self.model)
