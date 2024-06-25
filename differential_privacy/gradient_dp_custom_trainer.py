# custom_trainer.py
from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import random
import gc
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)

import torch
from ultralytics.engine.trainer import BaseTrainer
import random

# Global parameters for differential privacy
epsilon = 500.0  # Privacy budget
delta = 1e-5  # Probability of privacy guarantee not holding
sensitivity = 1.0  # Sensitivity of the gradient

class CustomDetectionTrainer(DetectionTrainer):
    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping, EMA update, and gradient noise for differential privacy."""
        self.scaler.unscale_(self.optimizer)  # Unscale gradients
        
        # Clip gradients to the sensitivity threshold
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=sensitivity)
        
        # Add noise to gradients for differential privacy
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, sigma, size=param.grad.size()).to(param.grad.device)
                param.grad.add_(noise)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)