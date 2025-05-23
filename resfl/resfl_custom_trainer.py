from ultralytics.models.yolo.detect import DetectionTrainer
import torch
import json
import gc
import time
import warnings
from torch import distributed as dist
import numpy as np
from ultralytics.utils import LOGGER, RANK, TQDM
from ultralytics.utils.torch_utils import autocast

###############################
# Gradient Reversal Layer (GRL)
###############################
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

#####################################
# Privacy Adversary Network Module  #
#####################################
class PrivacyAdversary(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PrivacyAdversary, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        if x.ndimension() > 2:
            x = x.mean(dim=[2, 3])
        return self.net(x)

#############################################
# Custom Detection Trainer with APD (No Fairness Loss)
#############################################
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, metadata_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        metadata_path = "/home/Virginia_Research/FACET/metadata.json"
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            LOGGER.info(f"Loaded metadata from {metadata_path}")
        else:
            self.metadata = {}
            LOGGER.info("No metadata path provided, initialized with empty metadata.")

        self.feature_space = None
        self.logits = None
        self.fairness_metric = 0.0  # This attribute will store a fairness metric for server-side UFM

        # Initialize APD components; lazy initialization of privacy adversary
        self.privacy_adversary = None
        self.adv_optimizer = None
        self.lambda_adv = 0.0001  # Weight for adversarial gradient reversal

        # Loss weighting hyperparameters
        self.weight_task_loss = 1.0
        self.weight_uncertainty_loss = 0.0001
        self.weight_adversarial_loss = 0.00001

    def _extract_skin_tone(self, img_metadata):
        if not img_metadata or 'persons' not in img_metadata or not img_metadata['persons']:
            LOGGER.info("No persons data in metadata.")
            return None, None
        max_skin_tone = None
        max_value = 0
        for person in img_metadata['persons']:
            skin_tones = {key: value for key, value in person.items() if key.startswith("skin_tone_") and isinstance(value, (int, float))}
            if skin_tones:
                current_max_key = max(skin_tones, key=skin_tones.get)
                current_max_value = skin_tones[current_max_key]
                if current_max_value > max_value:
                    max_skin_tone = current_max_key
                    max_value = current_max_value
        return max_skin_tone, max_value

    def register_feature_space_hook(self, layer_index):
        def hook_fn(module, input, output):
            self.feature_space = output
        hook = self.model.model[layer_index].register_forward_hook(hook_fn)
        LOGGER.info("Feature space hook successfully registered!")
        return hook

    def register_logits_hook(self):
        def hook_fn(module, input, output):
            self.logits = output
        hook = self.model.model[-2].register_forward_hook(hook_fn)
        LOGGER.info("Logits hook successfully registered!")
        return hook

    def _fairness_loss(self, embeddings, sensitive_attributes):
        # Compute mean pairwise L2 distance between cluster centers
        if embeddings.ndimension() == 4:
            embeddings = embeddings.mean(dim=[2, 3])
        cluster_means = []
        unique_attributes = torch.unique(sensitive_attributes)
        for attr in unique_attributes:
            mask = sensitive_attributes == attr
            cluster_embeddings = embeddings[mask]
            if cluster_embeddings.numel() > 0:
                cluster_means.append(cluster_embeddings.mean(dim=0))
        if len(cluster_means) > 1:
            cluster_means = torch.stack(cluster_means)
            pairwise_distances = torch.cdist(cluster_means, cluster_means, p=2)
            fairness_loss = pairwise_distances.mean()
        else:
            fairness_loss = torch.tensor(0.0, device=embeddings.device)
        return fairness_loss

    def _epistemic_uncertainty_loss(self, logits, sensitive_attributes):
        # Penalize high variance in logits across sensitive groups
        uncertainty_per_group = []
        unique_attributes = torch.unique(sensitive_attributes)
        for attr in unique_attributes:
            mask = sensitive_attributes == attr
            group_logits = logits[mask]
            if group_logits.shape[0] < 2:
                continue
            group_variance = torch.var(group_logits, dim=0)
            uncertainty_per_group.append(group_variance.mean())
        if not uncertainty_per_group:
            return torch.tensor(1e-6, device=logits.device)
        return torch.mean(torch.stack(uncertainty_per_group))


    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        nb = len(self.train_loader)
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        self.register_feature_space_hook(layer_index=-4)
        self.register_logits_hook()
        LOGGER.info(f"Starting training for {self.epochs} epochs...")
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
                    self.loss_items = outputs[1]
                    if isinstance(outputs, tuple):
                        task_loss, _ = outputs
                    else:
                        task_loss = outputs

                    # Prepare sensitive attributes from metadata
                    sensitive_attributes = torch.tensor([-1], device=self.device)
                    if self.metadata:
                        image_files = batch.get("im_file")
                        image_ids = [img.split('/')[-1].split('.')[0].replace('sa_', '') for img in image_files]
                        sensitive_attributes = []
                        for img_id in image_ids:
                            if str(img_id) in self.metadata:
                                skin_tone_key, skin_tone_value = self._extract_skin_tone(self.metadata[str(img_id)])
                                sensitive_attributes.append(skin_tone_value if skin_tone_value is not None else -1)
                            else:
                                sensitive_attributes.append(-1)
                        sensitive_attributes = torch.tensor(sensitive_attributes, device=self.device, dtype=torch.float32)
                    
                    # Compute fairness metric for later UFM calculation (but do not add to loss)
                    embeddings = self.feature_space
                    fairness_loss = torch.tensor(0.0, device=self.device)
                    if embeddings is not None and sensitive_attributes is not None:
                        fairness_loss = self._fairness_loss(embeddings, sensitive_attributes)
                    self.fairness_metric = fairness_loss.item()  # Store fairness metric for server use
                    
                    # Compute epistemic uncertainty loss from logits
                    logits = self.logits
                    epistemic_uncertainty_loss = torch.tensor(0.0, device=self.device)
                    if logits is not None and sensitive_attributes is not None:
                        epistemic_uncertainty_loss = self._epistemic_uncertainty_loss(logits, sensitive_attributes)

                    ##########################################
                    # Adversarial Privacy Disentanglement (APD)
                    ##########################################
                    adv_loss_main = torch.tensor(0.0, device=self.device)
                    if embeddings is not None and sensitive_attributes is not None:
                        if self.privacy_adversary is None:
                            emb_for_init = embeddings.mean(dim=[2, 3]) if embeddings.ndimension() == 4 else embeddings
                            input_dim = emb_for_init.shape[1]
                            self.privacy_adversary = PrivacyAdversary(input_dim).to(self.device).float()
                            self.adv_optimizer = torch.optim.Adam(self.privacy_adversary.parameters(), lr=1e-4)
                            LOGGER.info(f"Initialized Privacy Adversary with input_dim {input_dim}")
                        valid_mask = sensitive_attributes != -1
                        valid_sensitive = sensitive_attributes[valid_mask] if valid_mask.sum() > 0 else sensitive_attributes
                        with torch.cuda.amp.autocast(enabled=False):
                            self.adv_optimizer.zero_grad()
                            emb_detached = embeddings.detach()
                            emb_detached = emb_detached.mean(dim=[2, 3]) if emb_detached.ndimension() == 4 else emb_detached
                            emb_detached = emb_detached.float()
                            adv_pred = self.privacy_adversary(emb_detached)
                            target = sensitive_attributes.unsqueeze(1).float()
                            adv_loss_for_adv = torch.nn.functional.mse_loss(adv_pred, target)
                            self.scaler.scale(adv_loss_for_adv).backward()
                            torch.nn.utils.clip_grad_norm_(self.privacy_adversary.parameters(), max_norm=1.0)
                            self.adv_optimizer.step()
                        with torch.cuda.amp.autocast(enabled=False):
                            rev_emb = grad_reverse(embeddings, self.lambda_adv)
                            rev_emb = rev_emb.mean(dim=[2, 3]) if rev_emb.ndimension() == 4 else rev_emb
                            rev_emb = rev_emb.float()
                            for param in self.privacy_adversary.parameters():
                                param.requires_grad = False
                            adv_pred_main = self.privacy_adversary(rev_emb)
                            adv_loss_main = torch.nn.functional.mse_loss(adv_pred_main, target)
                            for param in self.privacy_adversary.parameters():
                                param.requires_grad = True

                    # Combine losses: task, uncertainty, adversarial (fairness loss not included in total loss)
                    task_loss = task_loss * self.weight_task_loss
                    epistemic_uncertainty_loss = epistemic_uncertainty_loss * self.weight_uncertainty_loss
                    adv_loss_main = adv_loss_main * self.weight_adversarial_loss
                    total_loss = task_loss + epistemic_uncertainty_loss + adv_loss_main

                    self.loss = total_loss

                self.scaler.scale(self.loss).backward()

                if i - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = i

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_items = self.tloss.flatten().tolist()
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + len(loss_items)))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *loss_items,
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and i in self.plot_idx:
                        self.plot_training_samples(batch, i)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            LOGGER.info(f"Epoch {epoch} completed. LR: {self.lr}")
            epoch += 1

            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
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
        gc.collect()
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")
