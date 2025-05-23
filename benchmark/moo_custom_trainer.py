
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

class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, metadata_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.epsilon = 0.5  # Privacy parameter for DP
        self.delta = 1e-3  # Privacy parameter for DP
        self.sensitivity = 1.0  # Sensitivity for DP

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


    def _extract_skin_tone(self, img_metadata):
        """Extract the highest skin tone value from the metadata."""
        if not img_metadata or 'persons' not in img_metadata or not img_metadata['persons']:
            LOGGER.info("No persons data in metadata.")
            return None, None

        max_skin_tone = None
        max_value = 0

        for person in img_metadata['persons']:
            skin_tones = {
                key: value for key, value in person.items() if key.startswith("skin_tone_") and isinstance(value, (int, float))
            }

            if skin_tones:
                current_max_key = max(skin_tones, key=skin_tones.get)
                current_max_value = skin_tones[current_max_key]

                if current_max_value > max_value:
                    max_skin_tone = current_max_key
                    max_value = current_max_value

        if max_skin_tone is None:
            LOGGER.info("No skin tone data found in metadata.")
            return None, None

        #LOGGER.info(f"Extracted skin tone: {max_skin_tone} with value: {max_value}")
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

    def _perturb_objective(self, loss):
        epsilon = self.epsilon
        delta = self.delta
        sensitivity = self.sensitivity

        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = torch.normal(0, sigma, size=loss.size()).to(loss.device)
        return loss + noise

    def _fairness_loss(self, embeddings, sensitive_attributes):
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

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100)
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

                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    outputs = self.model(batch)
                    self.loss_items = outputs[1]

                    if isinstance(outputs, tuple):
                        task_loss, _ = outputs
                    else:
                        task_loss = outputs

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


                    w0 = 0.1  # Weight for fairness loss
                    w1 = 1.0  # Weight for privacy loss

                    embeddings = self.feature_space
                    fairness_loss = torch.tensor(0.0, device=self.device)
                    if embeddings is not None and sensitive_attributes is not None:
                        fairness_loss = w0 * self._fairness_loss(embeddings, sensitive_attributes)

                    privacy_and_task_loss = w1 * self._perturb_objective(task_loss)

                    print(f"Task + Privacy loss: {privacy_and_task_loss}, Fairness loss: {fairness_loss}")


                    total_loss = fairness_loss + privacy_and_task_loss
                    self.loss = total_loss.detach().clone()

                self.scaler.scale(total_loss).backward()


                epsilon = self.epsilon
                delta = self.delta
                sensitivity = self.sensitivity
        
                # Compute the standard deviation for the noise
                sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

                # Add noise to each gradient
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # Sample noise with the same shape as the gradient
                            noise = torch.normal(mean=0, std=sigma, size=param.grad.shape, device=param.grad.device)
                            # Add the noise to the gradient
                            param.grad.add_(noise)

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

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
                    #LOGGER.info(f"Batch {i}: Loss items: {loss_items}")
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

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
