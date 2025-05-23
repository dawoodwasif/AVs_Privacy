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

class FairFedDetectionTrainer(DetectionTrainer):
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
        self.fairness_metric = 1.0  # Default fairness metric

    def _extract_skin_tone(self, img_metadata):
        if not img_metadata or 'persons' not in img_metadata or not img_metadata['persons']:
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

        return max_skin_tone, max_value

    def register_feature_space_hook(self, layer_index):
        def hook_fn(module, input, output):
            self.feature_space = output
        
        hook = self.model.model[layer_index].register_forward_hook(hook_fn)
        LOGGER.info("Feature space hook successfully registered!")
        return hook

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

        self.fairness_metric = fairness_loss.item()  # Store fairness metric
        # print(f"Fairness loss: {fairness_loss}")
        return fairness_loss

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
                    
                    w0 = 0.1  # Fairness loss weight
                    fairness_loss = self._fairness_loss(self.feature_space, sensitive_attributes)
                    
                    total_loss = task_loss + w0 * fairness_loss
                    self.loss = total_loss.detach().clone()
                
                self.scaler.scale(total_loss).backward()
                
                if i - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = i
                
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
