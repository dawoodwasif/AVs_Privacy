import torch
import random
from ultralytics import YOLO
import os

# Disable Weights and Biases logging
os.environ['WANDB_DISABLED'] = 'true'

# custom_yolo.py
from ultralytics.models.yolo.model import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel




dp_method = "output"

output_dp = 'False'

if dp_method == "gradient":
    from gradient_dp_custom_trainer import CustomDetectionTrainer
elif dp_method == "objective":
    from objective_dp_custom_trainer import CustomDetectionTrainer
elif dp_method == "input":
    from input_dp_custom_trainer import CustomDetectionTrainer
elif dp_method == "output":
    from output_dp_custom_trainer import CustomDetectionModel
    output_dp = 'True'


class CustomYOLO(YOLO):
    @property
    def task_map(self):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": CustomDetectionModel if output_dp else DetectionModel,
                "trainer": yolo.detect.DetectionTrainer if output_dp else CustomDetectionTrainer, # yolo.detect.DetectionTrainer,  # Use the custom trainer
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


# Define hyperparameters
epochs = 100
learning_rate = 1e-6  
img_size = 640
batch_size = 16
iou_threshold = 0.5
confidence_threshold = 0.001
save_period = 1

# Custom path for saving the model
experiment_name = f'model_weights_baseline_training_{dp_method}_perturbation'

# Load a pretrained YOLOv8 model
model = CustomYOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data='data.yaml',  # Path to the dataset configuration file
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    save=True,
    resume=False,
    project = experiment_name,
    iou=iou_threshold,
    conf=confidence_threshold,
    save_period=save_period,
    lr0=learning_rate,
    lrf=learning_rate    
)

# Validate the model
val_results = model.val(data="data.yaml",  split="test")

