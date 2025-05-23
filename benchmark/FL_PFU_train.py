# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import shutil
import cv2
import random
import matplotlib.pyplot as plt
import copy
import wandb

from ultralytics import YOLO

import torch


dataset_name = "FACET"
fed_name = f"PFU_{dataset_name}"

# Create the directory if it doesn't exist
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

def average_models(models):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float16)
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params



############################################################################################
from ultralytics.models.yolo.model import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from moo_custom_trainer import CustomDetectionTrainer



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
                "model": DetectionModel,
                "trainer":  CustomDetectionTrainer, # yolo.detect.DetectionTrainer,  # Use the custom trainer
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
    
##################################################################################################################
model = CustomYOLO("yolov8n.pt")
##################################################################################################################


num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.001

curr_path=os.getcwd()
configuration_path = os.path.join(curr_path, 'configurations')

curr_config_path = os.path.join(configuration_path, dataset_name.lower())

config_paths = []
config_path1 = os.path.join(curr_config_path, 'client1.yaml')
config_paths.append(config_path1)

config_path2 = os.path.join(curr_config_path, 'client2.yaml')
config_paths.append(config_path2)

config_path3 = os.path.join(curr_config_path, 'client3.yaml')
config_paths.append(config_path3)

config_path4 = os.path.join(curr_config_path, 'client4.yaml')
config_paths.append(config_path4)

import os
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    model = CustomYOLO('./yolov8n.yaml').load('yolov8n.pt')

    for _ in range(num_comm):
        print("-----------------------------------------------------------------------------")
        print('num_comm: ', _)
        print("-----------------------------------------------------------------------------")
        models = [copy.deepcopy(model) for _ in range(num_client)]
        for index, dup_model in enumerate(models):
            dup_model.train(data=config_paths[index], epochs=num_local_step, batch=32, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers = 0, lr0 = lr, lrf = lr)
        ensemble_model_params = average_models(models)
        model.model.load_state_dict(ensemble_model_params)
        name = fed_name + '_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(model, os.path.join(model_dir,name))

    torch.save(model, os.path.join(model_dir,f'{fed_name}_final_model.pt'))
