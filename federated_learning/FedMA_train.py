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

fed_name = "FedMA"

# Create the directory if it doesn't exist
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

def average_models_layerwise(models):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}

    # Match and average layer by layer
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float16)
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params

# Training Parameters
num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.0000001
curr_path = os.getcwd()
config_path = os.path.join(curr_path, 'config.yaml')

config_paths = []
config_path1 = os.path.join(curr_path, 'client1.yaml')
config_paths.append(config_path1)

config_path2 = os.path.join(curr_path, 'client2.yaml')
config_paths.append(config_path2)

config_path3 = os.path.join(curr_path, 'client3.yaml')
config_paths.append(config_path3)

config_path4 = os.path.join(curr_path, 'client4.yaml')
config_paths.append(config_path4)

import os
os.environ['WANDB_DISABLED'] = 'true'


if __name__ == '__main__':
    model = YOLO('./yolov8n.yaml').load('yolov8n.pt')

    for _ in range(num_comm):
        print('num_comm: ', _)
        models = [copy.deepcopy(model) for _ in range(num_client)]
        for index, client_model in enumerate(models):
            client_model.train(data=config_paths[index], epochs=num_local_step, batch=32, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers=0, lr0=lr, lrf=lr)
        
        ensemble_model_params = average_models_layerwise(models)
        model.model.load_state_dict(ensemble_model_params)
        
        name = fed_name + '_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(model, os.path.join(model_dir,name))

    torch.save(model, os.path.join(model_dir,f'{fed_name}_final_model.pt'))
