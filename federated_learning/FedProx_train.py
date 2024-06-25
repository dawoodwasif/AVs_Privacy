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

fed_name = "FedProx"

# Create the directory if it doesn't exist
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)


def average_models(models, mu=0.01):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name].float() for i in range(len(models))])
        mean_param = torch.mean(params, dim=0)

        # Apply FedProx regularization
        if mean_param.dtype == torch.float32 or mean_param.dtype == torch.float16:
            for i in range(len(models)):
                mean_param += mu * (model_params[i][param_name].float() - mean_param)

        averaged_params[param_name] = mean_param
    return averaged_params


num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.0000001
mu = 0.01

curr_path=os.getcwd()
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
        print("-----------------------------------------------------------------------------")
        print('num_comm: ', _)
        print("-----------------------------------------------------------------------------")
        models = [copy.deepcopy(model) for _ in range(num_client)]
        for index, dup_model in enumerate(models):
            dup_model.train(data=config_paths[index], epochs=num_local_step, batch=32, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers = 0, lr0 = lr, lrf = lr)
        ensemble_model_params = average_models(models, mu)
        model.model.load_state_dict(ensemble_model_params)
        
        name = fed_name + '_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(model, os.path.join(model_dir,name))

    torch.save(model, os.path.join(model_dir,f'{fed_name}_final_model.pt'))