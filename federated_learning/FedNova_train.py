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

def compute_local_updates(models, global_model):
    """ Compute and return the local updates for each model compared to the global model """
    local_updates = []
    for model in models:
        local_update = {}
        for name, param in model.model.state_dict().items():
            local_update[name] = param.data.clone() - global_model.model.state_dict()[name].data.clone()
        local_updates.append(local_update)
    return local_updates

def normalize_updates(local_updates, num_steps):
    """ Normalize the local updates by the number of local steps """
    normalized_updates = []
    for updates, steps in zip(local_updates, num_steps):
        normalized_update = {name: param / steps for name, param in updates.items()}
        normalized_updates.append(normalized_update)
    return normalized_updates

def aggregate_updates(normalized_updates, num_steps):
    """ Aggregate the normalized updates to form the global update """
    total_steps = sum(num_steps)
    aggregated_update = {}
    for name in normalized_updates[0].keys():
        aggregated_update[name] = sum(normalized_update[name] * steps for normalized_update, steps in zip(normalized_updates, num_steps)) / total_steps
    return aggregated_update

def apply_global_update(global_model, global_update):
    """ Apply the global update to the global model """
    with torch.no_grad():
        for name, param in global_model.model.named_parameters():
            if name in global_update:
                param.data += global_update[name]

min_num_local_step = 1
max_num_local_step = 3

num_comm = 20
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
    global_model = YOLO('./yolov8n.yaml').load('yolov8n.pt')

    for _ in range(num_comm):
        print("-----------------------------------------------------------------------------")
        print(f'num_comm: {_}')
        print("-----------------------------------------------------------------------------")
        models = [copy.deepcopy(global_model) for _ in range(num_client)]
        num_steps = []

        for index, model in enumerate(models):
            num_local_step = random.randint(min_num_local_step, max_num_local_step)  # Simulate varying local steps for heterogeneity
            num_steps.append(num_local_step)
            model.train(data=config_paths[index], epochs=num_local_step, batch=16, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers=0, lr0=lr, lrf=lr)

        # Compute and normalize local updates
        local_updates = compute_local_updates(models, global_model)
        normalized_updates = normalize_updates(local_updates, num_steps)

        # Aggregate normalized updates and apply to global model
        global_update = aggregate_updates(normalized_updates, num_steps)
        apply_global_update(global_model, global_update)

        # Save the updated global model
        model_path = 'FedNova_numlocalstep=' + str(min_num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(global_model, model_path)

    # Save the final model after all rounds
    #global_model.save('FedNova_final_model.pt')
