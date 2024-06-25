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

def get_layerwise_gradients(models, global_model):
    """ Compute and return gradients layer-wise for each model compared to the global model """
    layer_grads = {}
    for name, param in global_model.model.state_dict().items():
        if param.requires_grad:
            layer_grads[name] = []

    for model in models:
        local_params = model.model.state_dict()
        for name, param in local_params.items():
            if param.requires_grad:
                grad = param.data - global_model.model.state_dict()[name].data
                layer_grads[name].append(grad)
    
    # Average the gradients for each parameter, handle empty cases
    avg_grads = {}
    for name, grads in layer_grads.items():
        if grads:
            stacked_grads = torch.stack(grads)
            avg_grads[name] = torch.mean(stacked_grads, dim=0)
        else:
            print(f"No gradients collected for parameter {name}, this should be checked.")

    return avg_grads

def apply_gradients(global_model, gradients):
    """ Apply averaged gradients layer-wise to the global model """
    with torch.no_grad():
        for name, param in global_model.model.named_parameters():
            if param.requires_grad and name in gradients:
                param.data -= gradients[name]  # Update with negative gradient to minimize loss

num_local_step = 1
num_comm = 20
num_client = 4
lr = 0.0000001
curr_path = os.getcwd()

config_paths = [os.path.join(curr_path, f'client{i}.yaml') for i in range(1, 5)]
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    global_model = YOLO('./yolov8n.yaml').load('yolov8n.pt')

    for _ in range(num_comm):
        print("-----------------------------------------------------------------------------")
        print('num_comm: ', _)
        print("-----------------------------------------------------------------------------")
        models = [copy.deepcopy(global_model) for _ in range(num_client)]
        
        for index, model in enumerate(models):
            model.train(data=config_paths[index], epochs=num_local_step, batch=16, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers=0, lr0=lr, lrf=lr)
        
        # Gather and average gradients layer-wise
        gradients = get_layerwise_gradients(models, global_model)
        apply_gradients(global_model, gradients)

        #model = global_model.model.state_dict()
        
        # Save the updated global model
        model_path = 'LG-FedAvg_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(global_model, model_path)

    # Save the final model after all rounds
    torch.save(global_model, 'LG-FedAvg_final_model.pt')
