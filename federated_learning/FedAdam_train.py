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
import torch.optim as optim

def get_grads_(model, server_model, lr):
    grads = []
    server_model_params = server_model.model.state_dict()
    model_params = model.model.state_dict()
    for param_name in server_model_params:
        grads.append((model_params[param_name].clone().detach().flatten() - server_model_params[param_name].clone().detach().flatten()) / lr) 
    return torch.cat(grads)

def set_grads_(model, server_model, new_grads, lr):
    start = 0
    server_model_params = server_model.model.state_dict()
    model_params = model.model.state_dict()
    for param_name in server_model_params:
        dims = model_params[param_name].shape
        end = start + dims.numel()
        model_params[param_name].copy_(server_model_params[param_name].clone().detach() + new_grads[start:end].reshape(dims).clone() * lr)  
        start = end
    model.model.load_state_dict(model_params)
    return model

def average_models(models):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float16)
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params

def fed_adam_update(models, server_model, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = {param_name: torch.zeros_like(param) for param_name, param in server_model.model.named_parameters()}
    v = {param_name: torch.zeros_like(param) for param_name, param in server_model.model.named_parameters()}
    t = 0

    server_params = server_model.model.state_dict()
    for client_model in models:
        t += 1
        client_params = client_model.model.state_dict()
        for param_name in server_params:
            if param_name in m and param_name in v:
                g_t = (client_params[param_name] - server_params[param_name]) / lr
                m[param_name] = beta1 * m[param_name] + (1 - beta1) * g_t
                v[param_name] = beta2 * v[param_name] + (1 - beta2) * g_t ** 2
                m_hat = m[param_name] / (1 - beta1 ** t)
                v_hat = v[param_name] / (1 - beta2 ** t)
                server_params[param_name] += lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    
    server_model.model.load_state_dict(server_params)
    return server_model

# Training Parameters
num_local_step = 1
num_comm = 20
num_client = 4
lr = 0.0000001
curr_path = os.getcwd()
config_paths = [os.path.join(curr_path, f'client{i+1}.yaml') for i in range(num_client)]

os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    model = YOLO('./yolov8n.yaml').load('yolov8n.pt')

    for comm_round in range(num_comm):
        print(f"Comm Round: {comm_round}")
        models = [copy.deepcopy(model) for _ in range(num_client)]
        for index, client_model in enumerate(models):
            client_model.train(data=config_paths[index], epochs=num_local_step, batch=16, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers=0, lr0=lr, lrf=lr)
        
        # Use FedAdam update method
        model = fed_adam_update(models, model, lr)
        
        model_save_name = f'FedAdam_numlocalstep={num_local_step}_numclient={num_client}_numcomm={comm_round}.pt'
        torch.save(model, model_save_name)

    torch.save(model, 'FedAdam_final_model.pt')
