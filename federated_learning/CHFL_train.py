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

fed_name = "CHFL"

# Create the directory if it doesn't exist
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

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

def cluster_clients(models, num_clusters=2):
    # A simple clustering method, e.g., based on the initial model parameters similarity
    clusters = [[] for _ in range(num_clusters)]
    for i, model in enumerate(models):
        clusters[i % num_clusters].append(model)  # Simple round-robin clustering for demonstration
    return clusters

def train_and_aggregate_clusters(clusters, lr):
    cluster_models = []
    for cluster in clusters:
        # Perform local training within each cluster
        for client_model in cluster:
            client_model.train(data=config_paths[clusters.index(cluster)], epochs=num_local_step, batch=16, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers=0, lr0=lr, lrf=lr)
        # Aggregate models within the cluster
        cluster_model_params = average_models(cluster)
        cluster_model = YOLO('./yolov8n.yaml')
        cluster_model.model.load_state_dict(cluster_model_params)
        cluster_models.append(cluster_model)
    return cluster_models

# Training Parameters
num_local_step = 1
num_comm = 100
num_client = 4
num_clusters = 4  # Define number of clusters
lr = 0.0000001
curr_path = os.getcwd()
config_paths = [os.path.join(curr_path, f'client{i+1}.yaml') for i in range(num_client)]

os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    model = YOLO('./yolov8n.yaml').load('yolov8n.pt')

    for _ in range(num_comm):
        print('num_comm: ', _)
        models = [copy.deepcopy(model) for _ in range(num_client)]
        
        clusters = cluster_clients(models, num_clusters=num_clusters)
        cluster_models = train_and_aggregate_clusters(clusters, lr)
        
        # Global aggregation of cluster-specific models
        ensemble_model_params = average_models(cluster_models)
        model.model.load_state_dict(ensemble_model_params)
        
        name = fed_name + '_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(model, os.path.join(model_dir,name))

    torch.save(model, os.path.join(model_dir,f'{fed_name}_final_model.pt'))
