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

def evaluate_map50(trainedmodel, data_path, dataset='test'):
    metrics = trainedmodel.val(data=data_path, split=dataset, workers = 0)
    map50 = round(metrics.box.map50, 3)
    # print("The mAP of model on {0} dataset is {1}".format(dataset,map50))
    return metrics, map50

curr_path=os.getcwd()
config_path = os.path.join(curr_path, 'config.yaml')


if __name__ == '__main__':
    for num_round in [10, 20, 50, 100]:
        FedGH_model_name = 'model_weights/FedGH/FedGH_numlocalstep=1_numclient=4_numcomm=' + str(num_round-1) + '.pt'
        # FedAvg_model_name = 'model_weights/FedAvg/FedAvg_numlocalstep=1_numclient=4_numcomm=' + str(num_round-1) + '.pt'
        # FedMA_model_name = 'model_weights/FedMA/FedMA_numlocalstep=1_numclient=4_numcomm=' + str(num_round-1) + '.pt'
        # FedProx_model_name = 'model_weights/FedProx/FedProx_numlocalstep=1_numclient=4_numcomm=' + str(num_round-1) + '.pt'
        # CHFL_model_name = 'model_weights/CHFL/CHFL_numlocalstep=1_numclient=4_numcomm=' + str(num_round-1) + '.pt'

        FedGH_model = torch.load(FedGH_model_name)
        # FedAvg_model = torch.load(FedAvg_model_name)
        # FedMA_model = torch.load(FedMA_model_name)
        # FedProx_model = torch.load(FedProx_model_name)
        # CHFL_model = torch.load(CHFL_model_name)

        FedGH_metrics, FedGH_map50 = evaluate_map50(FedGH_model, config_path, dataset='test')
        # FedAvg_metrics, FedAvg_map50 = evaluate_map50(FedAvg_model, config_path, dataset='test')
        # FedMA_metrics, FedMA_map50 = evaluate_map50(FedMA_model, config_path, dataset='test')
        # FedProx_metrics, FedProx_map50 = evaluate_map50(FedProx_model, config_path, dataset='test')
        # CHFL_metrics, CHFL_map50 = evaluate_map50(CHFL_model, config_path, dataset='test')
        
        
        print('Round: ', num_round, ' FedGH: ', FedGH_map50)
        # print('Round: ', num_round, ' FedAvg: ', FedAvg_map50)
        # print('Round: ', num_round, ' FedMA: ', FedMA_map50)
        # print('Round: ', num_round, ' FedProx: ', FedProx_map50)
        # print('Round: ', num_round, ' CHFL: ', CHFL_map50)
