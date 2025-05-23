import os
import numpy as np
import torch
import copy
from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from resfl_custom_trainer_custom_trainer import CustomDetectionTrainer  # Our custom trainer implementing APD and UFM

# Configuration
num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.001
dataset_name = "FACET"
fed_name = f"RESFL_{dataset_name}"
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

# Function to aggregate models with UFM-based weighting
def average_models(models, ufm_metrics):
    # Retrieve state dictionaries from all models
    model_params = [model.model.state_dict() for model in models]
    common_keys = set.intersection(*(set(m.keys()) for m in model_params))
    averaged_params = {}
    
    # Normalize UFM-based weights: lower UFM should get higher weight.
    # Here we convert UFM values into weights by using: weight_i = 1 / (1 + UFM_i)
    ufm_tensor = torch.tensor(ufm_metrics, dtype=torch.float32)
    weights = 1.0 / (1.0 + ufm_tensor)
    if weights.sum() > 0:
        normalized_weights = weights / weights.sum()
    else:
        normalized_weights = torch.ones_like(weights) / len(weights)
    
    for param_name in common_keys:
        stacked_params = torch.stack([model_params[i][param_name].float() * normalized_weights[i] for i in range(len(models))])
        averaged_params[param_name] = stacked_params.sum(dim=0)
    
    return averaged_params

class CustomYOLO(YOLO):
    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": CustomDetectionTrainer,  # Using our custom trainer with APD and UFM computation
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

# Paths for client configurations
curr_path = os.getcwd()
configuration_path = os.path.join(curr_path, 'configurations', dataset_name.lower())
config_paths = [os.path.join(configuration_path, f'client{i+1}.yaml') for i in range(num_client)]

# Disable WANDB logging
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    # Load the initial global model
    model = CustomYOLO('./yolov8n.yaml').load('yolov8n.pt')
    print("Global model loaded successfully.")

    for comm_round in range(num_comm):
        print(f"---------------- Communication Round {comm_round} ----------------")
        # Create deep copies of the global model for each client
        models = [copy.deepcopy(model) for _ in range(num_client)]
        ufm_metrics = []
        valid_models = []

        # Train each client model independently
        for index, client_model in enumerate(models):
            trainer = client_model.trainer
            save_dir = f"runs_detect_client_{index}"
            try:
                client_model.train(
                    data=config_paths[index], epochs=num_local_step, batch=32, save=True,
                    resume=False, project=save_dir, name="train", iou=0.5, conf=0.001,
                    plots=False, workers=0, lr0=lr, lrf=lr
                )
                
                # Retrieve the UFM value computed on the client (stored in fairness_metric)
                ufm_value = client_model.trainer.fairness_metric
                print(f"Client {index} UFM: {ufm_value}")
                
                last_model_path = os.path.join(save_dir, "train/weights/last.pt")
                if os.path.exists(last_model_path):
                    ufm_metrics.append(ufm_value)
                    valid_models.append(client_model)
                else:
                    print(f"Warning: Checkpoint not found for client {index}. Skipping this client.")
            except Exception as e:
                print(f"Error training client {index}: {str(e)}")

        # Aggregate models using UFM-based weights if there are valid models
        if valid_models and ufm_metrics:
            aggregated_params = average_models(valid_models, ufm_metrics)
            model.model.load_state_dict(aggregated_params)
            model_save_path = os.path.join(model_dir, f'{fed_name}_comm_{comm_round}.pt')
            model.save(model_save_path)
            print(f"Aggregated global model saved for communication round {comm_round}")
        else:
            print(f"Skipping aggregation for round {comm_round} due to insufficient valid client models.")

    # Save the final global model
    final_model_path = os.path.join(model_dir, f"{fed_name}_final_model.pt")
    model.save(final_model_path)
    print(f"Final global model saved at {final_model_path}")
