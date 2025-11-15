import os
import numpy as np
import torch
import copy
from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from puffle_custom_trainer import PuffleDetectionTrainer  # Use custom trainer implementing PUFFLE approach

# Configuration
num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.001
dataset_name = "FACET"
fed_name = f"Puffle_{dataset_name}"
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

# Function to aggregate models via standard federated averaging
def average_models(models):
    # Retrieve state dictionaries from all models
    model_params = [model.model.state_dict() for model in models]
    common_keys = set.intersection(*(set(m.keys()) for m in model_params))
    averaged_params = {}
    for key in common_keys:
        # Compute simple average across clients for each parameter tensor
        params = torch.stack([model_params[i][key].float() for i in range(len(models))])
        averaged_params[key] = params.mean(dim=0)
    return averaged_params

class CustomYOLO(YOLO):
    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": PuffleDetectionTrainer,  # Using our Puffle trainer
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
    model = CustomYOLO('./yolov8n.yaml').load('yolov8n.pt')
    print("Model loaded successfully.")
    
    for comm_round in range(num_comm):
        print(f"---------------- Communication Round {comm_round} ----------------")
        models = [copy.deepcopy(model) for _ in range(num_client)]
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
                last_model_path = os.path.join(save_dir, "train/weights/last.pt")
                if os.path.exists(last_model_path):
                    valid_models.append(client_model)
                else:
                    print(f"Warning: Checkpoint not found for client {index}. Skipping this client.")
            except Exception as e:
                print(f"Error training client {index}: {str(e)}")
        
        # Aggregate models only if there are valid models
        if valid_models:
            ensemble_model_params = average_models(valid_models)
            model.model.load_state_dict(ensemble_model_params)
            model_save_path = os.path.join(model_dir, f'{fed_name}_comm_{comm_round}.pt')
            model.save(model_save_path)
            print(f"Aggregated model saved for communication round {comm_round}")
        else:
            print(f"Skipping aggregation for round {comm_round} due to lack of valid models.")

    # Save final model
    final_model_path = os.path.join(model_dir, f"{fed_name}_final_model.pt")
    model.save(final_model_path)
    print(f"Final model saved at {final_model_path}")
