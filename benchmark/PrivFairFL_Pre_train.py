# PrivFairFL_Pre_train.py

import os
import copy
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO as YOLOModel
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from privfair_custom_trainer import PrivFairDetectionTrainer

# Config
num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.001
dataset_name = "FACET"
fed_name = f"PrivFairPre_{dataset_name}"
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

# Equal-weight FedAvg aggregation
def average_models_equal(models):
    sd_list = [m.model.state_dict() for m in models]
    common = set.intersection(*(set(sd.keys()) for sd in sd_list))
    out = {}
    for k in common:
        out[k] = torch.stack([sd[k] for sd in sd_list], dim=0).mean(dim=0)
    return out

class CustomYOLO(YOLO):
    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": PrivFairDetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

# Paths for client configs
curr_path = os.getcwd()
configuration_path = os.path.join(curr_path, 'configurations', dataset_name.lower())
config_paths = [os.path.join(configuration_path, f'client{i+1}.yaml') for i in range(num_client)]

# Disable WANDB
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    model = CustomYOLO('./yolov8n.yaml').load('yolov8n.pt')
    print("Model loaded successfully (PrivFair-Pre).")

    for round_id in range(num_comm):
        print(f"===== Communication Round {round_id} =====")
        client_models = [copy.deepcopy(model) for _ in range(num_client)]
        trained_models = []

        for idx, client_model in enumerate(client_models):
            save_dir = f"runs_privfair_pre_client_{idx}"
            try:
                # Important: pass mode=pre via overrides so trainer picks it up
                client_model.train(
                    data=config_paths[idx],
                    epochs=num_local_step,
                    batch=32,
                    save=True,
                    resume=False,
                    project=save_dir,
                    name="train",
                    iou=0.5,
                    conf=0.001,
                    plots=False,
                    workers=0,
                    lr0=lr,
                    lrf=lr,
                    privfair_mode="pre",         # <-- here
                    groups=10,
                    target_tpr=0.85,
                )
                # checkpoint exists?
                last_model_path = os.path.join(save_dir, "train/weights/last.pt")
                if os.path.exists(last_model_path):
                    trained_models.append(client_model)
                else:
                    print(f"[Warn] Client {idx} missing checkpoint, skipping.")
            except Exception as e:
                print(f"[Error] Client {idx} failed: {e}")

        if trained_models:
            avg_params = average_models_equal(trained_models)
            model.model.load_state_dict(avg_params)
            model_save_path = os.path.join(model_dir, f'{fed_name}_comm_{round_id}.pt')
            model.save(model_save_path)
            print(f"Aggregated model saved for round {round_id}")
        else:
            print("Skipping aggregation due to no successful clients.")

    final_path = os.path.join(model_dir, f"{fed_name}_final_model.pt")
    model.save(final_path)
    print(f"Final model saved at {final_path}")
