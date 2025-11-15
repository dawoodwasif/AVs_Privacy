# PrivFairFL_Post_train.py

import os
import copy
import json
import torch
from statistics import mean
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
fed_name = f"PrivFairPost_{dataset_name}"
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

def average_models_equal(models):
    sd_list = [m.model.state_dict() for m in models]
    common = set.intersection(*(set(sd.keys()) for sd in sd_list))
    out = {}
    for k in common:
        out[k] = torch.stack([sd[k] for sd in sd_list], dim=0).mean(dim=0)
    return out

def aggregate_thresholds(threshold_lists, trim=0.1):
    """
    threshold_lists: list of List[float] from clients, same length G each
    returns List[float]
    """
    if not threshold_lists:
        return []
    G = len(threshold_lists[0])
    agg = []
    for g in range(G):
        vals = sorted([t[g] for t in threshold_lists])
        k = int(trim * len(vals))
        if 2*k >= len(vals):
            agg.append(mean(vals))
        else:
            agg.append(mean(vals[k: len(vals)-k]))
    return agg

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

curr_path = os.getcwd()
configuration_path = os.path.join(curr_path, 'configurations', dataset_name.lower())
config_paths = [os.path.join(configuration_path, f'client{i+1}.yaml') for i in range(num_client)]

os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    model = CustomYOLO('./yolov8n.yaml').load('yolov8n.pt')
    print("Model loaded successfully (PrivFair-Post).")

    # Server state for thresholds
    global_thresholds = None
    thresholds_path = os.path.join(model_dir, "global_thresholds.json")

    for round_id in range(num_comm):
        print(f"===== Communication Round {round_id} =====")
        client_models = [copy.deepcopy(model) for _ in range(num_client)]
        trained_models = []
        client_thresholds = []

        for idx, client_model in enumerate(client_models):
            save_dir = f"runs_privfair_post_client_{idx}"

            try:
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
                    privfair_mode="post",        # <-- here
                    groups=10,
                    target_tpr=0.85,
                )

                # collect thresholds from trainer
                trainer = client_model.trainer
                tau_g = getattr(trainer, "group_thresholds", None)
                if tau_g is not None and len(tau_g) > 0:
                    client_thresholds.append(tau_g)

                last_model_path = os.path.join(save_dir, "train/weights/last.pt")
                if os.path.exists(last_model_path):
                    trained_models.append(client_model)
                else:
                    print(f"[Warn] Client {idx} missing checkpoint, skipping.")
            except Exception as e:
                print(f"[Error] Client {idx} failed: {e}")

        # Aggregate models
        if trained_models:
            avg_params = average_models_equal(trained_models)
            model.model.load_state_dict(avg_params)
            model_save_path = os.path.join(model_dir, f'{fed_name}_comm_{round_id}.pt')
            model.save(model_save_path)
            print(f"Aggregated model saved for round {round_id}")
        else:
            print("Skipping aggregation due to no successful clients.")

        # Aggregate thresholds
        if client_thresholds:
            global_thresholds = aggregate_thresholds(client_thresholds, trim=0.1)
            with open(thresholds_path, "w") as f:
                json.dump({"thresholds": global_thresholds}, f, indent=2)
            print(f"Aggregated thresholds saved to {thresholds_path}")

    final_path = os.path.join(model_dir, f"{fed_name}_final_model.pt")
    model.save(final_path)
    print(f"Final model saved at {final_path}")
    if global_thresholds:
        print(f"Final aggregated thresholds: {global_thresholds}")
