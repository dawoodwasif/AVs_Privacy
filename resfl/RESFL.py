import os
import numpy as np
import torch
import copy
from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from resfl_custom_trainer_custom_trainer import CustomDetectionTrainer  # APD + UFM-capable trainer

# =========================
# Configuration
# =========================
num_local_step = 1
num_comm = 100
num_client = 4
lr = 0.001
dataset_name = "FACET"
fed_name = f"RESFL_{dataset_name}"
model_dir = os.path.join('model_weights', fed_name)
os.makedirs(model_dir, exist_ok=True)

# ===== Confidence gate parameters =====
BETA = 2.0                 # temperature for exp(-beta * UFM)
UFM_CLIP = (0.0, 5.0)      # clip range for robustness
MAP_FLOOR = 0.30           # confidence/utility floor (val mAP@50-95) to gate
# ======================================

# =========================
# Utilities
# =========================
def extract_val_map(trainer) -> float:
    """
    Try to robustly read a validation mAP from Ultralytics metrics across versions.
    Falls back to the mean of any keys containing 'mAP' if needed.
    """
    m = getattr(trainer, "metrics", None)
    if isinstance(m, dict):
        # Priority-ordered keys commonly present across versions
        for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "val/mAP50-95", "metrics/mAP50"):
            if k in m and isinstance(m[k], (int, float)):
                return float(m[k])
        # Fallback: average any mAP-like entries
        cands = [float(v) for k, v in m.items() if ("mAP" in k) and isinstance(v, (int, float))]
        if len(cands) > 0:
            return float(np.mean(cands))
    return float("nan")


def average_models(models, ufm_metrics, util_ok_flags, beta=BETA, clip=UFM_CLIP):
    """
    Gated, temperature-scaled weighting:
      - Clip UFM for robustness.
      - Compute w_i = exp(-beta * UFM_i).
      - Gate by util_ok_flags (e.g., mAP floor).
      - Normalize and weighted-average parameters.
    """
    u = torch.tensor(ufm_metrics, dtype=torch.float32)
    u = torch.clamp(u, clip[0], clip[1])
    w = torch.exp(-beta * u)                       # lower UFM => larger weight

    mask = torch.tensor(util_ok_flags, dtype=torch.bool)
    w = w * mask.float()                           # zero-out clients failing utility floor

    if float(w.sum().item()) == 0.0:
        # Fallback: uniform over all models if everyone is gated out
        w = torch.ones_like(w) / len(models)
    else:
        w = w / w.sum()

    model_params = [m.model.state_dict() for m in models]
    common_keys = set.intersection(*(set(s.keys()) for s in model_params))
    averaged = {}
    for k in common_keys:
        stacked = torch.stack([model_params[i][k].float() * w[i] for i in range(len(models))])
        averaged[k] = stacked.sum(dim=0)
    return averaged


class CustomYOLO(YOLO):
    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": CustomDetectionTrainer,  # APD + UFM trainer
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }


# =========================
# Paths / logging
# =========================
curr_path = os.getcwd()
configuration_path = os.path.join(curr_path, 'configurations', dataset_name.lower())
config_paths = [os.path.join(configuration_path, f'client{i+1}.yaml') for i in range(num_client)]

# Disable WANDB logging
os.environ['WANDB_DISABLED'] = 'true'


# =========================
# Main
# =========================
if __name__ == '__main__':
    # Load the initial global model
    model = CustomYOLO('./yolov8n.yaml').load('yolov8n.pt')
    print("Global model loaded successfully.")

    for comm_round in range(num_comm):
        print(f"---------------- Communication Round {comm_round} ----------------")

        # Create deep copies of the global model for each client
        models = [copy.deepcopy(model) for _ in range(num_client)]
        ufm_metrics = []
        util_ok_flags = []
        valid_models = []

        # Train each client model independently
        for index, client_model in enumerate(models):
            save_dir = f"runs_detect_client_{index}"
            try:
                client_model.train(
                    data=config_paths[index],
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
                    lrf=lr
                )

                # Retrieve UFM (stored in trainer.fairness_metric)
                ufm_value = float(getattr(client_model.trainer, "fairness_metric", float("inf")))
                # Optional: clip at source for stability
                ufm_value = float(np.clip(ufm_value, UFM_CLIP[0], UFM_CLIP[1]))
                print(f"Client {index} UFM (clipped): {ufm_value:.4f}")

                # Retrieve a validation mAP proxy for utility/confidence gating
                val_map = extract_val_map(client_model.trainer)
                util_ok = (not np.isnan(val_map)) and (val_map >= MAP_FLOOR)
                print(f"Client {index} val mAP: {val_map:.4f} | passes gate: {util_ok}")

                last_model_path = os.path.join(save_dir, "train/weights/last.pt")
                if os.path.exists(last_model_path):
                    ufm_metrics.append(ufm_value)
                    util_ok_flags.append(util_ok)
                    valid_models.append(client_model)
                else:
                    print(f"Warning: Checkpoint not found for client {index}. Skipping this client.")

            except Exception as e:
                print(f"Error training client {index}: {str(e)}")

        # Aggregate models using gated, temperature-scaled UFM weights
        if len(valid_models) > 0 and len(ufm_metrics) == len(valid_models):
            aggregated_params = average_models(valid_models, ufm_metrics, util_ok_flags, beta=BETA, clip=UFM_CLIP)
            model.model.load_state_dict(aggregated_params)
            model_save_path = os.path.join(model_dir, f'{fed_name}_comm_{comm_round}.pt')
            model.save(model_save_path)
            print(f"Aggregated global model saved for communication round {comm_round}")
        else:
            print("Skipping aggregation due to insufficient valid client models or metrics.")

    # Save the final global model
    final_model_path = os.path.join(model_dir, f"{fed_name}_final_model.pt")
    model.save(final_model_path)
    print(f"Final global model saved at {final_model_path}")
