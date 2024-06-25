import os
import torch
from ultralytics import YOLO
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import yaml
import tempfile

os.environ['WANDB_DISABLED'] = 'true'

# Read the data.yaml file as a global variable
with open('data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

def train_yolov8(config, checkpoint_dir=None):
    model = YOLO(config["model_path"]).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update hyperparameters
    # model.model.hyperparams['lr0'] = config["lr0"]
    # model.model.hyperparams['lr'] = config["lr"]
    # model.model.hyperparams['momentum'] = config["momentum"]
    # model.model.hyperparams['weight_decay'] = config["weight_decay"]

    # Write the data_config to a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(data_config, temp_file)
        temp_file_path = temp_file.name
    
    # Train the model
    model.train(data=temp_file_path, epochs=config["epochs"], imgsz=config["imgsz"], batch=config["batch_size"], save=True, resume=False, iou=config["iou"], conf=config["conf"])
    
    # Evaluate the model
    metrics = model.val(data=temp_file_path)
    
    # Report metrics to Ray Tune
    tune.report(mAP=metrics["metrics/mAP50(B)"])

# Define the search space
config = {
    "model_path": "yolov8n.pt",
    "lr0": tune.loguniform(1e-5, 1e-1),
    "lr": tune.loguniform(1e-6, 1e-2),
    "epochs": tune.choice([50, 100, 150, 200]),
    "imgsz": tune.choice([320, 480, 640]),
    "batch_size": tune.choice([8, 16, 32]),
    "iou": tune.uniform(0.3, 0.7),
    "conf": tune.loguniform(1e-4, 1e-1)
}

# Define the scheduler
scheduler = ASHAScheduler(
    metric="metrics/mAP50(B)",
    mode="max",
    max_t=150,
    grace_period=1,
    reduction_factor=2
)

# Run the hyperparameter search
analysis = tune.run(
    train_yolov8,
    resources_per_trial={"cpu": 2, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="mAP", mode="max")
print("Best hyperparameters found were: ", best_config)
