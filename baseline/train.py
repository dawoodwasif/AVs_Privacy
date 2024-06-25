from ultralytics import YOLO
import os

# Disable Weights and Biases logging
os.environ['WANDB_DISABLED'] = 'true'

# Define hyperparameters
epochs = 100
learning_rate = 1e-6  
img_size = 640
batch_size = 16
iou_threshold = 0.5
confidence_threshold = 0.001
save_period = 1

# Custom path for saving the model
project_name = 'AV_privacy'
experiment_name = 'baseline_training'

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data='data.yaml',  # Path to the dataset configuration file
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    save=True,
    resume=False,
    iou=iou_threshold,
    conf=confidence_threshold,
    save_period=save_period,
    lr0=learning_rate,
    lrf=learning_rate    
)

# Validate the model
val_results = model.val(data="data.yaml",  split="test")

# Print the results for verification
# print("Training Results:", train_results)
# print("Validation Results:", val_results)
