# Federated Learning Object Detection with YOLOv8

This repository contains the necessary code and configurations for running object detection tasks using YOLOv8 within a Federated Learning framework, specifically tailored for environments where data privacy is crucial. The system has been designed to work effectively with the KITSE Dataset for autonomous vehicle applications.

## Error Handling Improvement

During the development and initial runs, an error was identified related to empty metric arrays when interpolating for plotting purposes. The issue occurs when the metric array `y` intended for plotting remains empty due to conditional errors in data collection or failures in metric computation during training phases. To handle this scenario gracefully and avoid runtime errors, the following conditional check was introduced in the plotting function:

instead of line 89 in ultralytics/utils/callbacks/wb.py:
`y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()`

replace it by:

```
if y.size == 0:
    y_mean = 0
    y_log = 0
else:
    y_mean = np.mean(y, axis=0)
    y_log = np.interp(x_new, x, y_mean).round(3).tolist()
```   
This adjustment ensures that the system handles cases of missing data without interruption and avoids the ValueError that occurs when attempting to interpolate with an empty dataset.

Environment Setup and Running Instructions

### Clone the Repository

git clone <repository-url>
cd <repository-directory>

Create and Activate a Python Environment
It's recommended to use a virtual environment to manage dependencies effectively:


### Requirements


`conda create -y --name privacy python=3.10`

`conda activate privacy`

`pip install -r requirements.txt`

`apt-get install libglib2.0-0`


In yolov8.yaml, in line 5 change the number of classes according to the dataset:

`nc: 1  # number of classes 8 for KITTI and 1 for FACET`

Run the Training Script

Execute the training process by running:

`python FedAvg_train.py`

Configuration Files

Before running the training script, ensure that you have all the necessary configuration files in place, as specified in config_paths within train.py. These files should define the dataset paths and other training parameters specific to each federated client.