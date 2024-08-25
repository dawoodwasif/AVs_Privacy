# from roboflow import Roboflow

# %cd /home/AVs_Privacy/attack_models
# # Import the Roboflow library and create an instance with the provided API key
# api_key = "4DcIf06sr4RzL8YOksef"
# rf = Roboflow(api_key)

# # Access the Roboflow workspace named "lazydevs"
# # Access the project named "human-detection" within the workspace
# workspace_name = "lazydevs"
# project_name = "human-dectection"
# project = rf.workspace(workspace_name).project(project_name)

# # Download the dataset associated with version 4 of the project using YOLOv8 format
# # Note: You might want to include a specific version number or method for version selection.
# version = 4
# form = "yolov8"
# dataset = project.version(version).download(form)

import os
import shutil

# Define source directories
source_dirs = [
    '/home/AVs_Privacy/attack_models/Human-Dectection-4/train/images',
    '/home/AVs_Privacy/attack_models/Human-Dectection-4/train/labels',
    '/home/AVs_Privacy/attack_models/Human-Dectection-4/val/images',
    '/home/AVs_Privacy/attack_models/Human-Dectection-4/val/labels',
    '/home/AVs_Privacy/attack_models/Human-Dectection-4/test/images',
    '/home/AVs_Privacy/attack_models/Human-Dectection-4/test/labels',
    '/home/Virginia_Research/FACET/images/test',
    '/home/Virginia_Research/FACET/labels/test'
]

# Define target directories
target_image_dir = '/home/AVs_Privacy/attack_models/Human-Dectection-4/all/images'
target_label_dir = '/home/AVs_Privacy/attack_models/Human-Dectection-4/all/labels'

# Create target directories if they don't exist
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

# Function to copy files
def copy_files(source, target):
    if os.path.exists(source):
        for filename in os.listdir(source):
            src_file = os.path.join(source, filename)
            dst_file = os.path.join(target, filename)
            shutil.copy(src_file, dst_file)

# Copy images
for source in source_dirs:
    if 'images' in source:
        copy_files(source, target_image_dir)

# Copy labels
for source in source_dirs:
    if 'labels' in source:
        copy_files(source, target_label_dir)

print("Files copied successfully!")


