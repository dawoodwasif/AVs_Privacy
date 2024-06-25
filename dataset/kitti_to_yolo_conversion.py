import os
from PIL import Image
import pandas as pd
from tqdm import tqdm 

# Define the folder paths
src_image_dir = 'kitti-dataset/data_object_image_2/training/image_2'
src_label_dir = 'kitti-dataset/data_object_label_2/training/label_2'
dst_image_dir = 'images/all'
dst_label_dir = 'labels/all'

# Define the mapping from KITTI classes to class IDs for YOLO
classes = {
    'Car': 0,
    'Pedestrian': 1,
    'Van': 2,
    'Cyclist': 3,
    'Truck': 4,
    'Misc': 5,
    'Tram': 6,
    'Person_sitting': 7
}

# Make sure output directories exist
os.makedirs(dst_image_dir, exist_ok=True)
os.makedirs(dst_label_dir, exist_ok=True)

def convert_kitti_to_yolo(kitti_bbox, img_width, img_height):
    # Extract coordinates for bounding box
    x_min, y_min, x_max, y_max = kitti_bbox
    
    # Convert coordinates
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return (x_center, y_center, width, height)

def process_files():
    # Iterate over all label files in the source directory
    for label_file in tqdm(os.listdir(src_label_dir)):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.png')
            image_path = os.path.join(src_image_dir, image_file)
            
            # Read and convert image
            img = Image.open(image_path)
            img.save(os.path.join(dst_image_dir, image_file.replace('.png', '.jpg')), 'JPEG')
            img_width, img_height = img.size

            # Read label file
            with open(os.path.join(src_label_dir, label_file), 'r') as file:
                lines = file.readlines()

            yolo_labels = []
            for line in lines:
                parts = line.strip().split()
                obj_class = parts[0]
                if obj_class in classes and obj_class != 'DontCare':
                    bbox = list(map(float, parts[4:8]))
                    x_center, y_center, width, height = convert_kitti_to_yolo(bbox, img_width, img_height)
                    yolo_labels.append(f"{classes[obj_class]} {x_center} {y_center} {width} {height}")

            # Write YOLO formatted labels to file
            with open(os.path.join(dst_label_dir, label_file), 'w') as file:
                for label in yolo_labels:
                    file.write(label + '\n')

if __name__ == '__main__':
    process_files()
