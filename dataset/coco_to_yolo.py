import json
import os
import shutil
import pandas as pd
from tqdm import tqdm

# Load the CSV file
annotations_csv_path = 'annotations/annotations.csv'
annotations_df = pd.read_csv(annotations_csv_path)

# Create a dictionary to map person_id to attributes
person_attributes = {}
for _, row in annotations_df.iterrows():
    person_id = row['person_id']
    attributes = row.to_dict()
    person_attributes[person_id] = attributes


# Paths to the necessary files and directories
coco_boxes_path = 'annotations/coco_boxes.json'
image_dirs = ['imgs_1', 'imgs_2', 'imgs_3']
output_images_dir = 'images/all/'
output_labels_dir = 'labels/all/'
output_metadata_file = 'metadata.json'

# Load COCO annotations
with open(coco_boxes_path, 'r') as f:
    coco_data = json.load(f)

# Create output directories if they do not exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Dictionary to store metadata
metadata = {}

def find_image(image_name, image_dirs):
    for directory in image_dirs:
        image_path = os.path.join(directory, image_name)
        if os.path.exists(image_path):
            return image_path
    return None

# Process each annotation
for annotation in tqdm(coco_data['annotations']):
    person_id = annotation['person_id']
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']  # COCO bbox format: [x_min, y_min, width, height]
    
    # Get image dimensions
    image_info = next(item for item in coco_data['images'] if item['id'] == image_id)
    img_width = image_info['width']
    img_height = image_info['height']
    img_file_name = image_info['file_name']
    
    # Find and copy image to the new directory
    src_image_path = find_image(img_file_name, image_dirs)
    if src_image_path:
        dst_image_path = os.path.join(output_images_dir, img_file_name)
        if not os.path.exists(dst_image_path):
            shutil.copy(src_image_path, dst_image_path)
    else:
        print(f"Image {img_file_name} not found in the specified directories.")
        continue
    
    # Convert COCO bbox to YOLO format
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height

    # Create YOLO label string
    yolo_label = f"{category_id} {x_center} {y_center} {width} {height}\n"
    
    # Save label to corresponding file
    label_file_name = f"{img_file_name.split('.')[0]}.txt"
    label_file_path = os.path.join(output_labels_dir, label_file_name)
    with open(label_file_path, 'a') as f:
        f.write(yolo_label)
    
    # Add to metadata dictionary
    if image_id not in metadata:
        metadata[image_id] = {'image_file': img_file_name, 'persons': []}
    
    # Retrieve attributes for the person
    attributes = person_attributes.get(person_id, {})
    attributes.update({
        'person_id': person_id,
        'bbox': [x_center, y_center, width, height],
        'category_id': category_id
    })
    metadata[image_id]['persons'].append(attributes)

# Save metadata to a JSON file
with open(output_metadata_file, 'w') as f:
    json.dump(metadata, f, indent=4)

print("Conversion to YOLO format completed and metadata saved.")
