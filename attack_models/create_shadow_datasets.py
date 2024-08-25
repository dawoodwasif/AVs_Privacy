import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Paths to image and label folders
image_folder = '/home/AVs_Privacy/attack_models/Human-Dectection-4/all/images'
label_folder = '/home/AVs_Privacy/attack_models/Human-Dectection-4/all/labels'

# List all images and corresponding labels
images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])
labels = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.txt')])

# Ensure that each image has a corresponding label
data_pairs = list(zip(images, labels))
random.shuffle(data_pairs)

# Function to create shadow datasets
# We use a 80 10 10 split
def create_shadow_datasets(data_pairs, num_splits):
    shadow_datasets = []

    for split in num_splits:
        # Calculate the number of samples for the current split
        num_samples = int(len(data_pairs) * split)
        
        # Create the main train and test split
        selected_data, remaining_data = train_test_split(data_pairs, train_size=num_samples, random_state=42)
        
        # Further split the selected data into train (80%), validation (10%), and test (10%)
        train_data, val_test_data = train_test_split(selected_data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)
        
        shadow_datasets.append({'train': train_data, 'val': val_data, 'test': test_data})

    return shadow_datasets


# Create directories for shadow datasets
def organize_shadow_datasets(shadow_datasets, base_path):
    for i, dataset in tqdm(enumerate(shadow_datasets)):
        shadow_train_image_dir = os.path.join(base_path, f'shadow_{i+1}/train/images')
        shadow_train_label_dir = os.path.join(base_path, f'shadow_{i+1}/train/labels')
        shadow_val_image_dir = os.path.join(base_path, f'shadow_{i+1}/val/images')
        shadow_val_label_dir = os.path.join(base_path, f'shadow_{i+1}/val/labels')
        shadow_test_image_dir = os.path.join(base_path, f'shadow_{i+1}/test/images')
        shadow_test_label_dir = os.path.join(base_path, f'shadow_{i+1}/test/labels')
        
        os.makedirs(shadow_train_image_dir, exist_ok=True)
        os.makedirs(shadow_train_label_dir, exist_ok=True)
        os.makedirs(shadow_val_image_dir, exist_ok=True)
        os.makedirs(shadow_val_label_dir, exist_ok=True)
        os.makedirs(shadow_test_image_dir, exist_ok=True)
        os.makedirs(shadow_test_label_dir, exist_ok=True)
        
        # Copy train data
        for img, lbl in dataset['train']:
            shutil.copy(img, shadow_train_image_dir)
            shutil.copy(lbl, shadow_train_label_dir)
        
        # Copy validation data
        for img, lbl in dataset['val']:
            shutil.copy(img, shadow_val_image_dir)
            shutil.copy(lbl, shadow_val_label_dir)

        # Copy test data
        for img, lbl in dataset['test']:
            shutil.copy(img, shadow_test_image_dir)
            shutil.copy(lbl, shadow_test_label_dir)


# Define split sizes (e.g., 50%, 60%, 70%)
split_sizes = [0.5, 0.6, 0.7]
shadow_datasets = create_shadow_datasets(data_pairs, split_sizes)

base_path = '/home/AVs_Privacy/attack_models/Human-Dectection-4/shadow_datasets'
organize_shadow_datasets(shadow_datasets, base_path)


def create_yaml_file(base_path, dataset_name):
    data = {
        'names': ['Human'],
        'nc': 1,
        'train': os.path.join(base_path, f'{dataset_name}/train/images'),
        'val': os.path.join(base_path, f'{dataset_name}/val/images'),
        'test': os.path.join(base_path, f'{dataset_name}/test/images')
    }
    
    yaml_path = os.path.join(base_path, f'{dataset_name}.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

for i in range(len(shadow_datasets)):
    create_yaml_file(base_path, f'shadow_{i+1}')


