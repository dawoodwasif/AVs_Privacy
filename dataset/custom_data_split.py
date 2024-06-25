import os
import shutil
import random
from tqdm import tqdm 

# Define paths
src_images_dir = 'images/all'
src_labels_dir = 'labels/all'

train_images_dir = 'images/train'
train_labels_dir = 'labels/train'
val_images_dir = 'images/validation'
val_labels_dir = 'labels/validation'
test_images_dir = 'images/test'
test_labels_dir = 'labels/test'

# Split Ratios
split_ratios = {
    'train': 0.8,  # 80% for training
    'val': 0.1,    # 10% for validation
    'test': 0.1    # 10% for testing
}

def create_directories():
    # Create directories for the dataset split if they do not exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

def split_data():
    # Get all image filenames
    images = [f for f in os.listdir(src_images_dir) if f.endswith('.jpg')]
    random.shuffle(images)  # Shuffle to randomize the distribution

    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * split_ratios['train'])
    val_end = train_end + int(total_images * split_ratios['val'])

    # Split into train, validation, and test
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Function to copy files
    def copy_files(files, src_dir, dst_img_dir, dst_lbl_dir):
        for file in tqdm(files):
            # Copy image
            img_src = os.path.join(src_dir, file)
            img_dst = os.path.join(dst_img_dir, file)
            shutil.copy(img_src, img_dst)

            # Copy corresponding label
            label_file = file.replace('.jpg', '.txt')
            label_src = os.path.join(src_labels_dir, label_file)
            label_dst = os.path.join(dst_lbl_dir, label_file)
            shutil.copy(label_src, label_dst)

    # Copy files to respective directories
    copy_files(train_files, src_images_dir, train_images_dir, train_labels_dir)
    copy_files(val_files, src_images_dir, val_images_dir, val_labels_dir)
    copy_files(test_files, src_images_dir, test_images_dir, test_labels_dir)

if __name__ == '__main__':
    create_directories()
    split_data()
    print("Data split into training, validation, and test sets.")
