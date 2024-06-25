import os
import shutil
import random
from tqdm import tqdm

# Parameters
num_clients = 4
src_images_dir = 'images/train'
src_labels_dir = 'labels/train'

def create_client_directories(base_dir, num_clients):
    """ Create directories for each client """
    client_dirs = []
    for i in range(num_clients):
        img_dir = os.path.join(base_dir, f'images/train{chr(65 + i)}')
        lbl_dir = os.path.join(base_dir, f'labels/train{chr(65 + i)}')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        client_dirs.append((img_dir, lbl_dir))
    return client_dirs

def distribute_files_to_clients(src_images_dir, src_labels_dir, client_dirs, num_clients):
    """ Distribute files randomly to client directories """
    # Get all image files and shuffle them
    images = [f for f in os.listdir(src_images_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    # Calculate how many images each client should get
    total_images = len(images)
    images_per_client = total_images // num_clients

    # Distribute images to each client
    for i, (img_dir, lbl_dir) in tqdm(enumerate(client_dirs)):
        # Determine the slice of images for this client
        start_index = i * images_per_client
        if i == num_clients - 1:
            end_index = total_images  # Ensure the last client gets any remaining images due to integer division
        else:
            end_index = start_index + images_per_client

        # Copy image and label files
        for img_file in images[start_index:end_index]:
            # Copy image
            shutil.copy(os.path.join(src_images_dir, img_file), os.path.join(img_dir, img_file))
            # Copy corresponding label
            label_file = img_file.replace('.jpg', '.txt')
            shutil.copy(os.path.join(src_labels_dir, label_file), os.path.join(lbl_dir, label_file))

if __name__ == '__main__':
    # Create directories for each client
    client_directories = create_client_directories('.', num_clients)
    
    # Distribute files to clients
    distribute_files_to_clients(src_images_dir, src_labels_dir, client_directories, num_clients)
    
    print(f"Successfully distributed files into {num_clients} client directories.")
