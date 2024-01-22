from pathlib import Path
from sim_data_interface import config
import shutil
import numpy as np
import json
from io import StringIO

# The base output directory for the YOLO dataset
TARGET_DIR = Path('/Volumes/SANDISK/YOLO_DATASET/') # Change this to your desired output directory

CREATE_NEW_VERSION = False

version_number = 1

if CREATE_NEW_VERSION:
    while (TARGET_DIR / f'DATASETv{version_number}').exists():
        version_number += 1

TARGET_DIR = TARGET_DIR / f'DATASETv{version_number}'

# Create the target directory
if TARGET_DIR.exists():
    print(f'Warning: Target directory {TARGET_DIR} already exists. Overwriting.')
    data = input('Enter "yes" to confirm deletion...')
    if data == 'yes':
        shutil.rmtree(TARGET_DIR)
    else:
        print('Aborting...')
        exit()

TARGET_DIR.mkdir(parents=True)

# Create the subdirectories
TARGET_TEST_DIR = TARGET_DIR / 'test'
TARGET_TRAIN_DIR = TARGET_DIR / 'train'
TARGET_VALID_DIR = TARGET_DIR / 'valid'

# Create the subdirectories for each class
TARGET_TEST_IMG_DIR = TARGET_TEST_DIR / 'images'
TARGET_TEST_LABEL_DIR = TARGET_TEST_DIR / 'labels'

TARGET_TRAIN_IMG_DIR = TARGET_TRAIN_DIR / 'images'
TARGET_TRAIN_LABEL_DIR = TARGET_TRAIN_DIR / 'labels'

TARGET_VALID_IMG_DIR = TARGET_VALID_DIR / 'images'
TARGET_VALID_LABEL_DIR = TARGET_VALID_DIR / 'labels'

TARGET_TEST_IMG_DIR.mkdir(parents=True)
TARGET_TEST_LABEL_DIR.mkdir(parents=True)

TARGET_TRAIN_IMG_DIR.mkdir(parents=True)
TARGET_TRAIN_LABEL_DIR.mkdir(parents=True)

TARGET_VALID_IMG_DIR.mkdir(parents=True)
TARGET_VALID_LABEL_DIR.mkdir(parents=True)

# Determine the amount of images

GEN_IMAGES = list(config.IMAGES_DIR.glob('[!.]*.png'))
NUM_IMAGES = len(GEN_IMAGES)
print(f'Found {NUM_IMAGES} images')

# Shorten the set for testing -----
NUM_IMAGES = 100
GEN_IMAGES = GEN_IMAGES[:NUM_IMAGES]
# -------------------------------

# Determine the amount of images to use for training and validation
TRAIN_PERCENTAGE = 0.8
VALID_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.05

NUM_TRAIN = int(NUM_IMAGES * TRAIN_PERCENTAGE)
NUM_VALID = int(NUM_IMAGES * VALID_PERCENTAGE)
NUM_TEST = NUM_IMAGES - NUM_TRAIN - NUM_VALID

print(f'Using {NUM_TRAIN} images for training, {NUM_VALID} images for validation, and {NUM_TEST} images for testing')

def get_file_id(path_to_file : Path):
    # RGB files are named like this: 'rgb_000000.png'
    # Bbox class label files are named like this: 'bounding_box_2d_tight_000000.json'
    # Bbox legend files are named like this: 'bounding_box_2d_tight_000000.npy'
    # Semantic mask files are named like this: 'semantic_segmentation_000000.png'
    # Semantic legend files are named like this: 'semantic_segmentation_labels_000000.json'
    # Thus the file id is the number after the underscore
    return path_to_file.stem.split('_')[-1]

def get_file_by_id(id, root_dir : Path):
    # Returns the file with the given id from the given root directory
    # The file id is the number after the underscore
    id = str(id)
    files = list(root_dir.glob('[!.]*'))
    #files = [get_file_id(file) for file in files]
    for file in files:
        if get_file_id(file) == id:
            return file
    return None

def create_bbox_label_file(id : int):
    # Create a bbox label file for the given image id
    class_label = get_file_by_id(id, config.BOXES_LEGEND_DIR) #json type
    label_pos = get_file_by_id(id, config.BOXES_DIR) #npy type
    
    # Load the class label data
    with open(class_label) as class_label_file:
        class_label_data = json.load(class_label_file)
    # Load the label position data
    with open(label_pos, 'rb') as label_pos_file:
        label_pos_data = np.load(label_pos_file)

    # Create the label file
    #tempFile = StringIO()
    print(label_pos_data)

# Copy the images to the target directory
# print('Copying images...')
# for i, image in enumerate(GEN_IMAGES):
#     if i < NUM_TRAIN:
#         shutil.copy(image, TARGET_TRAIN_IMG_DIR / image.name)
#     elif i < NUM_TRAIN + NUM_VALID:
#         shutil.copy(image, TARGET_VALID_IMG_DIR)
#     else:
#         shutil.copy(image, TARGET_TEST_IMG_DIR)


if __name__ == '__main__':
    create_bbox_label_file("0000")


