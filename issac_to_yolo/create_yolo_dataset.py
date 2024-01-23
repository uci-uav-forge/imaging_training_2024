from pathlib import Path
import shutil
import numpy as np
import json
from io import StringIO
from PIL import Image
from tqdm import tqdm
from yolo_config import *


DEBUG = True


# Create the target directory
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
        shutil.rmtree(TARGET_DIR, ignore_errors=True)
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

GEN_IMAGES = list(IMAGES_DIR.glob('[!.]*.png'))
NUM_IMAGES = len(GEN_IMAGES)
print(f'Found {NUM_IMAGES} images')

'''########################################################'''
# Shorten the set for testing -------------------------------
if DEBUG:
    NUM_IMAGES = 50
    GEN_IMAGES = GEN_IMAGES[:NUM_IMAGES]
# ------------------------------------------------------------

# Determine the amount of images to use for training and validation
TRAIN_PERCENTAGE = 0.8
VALID_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.05

NUM_TRAIN = int(NUM_IMAGES * TRAIN_PERCENTAGE)
NUM_VALID = int(NUM_IMAGES * VALID_PERCENTAGE)
NUM_TEST = NUM_IMAGES - NUM_TRAIN - NUM_VALID

print(f'Using {NUM_TRAIN} images for training, {NUM_VALID} images for validation, and {NUM_TEST} images for testing')

found_classes = {}

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

def create_bbox_label_file(image_path : Path):

    # Get the image id
    id_value = get_file_id(image_path)

    # Create a bbox label file for the given image id
    label_pos = get_file_by_id(id_value, BOXES_DIR) #npy type
    
    # Load the label position data
    with open(label_pos, 'rb') as label_pos_file:
        label_pos_data = np.load(label_pos_file)

    # Get the image dimensions
    img = Image.open(image_path).convert('RGB')
    image = np.array(img)
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Create temp file
    tempFile = StringIO()

    # translate label position data to yolo format
    # yolo format: <object-class> <x_center> <y_center> <width> <height>
    # entry format: <object-class> <x1> <y1> <x2> <y2>
    for entry in label_pos_data:
        if str(entry[0]) not in found_classes.keys():
             # Load the class label data
            class_label = get_file_by_id(id_value, BOXES_LEGEND_DIR) #json type
            with open(class_label) as class_label_file:
                class_label_data = json.load(class_label_file)
            found_classes[str(entry[0])] = class_label_data[str(entry[0])]["class"]

        if entry[0] != 0:
            # Calculate the yolo format values
            x_center = ((entry[1] + entry[3]) / 2) / image_width
            y_center = ((entry[2] + entry[4]) / 2) / image_height
            width = (entry[3] - entry[1]) / image_width
            height = (entry[4] - entry[2]) / image_height

            # Write the yolo format values to the temp file
            tempFile.write(f'{entry[0]} {x_center} {y_center} {width} {height}\n')
        
    return id_value, tempFile.getvalue()

def write_label_file(id_value, label_file, target_dir):
    with open(target_dir / f'{id_value}.txt', 'w') as f:
        f.write(label_file)

# Copy the images to the target directory
print('Copying images and generating label files...')
for i, image in enumerate(tqdm(GEN_IMAGES)):
    id_name, label_file = create_bbox_label_file(image)
    if i < NUM_TRAIN:
        write_label_file(id_name, label_file, TARGET_TRAIN_LABEL_DIR)
        shutil.copy(image, TARGET_TRAIN_IMG_DIR / (id_name+'.png'))
    elif i < NUM_TRAIN + NUM_VALID:
        write_label_file(id_name, label_file, TARGET_VALID_LABEL_DIR)
        shutil.copy(image, TARGET_VALID_IMG_DIR / (id_name+'.png'))
    else:
        write_label_file(id_name, label_file, TARGET_TEST_LABEL_DIR)
        shutil.copy(image, TARGET_TEST_IMG_DIR / (id_name+'.png'))

print('Done generating data!')
print("Now making data.yaml file...")

# Create the data.yaml file

def create_data_yaml_file():
    # Creates the data.yaml file

    # create a list of all the classes in order
    keys_of_classes = list(found_classes.keys())
    keys_of_classes = [int(key) for key in keys_of_classes]
    number_of_classes = max(keys_of_classes)
    print(f'Found {number_of_classes} classes')
    classes = [0]*(number_of_classes+1)
    for i in range(number_of_classes+1):
        classes[i] = found_classes.get(str(i), "UNKNOWN CLASS")

    data_yaml = [
    "train: ../train/images\n",
    "val: ../valid/images\n",
    "test: ../test/images\n\n",
    f"nc: {number_of_classes+1}\n",
    f"names: {classes}\n"
    ]
    with open(TARGET_DIR / 'data.yaml', 'w') as f:
        f.writelines(data_yaml)

create_data_yaml_file()

print('Done!')

