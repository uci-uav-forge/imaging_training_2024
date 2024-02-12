from pathlib import Path
import shutil
import numpy as np
import json
from io import StringIO
from PIL import Image
from tqdm import tqdm
from yolo_config import *
import math
import time
# from multiprocessing import Pool (this is for processes, not threads)
from multiprocessing.pool import ThreadPool as Pool #(for threads)

if __name__ == '__main__':
    # Create the target directory
    version_number = 1

    while (TARGET_DIR / f'DATASETv{version_number}').exists():
        version_number += 1

    if not CREATE_NEW_VERSION and version_number > 1:
        version_number -= 1
        
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

    print(f'Creating target directory {TARGET_DIR}')
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

    GEN_IMAGES = list(DATA_DIR.glob('rgb*.png'))
    NUM_IMAGES = len(GEN_IMAGES)
    print(f'Found {NUM_IMAGES} images')

    '''########################################################'''
    # Shorten the set for testing -------------------------------
    if DEBUG:
        NUM_IMAGES = 100
        GEN_IMAGES = GEN_IMAGES[:NUM_IMAGES]
    # ------------------------------------------------------------

    # Determine the amount of images to use for training and validation
    TRAIN_PERCENTAGE = 0.8
    VALID_PERCENTAGE = 0.15
    TEST_PERCENTAGE = 0.05

    NUM_TRAIN = int(NUM_IMAGES * TRAIN_PERCENTAGE)
    NUM_VALID = int(NUM_IMAGES * VALID_PERCENTAGE)
    NUM_TEST = NUM_IMAGES - NUM_TRAIN - NUM_VALID

    height, width, channels = np.array(Image.open(GEN_IMAGES[0])).shape
    gen_per_image = math.ceil(height / TILE_SIZE) * math.ceil(width / TILE_SIZE)

    print(f'Using {NUM_TRAIN} images for training, {NUM_VALID} images for validation, and {NUM_TEST} images for testing')
    print(f'Each image will be split into {gen_per_image} tiles')

    found_classes = {}

def get_file_id(path_to_file : Path):
    # RGB files are named like this: 'rgb_000000.png'
    # Bbox class label files are named like this: 'bounding_box_2d_tight_000000.json'
    # Bbox legend files are named like this: 'bounding_box_2d_tight_000000.npy'
    # Semantic mask files are named like this: 'semantic_segmentation_000000.png'
    # Semantic legend files are named like this: 'semantic_segmentation_labels_000000.json'
    # Thus the file id is the number after the underscore
    return path_to_file.stem.split('_')[-1]

def get_file_by_id(id, type, root_dir : Path = None):
    # Returns the file with the given id from the given root directory
    # The file id is the number after the underscore
    id = str(id)
    if root_dir is None:
        root_dir = DATA_DIR
        if type == 'rgb':
            files = list(root_dir.glob('rgb*.png'))
        elif type == 'semantic':
            files = list(root_dir.glob('semantic_segmentation*.png'))
        elif type == 'semantic_legend':
            files = list(root_dir.glob('semantic_segmentation_labels*.json'))
        elif type == 'bbox_legend':
            files = list(root_dir.glob('bounding_box_2d_tight_labels*.json'))
        elif type == 'bbox_pos':
            files = list(root_dir.glob('bounding_box_2d_tight*.npy'))
    else:
        files = list(root_dir.glob('[!.]*'))

    for file in files:
        if get_file_id(file) == id:
            return file
    return None

def determineOverlap(section : int, total_length : int) -> tuple:
    amount = math.ceil(total_length / section)
    overlap = int(((amount * section) - total_length) / (amount-1))
    return (amount, overlap)

def sliceImage(image_arr : np.array, tileSize : int) -> list[Image.Image]:
    height, width, channels = image_arr.shape
    height_amount, height_overlap = determineOverlap(tileSize, height)
    width_amount, width_overlap = determineOverlap(tileSize, width)

    tiles = []
    pos_data = []
    for i in range(height_amount):
        for j in range(width_amount):

            y_start = i*(tileSize-height_overlap)
            y_end = y_start+tileSize

            x_start = j*(tileSize-width_overlap)
            x_end = x_start+tileSize

            tiles.append(Image.fromarray(image_arr[y_start:y_end, x_start:x_end]))
            pos_data.append([x_start, y_start, x_end, y_end])

    return tiles, pos_data

def create_bbox_label_file(image_path : Path):
    # Get the image id
    id_value = get_file_id(image_path)

    # Create a bbox label file for the given image id
    label_pos = get_file_by_id(id_value, "bbox_pos") #npy type
    
    # Load the label position data
    with open(label_pos, 'rb') as label_pos_file:
        label_pos_data = np.load(label_pos_file)

    # Get the image and convert it to an array
    img = Image.open(image_path)
    image = np.array(img)
    
    # Slice the image
    tiles, tile_pos_data = sliceImage(image, TILE_SIZE) #0.03 seconds
    label_files = []
    # Create a bbox label file for each tile
    # tile_pos_data: [tile_xmin, tile_ymin, tile_xmax, tile_ymax] per tile
    for tile_xmin, tile_ymin, tile_xmax, tile_ymax in tile_pos_data:
        # Create temp file
        tempFile = StringIO()

        # translate label position data to yolo format
        # yolo format: <object-class> <x_center> <y_center> <width> <height>
        # entry format: <object-class> <x1> <y1> <x2> <y2>

        # For each entry in the original label position data we need to check all the tiles to see if it is in the tile
        for entry in label_pos_data:

            cls, x1, y1, x2, y2, rot = entry

            if str(cls) not in found_classes.keys():
                # Load the class label data
                class_label = get_file_by_id(id_value, 'bbox_legend') #json type
                with open(class_label) as class_label_file:
                    class_label_data = json.load(class_label_file)
                found_classes[str(cls)] = class_label_data[str(cls)]["class"]

            if cls != 0:
                # tile_pos_data: [tile_xmin, tile_ymin, tile_xmax, tile_ymax] per tile
                # entry format: <object-class> <x1> <y1> <x2> <y2>
                # Check if at least one of the points are in the tile
                if ((tile_xmin <= x1 <= tile_xmax and tile_ymin <= y1 <= tile_ymax)
                or (tile_xmin <= x2 <= tile_xmax and tile_ymin <= y2 <= tile_ymax)):
                    # Means that a point is in the tile

                    def constrain(val, min_val, max_val):
                        return min(max_val, max(min_val, val))

                    # Constrain the values to be within the tile
                    x1 = constrain(x1, tile_xmin, tile_xmax)
                    y1 = constrain(y1, tile_ymin, tile_ymax)
                    x2 = constrain(x2, tile_xmin, tile_xmax)
                    y2 = constrain(y2, tile_ymin, tile_ymax)

                    # Translate the values to be relative to the tile
                    x1 -= tile_xmin
                    y1 -= tile_ymin
                    x2 -= tile_xmin
                    y2 -= tile_ymin

                    # Calculate the yolo format values
                    x_center = ((x1 + x2) / 2) / TILE_SIZE
                    y_center = ((y1 + y2) / 2) / TILE_SIZE
                    width = (x2 - x1) / TILE_SIZE
                    height = (y2 - y1) / TILE_SIZE

                    # Write the yolo format values to the temp file
                    if DEBUG: tempFile.write(f"({cls} {x1} {y1} {x2} {y2})\n")
                    tempFile.write(f'{cls} {x_center} {y_center} {width} {height}\n')
        label_files.append(tempFile.getvalue())
    names = [id_value+'_'+str(i) for i in range(len(tiles))]
    return names, tiles, label_files

def write_label_files(id_values, label_files, target_dir):
    for id_value, label_file in zip(id_values, label_files):
        with open(target_dir / f'{id_value}.txt', 'w') as f:
            f.write(label_file)

def tile_writer(id_value, tile, target_dir):
    tile.save(target_dir / f'{id_value}.png', compress_level=3)

def write_tiles(id_values, tiles, target_dir):
    pool = Pool(6) # 6 threads seems to work best
    for id_value, tile in zip(id_values, tiles):
        pool.apply_async(tile_writer, args=(id_value, tile, target_dir))
    pool.close()
    pool.join()

def generate_dataset():
    # Copy the images to the target directory
    print('Generating images and generating label files...')
    print('Dataset location:', TARGET_DIR)
    for i, image in enumerate(tqdm(GEN_IMAGES)):
        # Temporary to eliminate blury images
        if int(get_file_id(image)) % 2 == 0:
            continue
        #
        id_names, tiles, label_files = create_bbox_label_file(image)
        if i < NUM_TRAIN:
            write_label_files(id_names, label_files, TARGET_TRAIN_LABEL_DIR)
            if DEBUG: start_time = time.time()
            write_tiles(id_names, tiles, TARGET_TRAIN_IMG_DIR)
            if DEBUG: print(f"time to write tiles: {time.time() - start_time}")
        elif i < NUM_TRAIN + NUM_VALID:
            write_label_files(id_names, label_files, TARGET_VALID_LABEL_DIR)
            write_tiles(id_names, tiles, TARGET_VALID_IMG_DIR)
        else:
            write_label_files(id_names, label_files, TARGET_TEST_LABEL_DIR)
            write_tiles(id_names, tiles, TARGET_TEST_IMG_DIR)

    print('Done generating data!')
    print("Now making data.yaml file...")

# Create the data.yaml file

def create_data_yaml_file():
    # Creates the data.yaml file

    # create a list of all the classes in order
    keys_of_classes = list(found_classes.keys())
    keys_of_classes = [int(key) for key in keys_of_classes]
    number_of_classes = max(keys_of_classes)
    print(f'Found {number_of_classes+1} classes')
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

if __name__ == '__main__':
    generate_dataset()
    create_data_yaml_file()

    print('Done!')

