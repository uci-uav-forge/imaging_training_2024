import math

import numpy as np
from PIL import Image

from .formatter_types import ClassSelection, OutputLocations
from .yolo_config import TARGET_DIR, CREATE_NEW_VERSION, TILE_SIZE, DATA_DIR, DEBUG, TRAIN_RATIO, VAL_RATIO
from .yolo_formatter import YOLOFormatter


def create_yolo_dataset(class_selection: ClassSelection = ClassSelection.SHAPES_AND_CHARACTERS):
    """
    Driver function to create the YOLO dataset.

    Parameters:
        class_selection: The category of classes to include in the dataset
    """
    output_locations = OutputLocations.create_from_base_path(TARGET_DIR, CREATE_NEW_VERSION)

    formatter = YOLOFormatter(output_locations, class_selection, TILE_SIZE)

    # Determine the amount of images
    source_img_paths = list(DATA_DIR.glob('rgb*.png'))
    num_imgs_source = len(source_img_paths)
    print(f'Found {num_imgs_source} images')

    # Shorten the set for testing -------------------------------
    if DEBUG:
        num_imgs_source = 100
        source_img_paths = source_img_paths[:num_imgs_source]
    # ------------------------------------------------------------

    # Determine the amount of images to use for training and validation
    num_train = int(num_imgs_source * TRAIN_RATIO)
    num_valid = int(num_imgs_source * VAL_RATIO)
    num_test = num_imgs_source - num_train - num_valid

    height, width, channels = np.array(Image.open(source_img_paths[0])).shape
    tiles_per_image = math.ceil(height / TILE_SIZE) * math.ceil(width / TILE_SIZE)

    print(
        f'Using {num_train} images for training, {num_valid} images for validation, and {num_test} images for testing')
    print(f'Each image will be split into {tiles_per_image} tiles')

    formatter.create_dataset(num_train, num_valid, source_img_paths)
    print('Done!')


if __name__ == '__main__':
    create_yolo_dataset(ClassSelection.SHAPES_AND_CHARACTERS)
