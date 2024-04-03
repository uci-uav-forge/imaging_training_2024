from pathlib import Path
import random
import shutil
from typing import Generator, Iterable, Literal, NamedTuple
import numpy as np
import json
from io import StringIO
from PIL import Image
from tqdm import tqdm
from yolo_config import TARGET_DIR, CREATE_NEW_VERSION, TILE_SIZE, DEBUG, TRAIN_RATIO, VAL_RATIO, DATA_DIR
import math
import time
# from multiprocessing import Pool (this is for processes, not threads)
from multiprocessing.pool import ThreadPool as Pool #(for threads)

class Tile(NamedTuple):
    image: Image.Image
    x_start: int
    y_start: int
    x_end: int
    y_end: int


class TileData(NamedTuple):
    id: str
    image: Image.Image
    labels: str


class OutputLocations(NamedTuple):
    train_img: Path
    train_label: Path
    valid_img: Path
    valid_label: Path
    test_img: Path
    test_label: Path
    
    @staticmethod
    def create_from_base_path(base: Path, create_new: bool) -> "OutputLocations":
        """
        Finds the next available version directory within the base directory and creates the necessary subdirectories.
        
        If create_new is True, it will create a new version directory if one already exists.
        
        If create_new is False, it will overwrite the highest contiguous version directory.
        """
        version_dir = OutputLocations._create_version_dir(base, create_new)
        
        train_dir = version_dir / 'train'
        valid_dir = version_dir / 'valid'
        test_dir = version_dir / 'test'
                
        train_img = train_dir / 'images'
        train_label = train_dir / 'labels'
        valid_img = valid_dir / 'images'
        valid_label = valid_dir / 'labels'
        test_img = test_dir / 'images'
        test_label = test_dir / 'labels'
        
        for dir in [train_img, train_label, valid_img, valid_label, test_img, test_label]:
            dir.mkdir(parents=True)
        
        return OutputLocations(
            train_img=train_img,
            train_label=train_label,
            valid_img=valid_img,
            valid_label=valid_label,
            test_img=test_img,
            test_label=test_label
        )
        
    @staticmethod
    def _create_version_dir(base: Path, create_new: bool) -> Path:
        """
        Find a dataset dir within the base dir following the pattern [base]/DATASETv[version] for the new dataset.
        
        If create_new is True, it will create a new version directory if one already exists.
        
        If create_new is False, it will overwrite the highest contiguous version directory.
        """
        version = 1

        while (base / f'DATASETv{version}').exists():
            version += 1

        if not create_new and version > 1:
            version -= 1
        
        target = base / f'DATASETv{version}'
        
        if target.exists():
            print(f'Warning: Target directory {target} already exists. Overwriting.')
            data = input('Enter "yes" to confirm deletion...')
            if data == 'yes':
                shutil.rmtree(target, ignore_errors=True)
            else:
                print('Aborting...')
                exit()
                
        print(f'Creating target directory {target}')
        target.mkdir(parents=True)
        
        return target


class YOLOFormatter:
    def __init__(self, output_locations: OutputLocations, tile_size: int):
        self.found_classes: dict[str, str]= {}
        self.output_locations = output_locations
        self.tile_size = tile_size
    
    @staticmethod
    def get_file_id(path_to_file : Path):
        """
        Gets the file id from the given file path.
        """
        # RGB files are named like this: 'rgb_000000.png'
        # Bbox class label files are named like this: 'bounding_box_2d_tight_000000.json'
        # Bbox legend files are named like this: 'bounding_box_2d_tight_000000.npy'
        # Semantic mask files are named like this: 'semantic_segmentation_000000.png'
        # Semantic legend files are named like this: 'semantic_segmentation_labels_000000.json'
        # Thus the file id is the number after the underscore
        return path_to_file.stem.split('_')[-1]

    @staticmethod
    def get_file_by_id(
        id: str, 
        type: Literal['rgb', 'semantic', 'semantic_legend', 'bbox_legend', 'bbox_pos'], 
        root_dir : Path|None = None
    ) -> Path:
        """
        Gets the Path for a file with the given id and type.
        
        The file ID is the number after the underscore in the file name.
        E.g., 'rgb_420.png' has the ID '420'.
        """
        # Returns the file with the given id from the given root directory
        # The file id is the number after the underscore
        if root_dir is None:
            root_dir = DATA_DIR
            match type:
                case 'rgb':
                    pattern = 'rgb*.png'
                case 'semantic':
                    pattern = 'semantic_segmentation*.png'
                case 'semantic_legend':
                    pattern = 'semantic_segmentation_labels*.json'
                case 'bbox_legend':
                    pattern = 'bounding_box_2d_tight_labels*.json'
                case 'bbox_pos':
                    pattern = 'bounding_box_2d_tight*.npy'
                case _:
                    raise ValueError(f'Invalid type {type}')
        else:
            pattern = '[!.]*'

        for file in root_dir.glob(pattern):
            if YOLOFormatter.get_file_id(file) == id:
                return file
        
        raise FileNotFoundError(f'File with id {id} and type {type} not found in {root_dir}')

    @staticmethod
    def _size_and_overlap(section: int, total_length: int) -> tuple[int, int]:
        amount = math.ceil(total_length / section)
        overlap = int(((amount * section) - total_length) / (amount-1))
        return amount, overlap

    @staticmethod
    def _subtile(
        image_arr : np.ndarray, 
        tileSize : int
    ) -> Generator[Tile, None, None]:
        """
        Splits an image array into tiles of the given size and yields each tile.
        
        The tiles are yielded in row-major order.
        """
        height, width, channels = image_arr.shape
        
        height_amount, height_overlap = YOLOFormatter._size_and_overlap(tileSize, height)
        width_amount, width_overlap = YOLOFormatter._size_and_overlap(tileSize, width)

        for i in range(height_amount):
            for j in range(width_amount):

                y_start = i*(tileSize-height_overlap)
                y_end = y_start+tileSize

                x_start = j*(tileSize-width_overlap)
                x_end = x_start+tileSize
                
                tile = Image.fromarray(image_arr[y_start:y_end, x_start:x_end])

                yield Tile(
                    image=tile,
                    x_start=x_start,
                    y_start=y_start,
                    x_end=x_end,
                    y_end=y_end
                )

    def _get_subtile_data(self, image_path: Path) -> Generator[TileData, None, None]:
        """
        Yields the labels for each subtile of the given image.
        
        If an unknown class is found, it will be added to the `found_classes` dictionary.
        
        Returns:
            A generator of tuples containing the subtile id, the subtile image, and the label file content.
        """
        # Get the image id and the labels file
        image_id = YOLOFormatter.get_file_id(image_path)
        label_pos = YOLOFormatter.get_file_by_id(image_id, "bbox_pos") #npy type

        # Load the label position data
        label_pos_data = np.load(label_pos)

        # Get the image and convert it to an array
        source = np.array(Image.open(image_path))
        
        # Create a bbox label file for each tile
        # tile_pos_data: [tile_xmin, tile_ymin, tile_xmax, tile_ymax] per tile
        for tile_id, (subtile, tile_xmin, tile_ymin, tile_xmax, tile_ymax) in enumerate(YOLOFormatter._subtile(source, self.tile_size)):
            # TODO: Decompose the following code into smaller functions
            
            # Create temp file
            tempFile = StringIO()

            # translate label position data to yolo format
            # yolo format: <object-class> <x_center> <y_center> <width> <height>
            # entry format: <object-class> <x1> <y1> <x2> <y2>

            # For each entry in the original label position data we need to check all the tiles to see if it is in the tile
            for entry in label_pos_data:

                cls, x1, y1, x2, y2, rot = entry

                if str(cls) not in self.found_classes.keys():
                    # Load the class label data
                    class_label = YOLOFormatter.get_file_by_id(image_id, 'bbox_legend') #json type
                    with open(class_label) as class_label_file:
                        class_label_data = json.load(class_label_file)
                    self.found_classes[str(cls)] = class_label_data[str(cls)]["class"]

                if cls == 0: continue # Skip background
                
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
                    x_center = ((x1 + x2) / 2) / self.tile_size
                    y_center = ((y1 + y2) / 2) / self.tile_size
                    width = (x2 - x1) / self.tile_size
                    height = (y2 - y1) / self.tile_size

                    # Write the yolo format values to the temp file
                    if DEBUG: tempFile.write(f"({cls} {x1} {y1} {x2} {y2})\n")
                    tempFile.write(f'{cls} {x_center} {y_center} {width} {height}\n')
        
        yield TileData(f"{image_id}_{tile_id}", subtile, tempFile.getvalue())
        
    @staticmethod
    def _write_tiles_and_labels(tile_data_iterable: Iterable[TileData], tiles_dir: Path, labels_dir: Path):
        def _write_tile(id: str, tile: Image.Image):
            tile.save(tiles_dir / f'{id}.png', compress_level=3)
            
        def _write_label(id: str, label: str):
            with open(labels_dir / f'{id}.txt', 'w') as f:
                f.write(label)
                
        pool = Pool(6) # 6 threads seems to work best
        
        for id, tile, labels in tile_data_iterable:
            pool.apply_async(_write_tile, args=(id, tile))
            pool.apply_async(_write_label, args=(id, labels))
            
        pool.close()
        pool.join()

    def _create_tiles(
        self,
        output_locations: OutputLocations,
        num_train: int,
        num_valid: int,
        source_paths: list[Path]
    ):
        # Copy the images to the target directory
        print('Dataset location:', TARGET_DIR)
        
        for i, image in enumerate(
            tqdm(
                random.choices(source_paths, k=len(source_paths)), # Randomize the order of the images
                total=len(source_paths),
                desc='Generating tiles...',
                unit='images',
                position=0,
                leave=True
            )
        ):
            # Temporary to eliminate blury images
            # Skip every other image
            if int(YOLOFormatter.get_file_id(image)) % 2 == 0:
                continue
            
            if i < num_train:
                tiles_dir = output_locations.train_img
                label_dir = output_locations.train_label
            elif i < num_train + num_valid:
                tiles_dir = output_locations.valid_img
                label_dir = output_locations.valid_label
            else:
                tiles_dir = output_locations.test_img
                label_dir = output_locations.test_label
            
            if DEBUG: start_time = time.time()
            YOLOFormatter._write_tiles_and_labels(self._get_subtile_data(image), tiles_dir, label_dir)
            if DEBUG: print(f"time to write tiles: {time.time() - start_time}")

        print('Done generating data!')

    def _create_data_yaml_file(self):
        # Creates the data.yaml file

        # create a list of all the classes in order
        class_keys = list(map(int, self.found_classes.keys()))
        number_of_classes = max(class_keys)
        print(f'Found {number_of_classes+1} classes')
        
        classes: list[str] = [""]*(number_of_classes+1)
        for i in range(number_of_classes+1):
            classes[i] = self.found_classes.get(str(i), "UNKNOWN CLASS")

        data_yaml = [
        "train: ../train/images\n",
        "val: ../valid/images\n",
        "test: ../test/images\n\n",
        f"nc: {number_of_classes+1}\n",
        f"names: {classes}\n"
        ]
        with open(TARGET_DIR / 'data.yaml', 'w') as f:
            f.writelines(data_yaml)
            
    def create_dataset(self, num_train: int, num_valid: int, source_paths: list[Path]):
        print('Generating images and generating label files...')
        self._create_tiles(self.output_locations, num_train, num_valid, source_paths)
        
        print("Now making data.yaml file...")
        self._create_data_yaml_file()

def create_yolo_dataset():
    output_locations = OutputLocations.create_from_base_path(TARGET_DIR, CREATE_NEW_VERSION)
    
    formatter = YOLOFormatter(output_locations, TILE_SIZE)

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

    print(f'Using {num_train} images for training, {num_valid} images for validation, and {num_test} images for testing')
    print(f'Each image will be split into {tiles_per_image} tiles')

    formatter.create_dataset(num_train, num_valid, source_img_paths)
    print('Done!')


if __name__ == '__main__':
    create_yolo_dataset()    
