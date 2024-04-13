from enum import Enum
from pathlib import Path
import random
import shutil
from typing import Callable, Generator, Iterable, Literal, NamedTuple
import numpy as np
import json
from io import StringIO
from PIL import Image
from tqdm import tqdm
from yolo_config import TARGET_DIR, CREATE_NEW_VERSION, TILE_SIZE, DEBUG, TRAIN_RATIO, VAL_RATIO, DATA_DIR
import math
import time
# from multiprocessing import Pool (this is for processes, not threads)
from multiprocessing.pool import ThreadPool as Pool  #(for threads)


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


class Bbox(NamedTuple):
    y: float
    x: float
    height: float
    width: float


class OutputLocations(NamedTuple):
    base_dir: Path
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
            base_dir=version_dir,
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


class ClassSelection(Enum):
    """
    NOTE: Add color if that becomes a thing
    """
    SHAPES = 0
    CHARACTERS = 1
    SHAPES_AND_CHARACTERS = 2


class ClassnameMap:
    """
    Bidirectional mapping between classnames and class ids.
    
    Class IDs are 0-indexed and counted upwards.
    
    Removal of a class decreases the class id of all classes with a higher id.
    """

    def __init__(self):
        self.classname_to_id: dict[str, int] = {}
        self.id_to_classname: dict[int, str] = {}

    def add_class(self, classname: str) -> int:
        """
        Adds a class to the mapping and returns the class id.
        """
        if classname in self.classname_to_id:
            return self.classname_to_id[classname]

        class_id = len(self.classname_to_id)
        self.classname_to_id[classname] = class_id
        self.id_to_classname[class_id] = classname

        return class_id

    def remove_class(self, classname: str):
        """
        Removes a class from the mapping.
        
        Raises a KeyError if the class is not in the mapping.
        """
        class_id = self.classname_to_id.pop(classname)
        del self.id_to_classname[class_id]

        for id, name in self.id_to_classname.items():
            if id > class_id:
                self.classname_to_id[name] -= 1

    def get_class_id(self, classname: str) -> int:
        """
        Gets the class id for the given classname.
        
        Raises a KeyError if the class is not in the mapping.
        """
        return self.classname_to_id[classname]

    def get_classname(self, class_id: int) -> str:
        """
        Gets the classname for the given class id.
        
        Raises a KeyError if the class id is not in the mapping.
        """
        return self.id_to_classname[class_id]

    def classnames(self) -> Generator[str, None, None]:
        """
        Yields the classnames in the mapping in order of class id.
        """
        for id in range(len(self)):
            yield self.id_to_classname[id]

    def ids(self) -> Generator[int, None, None]:
        """
        Yields the class ids in the mapping in order of class id.
        """
        for id in range(len(self)):
            yield id

    def __len__(self):
        return len(self.classname_to_id)

    def __iter__(self):
        """
        Yields the classnames in the mapping in order of class id.
        """
        return self.classnames()

    def __contains__(self, classname: str):
        """
        Whether the name is in the mapping.
        """
        return classname in self.classname_to_id


class YOLOFormatter:
    """
    Filter functions for each class category that take in a class name and return True if the class should be included.
    """
    FILTER_FUNCTIONS: dict[ClassSelection, Callable[[str], bool]] = {
        ClassSelection.SHAPES: lambda name: name.lower() != 'background' and len(name) > 1,
        ClassSelection.CHARACTERS: lambda name: name.lower() != 'background' and len(name) == 1,
        ClassSelection.SHAPES_AND_CHARACTERS: lambda name: name.lower() != 'background'
    }

    def __init__(
            self,
            output_locations: OutputLocations,
            class_seleection: ClassSelection = ClassSelection.SHAPES_AND_CHARACTERS,
            tile_size: int = TILE_SIZE
    ):
        """
        NOTE: Class IDs will not not be the same between Isaac and output dataset because this can filter out classes.
        
        Parameters:
            output_locations: The locations to save the output tiles and labels
            class_selection: Which category of classes to include in the output dataset
            tile_size: The size of the tiles to create
        """
        self.output_locations = output_locations
        self.class_selection = class_seleection
        self.tile_size = tile_size

        # Includes all found classnames, including ones that are filtered out
        # so that they can be skipped efficiently
        self.isaac_classes: dict[int, str] = {}

        # Includes only classnames that are not filtered out
        self.output_classes: ClassnameMap = ClassnameMap()

    @staticmethod
    def get_file_id(path_to_file: Path):
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
            root_dir: Path | None = None
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
        overlap = int(((amount * section) - total_length) / (amount - 1))
        return amount, overlap

    @staticmethod
    def _subtile(
            image_arr: np.ndarray,
            tileSize: int
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
                y_start = i * (tileSize - height_overlap)
                y_end = y_start + tileSize

                x_start = j * (tileSize - width_overlap)
                x_end = x_start + tileSize

                tile = Image.fromarray(image_arr[y_start:y_end, x_start:x_end])

                yield Tile(
                    image=tile,
                    x_start=x_start,
                    y_start=y_start,
                    x_end=x_end,
                    y_end=y_end
                )

    @staticmethod
    def _isaac_to_yolo_bbox(tile: Tile, x1: int, y1: int, x2: int, y2: int) -> Bbox:
        _, x_start, y_start, x_end, y_end = tile

        tile_height = y_end - y_start
        tile_width = x_end - x_start

        def constrain(val, min_val, max_val):
            return min(max_val, max(min_val, val))

        # Constrain the values to be within the tile
        x1 = constrain(x1, x_start, x_end)
        y1 = constrain(y1, y_start, y_end)
        x2 = constrain(x2, x_start, x_end)
        y2 = constrain(y2, y_start, y_end)

        # Translate the values to be relative to the tile
        x1 -= x_start
        y1 -= y_start
        x2 -= x_start
        y2 -= y_end

        # Calculate the yolo format values
        y_center = ((y1 + y2) / 2) / tile_height
        x_center = ((x1 + x2) / 2) / tile_width
        box_height = (y2 - y1) / tile_height
        box_width = (x2 - x1) / tile_width

        return Bbox(y_center, x_center, box_height, box_width)

    def _get_subtile_data(self, image_path: Path) -> Generator[TileData, None, None]:
        """
        Yields the subtile and corresponding labels of the given image.
        
        If an unknown class is found, it will be added to the `found_classes` dictionary.
        
        If `DEBUG` is True, it will print the original label in parentheses in the label file,
        which will make it unusable.
        
        Returns:
            A generator of tuples containing the subtile id, the subtile image, and the label file content.
        """
        # Function determining whther to include a class in the dataset
        filter_func = YOLOFormatter.FILTER_FUNCTIONS[self.class_selection]

        # Get the image id and the labels file
        image_id = YOLOFormatter.get_file_id(image_path)
        label_pos = YOLOFormatter.get_file_by_id(image_id, "bbox_pos")  #npy type

        # Load the label position data
        label_pos_data = np.load(label_pos)

        # Get the image and convert it to an array
        source = np.array(Image.open(image_path))

        # Create a bbox label file for each tile
        # tile_pos_data: [tile_xmin, tile_ymin, tile_xmax, tile_ymax] per tile
        for tile_id, tile in enumerate(YOLOFormatter._subtile(source, self.tile_size)):
            subtile, tile_xmin, tile_ymin, tile_xmax, tile_ymax = tile
            # TODO: Decompose the following code into smaller functions

            # Create temp file
            temp_file = StringIO()

            # translate label position data to yolo format
            # yolo format: <object-class> <x_center> <y_center> <width> <height>
            # entry format: <object-class> <x1> <y1> <x2> <y2>

            # For each entry in the original label position data we need to check all the tiles to see if it is in the tile
            for entry in label_pos_data:
                isaac_id, x1, y1, x2, y2, rot = entry

                if isaac_id == 0: continue  # Skip background

                if isaac_id not in self.isaac_classes:
                    # Load the class label data
                    class_label = YOLOFormatter.get_file_by_id(image_id, 'bbox_legend')  #json type
                    with open(class_label) as class_label_file:
                        class_label_data = json.load(class_label_file)
                    self.isaac_classes[int(isaac_id)] = class_label_data[str(isaac_id)]["class"]

                classname = self.isaac_classes[int(isaac_id)]

                # Skip the class if it doesn't meet the filter criteria
                if not filter_func(classname):
                    continue

                if not classname in self.output_classes:
                    self.output_classes.add_class(classname)

                output_class_id = self.output_classes.get_class_id(classname)

                # tile_pos_data: [tile_xmin, tile_ymin, tile_xmax, tile_ymax] per tile
                # entry format: <object-class> <x1> <y1> <x2> <y2>
                # Check if at least one of the points are in the tile
                one_point_in_tile = (
                        (tile_xmin <= x1 <= tile_xmax and tile_ymin <= y1 <= tile_ymax) or
                        (tile_xmin <= x2 <= tile_xmax and tile_ymin <= y2 <= tile_ymax)
                )

                if not one_point_in_tile:
                    continue

                if DEBUG:
                    temp_file.write(f"({output_class_id} {x1} {y1} {x2} {y2})\n")

                x_center, y_center, box_height, box_width = self._isaac_to_yolo_bbox(
                    tile, x1, y1, x2, y2
                )

                # Write the yolo format values to the temp file
                temp_file.write(f'{output_class_id} {x_center} {y_center} {box_width} {box_height}\n')

            yield TileData(f"{image_id}_{tile_id}", subtile, temp_file.getvalue())

    @staticmethod
    def _write_tiles_and_labels(tile_data_iterable: Iterable[TileData], tiles_dir: Path, labels_dir: Path):
        def _write_tile(id: str, tile: Image.Image):
            tile.save(tiles_dir / f'{id}.png', compress_level=3)

        def _write_label(id: str, label: str):
            with open(labels_dir / f'{id}.txt', 'w') as f:
                f.write(label)

        pool = Pool(6)  # 6 threads seems to work best

        for id, tile, labels in tile_data_iterable:
            pool.apply_async(_write_tile, args=(id, tile))
            pool.apply_async(_write_label, args=(id, labels))

        pool.close()
        pool.join()

    def _create_tiles(
            self,
            num_train: int,
            num_valid: int,
            source_paths: list[Path]
    ):
        base_dir, train_img, train_label, valid_img, valid_label, test_img, test_label = self.output_locations

        # Copy the images to the target directory
        print('Dataset location:', base_dir)

        for i, image in enumerate(
                tqdm(
                    random.choices(source_paths, k=len(source_paths)),  # Randomize the order of the images
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
                tiles_dir = train_img
                label_dir = train_label
            elif i < num_train + num_valid:
                tiles_dir = valid_img
                label_dir = valid_label
            else:
                tiles_dir = test_img
                label_dir = test_label

            if DEBUG: start_time = time.time()
            YOLOFormatter._write_tiles_and_labels(self._get_subtile_data(image), tiles_dir, label_dir)
            if DEBUG: print(f"time to write tiles: {time.time() - start_time}")

        print('Done generating data!')

    def _create_data_yaml_file(self):
        # Creates the data.yaml file

        number_of_classes = len(self.output_classes)
        print(f'Found {number_of_classes} classes')

        classnames = list(self.output_classes.classnames())

        data_yaml = [
            "train: ../train/images\n",
            "val: ../valid/images\n",
            "test: ../test/images\n\n",
            f"nc: {number_of_classes}\n",
            f"names: {classnames}\n"
        ]
        with open(self.output_locations.base_dir / 'data.yaml', 'w') as f:
            f.writelines(data_yaml)

    def create_dataset(
            self,
            num_train: int,
            num_valid: int,
            source_paths: list[Path],
    ):
        """
        Creates the YOLO dataset from the list of source RGB image paths, dividing them into tiles.
        
        Finds the associated label files in the same directory from the image name.
        
        Splits the dataset into training, validation, and testing sets, per YOLO convention.
        The number of test images is the remainder after the training and validation images are selected.
        
        Parameters:
            num_train: The number of images to use for training
            num_valid: The number of images to use for validation
            source_paths: The list of paths to the source RGB images
        """
        print('Generating images and generating label files...')
        self._create_tiles(num_train, num_valid, source_paths)

        print("Now making data.yaml file...")
        self._create_data_yaml_file()


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
    create_yolo_dataset(ClassSelection.SHAPES)
