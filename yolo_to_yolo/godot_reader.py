from itertools import repeat
from pathlib import Path
from typing import Iterable, Generator
import multiprocessing

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_to_yolo.data_types import YoloImageData, YoloLabel, YoloBbox, Point, YoloOutline
from yolo_to_yolo.yolo_io_types import PredictionTask, DatasetDescriptor, YoloSubsetDirs, Task, ClassnameMap, GenericYoloReader
from data_transformers import BBoxToCropTransformer


class GodotMultiLabelReader(GenericYoloReader):
    """
    Reader for YOLO training data.

    Example:
        reader = YoloReader("YOLO_DATASET/data.yaml")
        for yolo_image_data, task in reader:
            ...
    """
    def read(
        self,
        tasks: tuple[Task, ...] = (Task.TRAIN, Task.VAL, Task.TEST),
        img_file_pattern: str = "*.png"
    ) -> Generator[YoloImageData, None, None]:
        tiling_transformer = BBoxToCropTransformer()
        id = 0
        for task in tasks:
            images_dir, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
            image_paths: Iterable[Path] = images_dir.glob(img_file_pattern)
            for img_path in image_paths:
                image = np.array(Image.open(img_path))
                img_h, img_w = image.shape[:2]
                img_id = self._get_id_from_filename(img_path)
                _, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
                labels_path = labels_dir / f'{img_id}.txt'
                with open(labels_path, 'r') as f:
                    for line in f.readlines():
                        shape_name, char_name, shape_col, letter_col, x, y, w, h = line.split()
                        x, y, w, h = map(float, (x, y, w, h))
                        x_pix, y_pix, w_pix, h_pix = map(int, (x*img_w, y*img_h, w*img_w, h*img_h))
                        crop_img = image[x_pix - w_pix//2:x_pix + w_pix//2, y_pix - h_pix//2:y_pix + h_pix//2]

                        img_data = YoloImageData(
                            img_id = str(id),
                            task = task,
                            image = crop_img,
                            labels = [
                                YoloLabel(
                                    location = YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0),
                                    classname = shape_name
                                ),
                                YoloLabel(
                                    location = YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0),
                                    classname = char_name
                                ),
                                YoloLabel(
                                    location = YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0),
                                    classname = shape_col
                                ),
                                YoloLabel(
                                    location = YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0),
                                    classname = letter_col
                                )
                            ]
                        )
                        id+=1
                        yield img_data


