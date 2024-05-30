from pathlib import Path
from typing import Iterable, Generator

import numpy as np
from PIL import Image

from yolo_to_yolo.generic_reader import GenericYoloReader

from .data_types import YoloImageData, YoloLabel, YoloBbox
from .yolo_io_types import DatasetDescriptor, Task, PredictionTask
from .godot_utils import get_polygon, give_normalized_bounding_box
import os


class GodotReader:
    """
    Reader for YOLO training data.

    Outputs a box for each different label (so 4 boxes per target). Needs to be pipelined
    into a data transformer that will group the boxes and filter labels to be used in training.

    Example:
        reader = GodotReader("/datasets/godot_raw/godot_data_0", PredictionTask.DETECTION)
        writer = YoloWriter("/datasets/godot_processed/0", PredictionTask.DETECTION)
        writer.write(reader.read())
    """
    def __init__(
        self,
        dataset_folder_path: Path,
        split_proportions: tuple[float,float,float] = (0.7, 0.2, 0.1)
    ) -> None:
        self.dataset_folder_path = dataset_folder_path
        self.split_proportions = split_proportions

    def read(
        self,
    ) -> Generator[YoloImageData, None, None]:
        num_imgs = os.listdir(self.dataset_folder_path / "images")
        for i in range(len(num_imgs)):
            progress = i / len(num_imgs)
            img_path = self.dataset_folder_path / "images" / f"image{i}.png"
            masks_path = self.dataset_folder_path / "masks" / f"{i}"
            if progress < self.split_proportions[0]:
                task = Task.TRAIN
            elif progress < self.split_proportions[0] + self.split_proportions[1]:
                task = Task.VAL
            else:
                task = Task.TEST
            yield self._process_img_path(img_path, masks_path, task, i)
    
    def _process_img_path(self, img_path: Path, masks_path: Path, task: Task, id: int) -> YoloImageData:
        image = np.array(Image.open(img_path))
        data_labels = []
        for mask_fname in os.listdir(masks_path):
            # file names will be like shape_name,letter_name,shape_col,letter_col_index.png
            mask_path = masks_path / mask_fname
            mask = np.array(Image.open(mask_path))
            polygon = get_polygon(mask)
            if len(polygon) == 0:
                continue
            normalized_polygon = polygon / np.array([mask.shape[1], mask.shape[0]])
            bbox = give_normalized_bounding_box(normalized_polygon)
            labels, index = mask_fname.split("_")
            if labels == 'person':
                data_labels.append(
                    YoloLabel(
                        location=bbox,
                        classname=labels
                    )
                )
                continue
            shape_name, letter_name, shape_col, letter_col = labels.split(",")
            data_labels.extend([
                    YoloLabel(
                        location=bbox,
                        classname=shape_name
                    ),
                    YoloLabel(
                        location=bbox,
                        classname=letter_name
                    ),
                    YoloLabel(
                        location=bbox,
                        classname=f"shape:{shape_col}"
                    ),
                    YoloLabel(
                        location=bbox,
                        classname=f"char:{letter_col}"
                    )
                ] 
            )
        return YoloImageData(
            img_id=str(id),
            task=task,
            image=image,
            labels=data_labels
        )



    @staticmethod
    def _get_id_from_filename(filename: Path) -> str:
        return filename.stem
