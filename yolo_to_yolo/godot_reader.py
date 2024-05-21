from pathlib import Path
from typing import Iterable, Generator

import numpy as np
from PIL import Image

from yolo_to_yolo.generic_reader import GenericYoloReader

from .data_types import YoloImageData, YoloLabel, YoloBbox
from .yolo_io_types import DatasetDescriptor, Task, PredictionTask


class GodotMultiLabelReader(GenericYoloReader):
    """
    Reader for YOLO training data.

    Example:
        reader = YoloReader("YOLO_DATASET/data.yaml")
        for yolo_image_data, task in reader:
            ...
    """
    def __init__(
        self,
        yaml_path: Path,
    ) -> None:
        self.prediction_task = PredictionTask.CLASSIFICATION

        self.yaml_path = yaml_path

        self.descriptor = DatasetDescriptor.from_yaml(self.yaml_path)
        # same as superclass constructor but without path checking because we need to swap order
        # of task and directory (e.g. train/images -> images/train)

    def read(
        self,
        tasks: tuple[Task, ...] = (Task.TRAIN, Task.VAL, Task.TEST),
        img_file_pattern: str = "*.png"
    ) -> Generator[YoloImageData, None, None]:
        for task in tasks:
            images_dir, _ = self.descriptor.get_image_and_labels_dirs(task)
            images_dir = Path(str(images_dir).replace(task.value,"$TASK").replace("images", task.value).replace("$TASK","images"))
            image_paths: Iterable[Path] = images_dir.glob(img_file_pattern)
            for img_path in image_paths:
                yield self._process_img_path(img_path, task)
    
    def _process_img_path(self, img_path: Path, task) -> YoloImageData:
            image = np.array(Image.open(img_path))
            img_id = self._get_id_from_filename(img_path)
            _, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
            labels_path = labels_dir / f'{img_id}.txt'
            labels_path = str(labels_path).replace(task.value,"$TASK").replace("labels", task.value).replace("$TASK","labels")
            labels: list[YoloLabel] = []
            with open(labels_path, 'r') as f:
                for line in f.readlines():
                    shape_name, char_name, shape_col, letter_col, x, y, w, h = line.split()
                    x, y, w, h = map(float, (x, y, w, h))
                    for class_name in [shape_name, char_name, shape_col, letter_col]:
                        labels.append(YoloLabel(
                            location = YoloBbox(x=x, y=y, w=w, h=h),
                            classname = class_name
                        ))
            img_data = YoloImageData(
                img_id = img_id,
                task = task,
                image = image,
                labels = labels
            )
            return img_data

    @staticmethod
    def _get_id_from_filename(filename: Path) -> str:
        return filename.stem
