from pathlib import Path
from typing import Iterable, Generator

import numpy as np
from PIL import Image

from .data_types import YoloImageData, YoloLabel, YoloBbox, GenericYoloReader, DatasetDescriptor
from .yolo_io_types import Task, PredictionTask


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
        id = 0
        for task in tasks:
            images_dir, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
            images_dir = Path(str(images_dir).replace(task.value,"$TASK").replace("images", task.value).replace("$TASK","images"))
            print(images_dir)
            image_paths: Iterable[Path] = images_dir.glob(img_file_pattern)


            for img_path in image_paths:
                image = np.array(Image.open(img_path))
                img_h, img_w = image.shape[:2]
                img_id = self._get_id_from_filename(img_path)
                _, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
                labels_path = labels_dir / f'{img_id}.txt'
                labels_path = str(labels_path).replace(task.value,"$TASK").replace("labels", task.value).replace("$TASK","labels")
                with open(labels_path, 'r') as f:
                    for line in f.readlines():
                        shape_name, char_name, shape_col, letter_col, x, y, w, h = line.split()
                        x, y, w, h = map(float, (x, y, w, h))
                        x_pix, y_pix, w_pix, h_pix = map(int, (x*img_w, y*img_h, w*img_w, h*img_h))
                        crop_img = image[y_pix - h_pix//2:y_pix + h_pix//2,x_pix - w_pix//2:x_pix + w_pix//2]

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
    @staticmethod
    def _get_id_from_filename(filename: Path) -> str:
        return filename.stem
