from typing import NamedTuple

import numpy as np
from pathlib import Path
from typing import Generator

from .yolo_io_types import Task, PredictionTask, YoloSubsetDirs, DatasetDescriptor


class Point(NamedTuple):
    x: float
    y: float


class YoloBbox(NamedTuple):
    x: float
    y: float
    w: float
    h: float


class YoloOutline(NamedTuple):
    points: list[Point]


class TargetAnnotation(NamedTuple):
    location: YoloOutline | YoloBbox
    classname: str


class YoloLabel(NamedTuple):
    location: YoloOutline | YoloBbox
    classname: str


class YoloImageData(NamedTuple):
    """
    One image/tile's annotations for YOLO training. Can be for detection or segmentation.
    """
    img_id: str
    task: Task
    image: np.ndarray
    labels: list[YoloLabel]

class GenericYoloReader:
    def __init__(
        self,
        yaml_path: Path,
        prediction_task: PredictionTask,
    ) -> None:
        self.prediction_task = prediction_task

        self.yaml_path = yaml_path

        self.descriptor = DatasetDescriptor.from_yaml(self.yaml_path)
        self.descriptor.check_dirs_exist()

    @property
    def parent_dir(self) -> Path:
        return self.descriptor.parent_dir

    @property
    def train_dirs(self) -> YoloSubsetDirs:
        return self.descriptor.train_dirs

    @property
    def val_dirs(self) -> YoloSubsetDirs:
        return self.descriptor.val_dirs

    @property
    def test_dirs(self) -> YoloSubsetDirs:
        return self.descriptor.test_dirs

    @property
    def classes(self) -> tuple[str, ...]:
        return self.descriptor.classes

    def read(
        self,
        tasks: tuple[Task, ...] = (Task.TRAIN, Task.VAL, Task.TEST),
        img_file_pattern: str = "*.png"
    ) -> Generator[YoloImageData, None, None]:
      raise NotImplementedError()
