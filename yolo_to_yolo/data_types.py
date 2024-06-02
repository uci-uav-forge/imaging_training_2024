from typing import NamedTuple
from enum import Enum

import numpy as np

from .yolo_io_types import Task


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

class YoloClassType(Enum):
    SHAPE = 1
    CHARACTER = 2
    COLOR = 3
    UNKNOWN = 4

class YoloLabel(NamedTuple):
    location: YoloOutline | YoloBbox
    classname: str

    @property
    def class_type(self) -> YoloClassType:
        if self.classname in ("circle", "semicircle", "quarter circle", "quartercircle", "quarter_circle", "triangle", "rectangle", "pentagon", "star", "cross"):
            return YoloClassType.SHAPE
        elif len(self.classname) == 1:
            return YoloClassType.CHARACTER
        elif self.classname in ("red", "green", "blue", "orange", "purple", "white", "black", "brown"):
            return YoloClassType.COLOR
        else:
            return YoloClassType.UNKNOWN


class YoloImageData(NamedTuple):
    """
    One image/tile's annotations for YOLO training. Can be for detection or segmentation.
    """
    img_id: str
    task: Task
    image: np.ndarray
    labels: list[YoloLabel]
