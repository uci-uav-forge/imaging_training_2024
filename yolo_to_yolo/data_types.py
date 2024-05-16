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

