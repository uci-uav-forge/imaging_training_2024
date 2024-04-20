from typing import NamedTuple

import numpy as np


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
    image: np.ndarray
    labels: list[YoloLabel]
