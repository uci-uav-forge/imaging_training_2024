from yolo_to_yolo.godot_reader import GodotReader
from yolo_to_yolo.yolo_io import YoloWriter
from yolo_to_yolo.yolo_io_types import PredictionTask
from yolo_to_yolo.data_types import YoloImageData
from pathlib import Path
import cv2 as cv
import os

# run me with `py -m yolo_to_yolo.run_godot_reader.py`
if __name__ == "__main__":
    dataset_id = '1716327957'
    in_path = f'/datasets/godot_raw/godot_data_{dataset_id}'
    out_path = f'/datasets/godot_processed/{dataset_id}'

    os.makedirs(out_path, exist_ok=True)

    reader = GodotReader(
        Path(in_path),
    )

    shape_classnames = [
        "circle",
        "semicircle",
        "quarter circle",
        "triangle",
        "rectangle",
        "pentagon",
        "star",
        "cross",
        "person"
    ]

    only_shape_boxes = map(
        lambda box: YoloImageData(
            box.img_id,
            box.task,
            box.image,
            [l for l in box.labels if l.classname in shape_classnames]
        ),
        reader.read()
    )

    writer = YoloWriter(
        Path(out_path),
        PredictionTask.DETECTION,
        shape_classnames,
    )

    writer.write(reader.read())
    
