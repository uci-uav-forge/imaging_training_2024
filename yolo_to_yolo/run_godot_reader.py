from yolo_to_yolo.godot_reader import GodotReader
from yolo_to_yolo.yolo_io import YoloWriter
from yolo_to_yolo.yolo_io_types import PredictionTask
from yolo_to_yolo.data_types import YoloImageData
from pathlib import Path
from tqdm import tqdm

# run me with py -m yolo_to_yolo.run_godot_reader
if __name__ == "__main__":
    dataset_id = '4000'
    in_path = f'/datasets/godot_raw/godot_data_{dataset_id}'
    out_path = f'/datasets/godot_processed/{dataset_id}_all_labels'

    reader = GodotReader(
        Path(in_path),
    )

    shape_classnames = [
        "circle",
        "semicircle",
        "quartercircle",
        "triangle",
        "rectangle",
        "pentagon",
        "star",
        "cross",
        "person",
        *"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "shape:white",
        "shape:black",
        "shape:red",
        "shape:blue",
        "shape:green",
        "shape:purple",
        "shape:brown",
        "shape:orange",
        "char:white",
        "char:black",
        "char:red",
        "char:blue",
        "char:green",
        "char:purple",
        "char:brown",
        "char:orange"
    ]

    writer = YoloWriter(
        Path(out_path),
        PredictionTask.DETECTION,
        shape_classnames
    )

    writer.write(tqdm(reader.read()))
    
