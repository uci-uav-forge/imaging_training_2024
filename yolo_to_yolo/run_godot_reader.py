from yolo_to_yolo.godot_reader import GodotMultiLabelReader
from yolo_to_yolo.yolo_io_types import PredictionTask
from pathlib import Path
import cv2 as cv
import os

# run me with `py -m yolo_to_yolo.run_godot_reader.py`
if __name__ == "__main__":
    yaml_path = '/home/forge/eric/uavf_2024/imaging_training_2024/godot_data_utils/yolo/2024-shapes-all-labels.yaml'

    out_dir = 'cls_dataset'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/labels", exist_ok=True)

    reader = GodotMultiLabelReader(
        Path(yaml_path),
    )

    i= 0
    for data in reader.read():
        os.makedirs(f"{out_dir}/images/{data.task}", exist_ok=True)
        os.makedirs(f"{out_dir}/labels/{data.task}", exist_ok=True)
        cv.imwrite(f"{out_dir}/images/{data.task}/{data.img_id}.png", data.image)
        with open(f"{out_dir}/labels/{data.task}/{data.img_id}.txt", "w+") as f:
            f.write(" ".join(label.classname for label in data.labels))
        i+=1
        if i>100:
            break

