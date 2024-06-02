from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from yolo_io import YoloReader, YoloWriter
from yolo_io_types import PredictionTask, Task

from data_types import YoloBbox, YoloLabel, YoloImageData
from data_transformers import BBoxToCropTransformer
import shutil

import cv2

def main():
    print("wip")
    return
    transformer = BBoxToCropTransformer(min_size=(50, 50), min_padding=10, min_char_overlap=0)

    input_yaml = Path("data/godot_processed/small_100/data.yaml")
    if not input_yaml.exists():
        print("yaml file doesn't exist at specified path")
        return

    output_dir = Path("data/yolo_transformer_test_output")
    # if output dir exists delete it
    if output_dir.exists():
        shutil.rmtree(output_dir)
    # create output dir
    output_dir.mkdir()

    # Create reader and writer instances
    reader = YoloReader(yaml_path=input_yaml, prediction_task=PredictionTask.DETECTION)
    writer = YoloWriter(out_dir=output_dir, prediction_task=PredictionTask.DETECTION, classes=reader.classes)

    tasks_to_process = (Task.TEST, Task.TRAIN, Task.VAL)
    for task in tasks_to_process:
        print(f"Processing {task.value} data")
        image_data = reader.read(tasks=(task,))
        for yolo_image_data in image_data:
            # print(yolo_image_data)
            print("TRANSFORMING DATA")
            transformed = list(transformer(yolo_image_data))
            print("deez")
            print(transformed)
            for i, data in enumerate(transformed):
                # print("==========")
                # print(data)
                #writer.write([data])
                pass
            print("Stopping early because transformer is being cringe")
            return

    print("Done!")

if __name__ == "__main__":
    main()
