from itertools import chain
from pathlib import Path
from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.yolo_io import YoloWriter, YoloReader, YoloLabel
from yolo_to_yolo.yolo_io_types import PredictionTask
from tqdm import tqdm


if __name__ == "__main__":
    reader_1 = YoloReader(
        Path('/home/forge/uavf_2024/imaging_training_2024/godot_ds/2024-shapes-det.yaml'),
        PredictionTask.DETECTION,
    )
    # reader_1 = YoloReader(
    #     Path('/datasets/godot_processed/small_100/data.yaml'),
    #     PredictionTask.DETECTION,
    # )
    reader_2 = YoloReader(
        Path('/datasets/godot_processed/small_100/data.yaml'),
        PredictionTask.DETECTION,
    )
    # reader_2 = YoloReader(
    #     Path('/home/forge/uavf_2024/imaging_training_2024/godot_ds/2024-shapes-det.yaml'),
    #     PredictionTask.DETECTION,
    # )
    writer = YoloWriter(
        Path('/home/forge/uavf_2024/imaging_training_2024/yolo_2'),
        PredictionTask.DETECTION,
        reader_1.classes
    )
    reader_1_class_dict = reader_1.classes
    reader_2_class_dict = reader_2.classes
    
    changed_data_1 = []
    i = 0
    for original_data in reader_1.read():
        i+=1
        new_yolo_label_locations = []
        for label in original_data.labels:
            new_yolo_label_locations.append(YoloLabel(label.location, label.classname))
        changed_data_1.append((YoloImageData(
            "1_" + original_data.img_id, original_data.task, original_data.image, new_yolo_label_locations)))

    changed_data_2 = (
    (YoloImageData(
        "2_" + original_data.img_id, original_data.task, original_data.image, original_data.labels
    )) for original_data in reader_2.read())


    writer.write(tqdm(chain(changed_data_1, changed_data_2), desc="Processing data"))
