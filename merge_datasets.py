from itertools import chain
from pathlib import Path
from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.yolo_io import YoloWriter, YoloReader, YoloLabel
from yolo_to_yolo.yolo_io_types import PredictionTask
from tqdm import tqdm
if __name__ == "__main__":
    reader_1 = YoloReader(
        Path('/home/forge/uavf_2024/imaging_training_2024/yolo_ds/2024-shapes-det.yaml'),
        PredictionTask.DETECTION,
    )
    reader_2 = YoloReader(
        Path('/home/forge/uavf_2024/imaging_training_2024/yolo_ds/2024-shapes-det.yaml'),
        PredictionTask.DETECTION,
    )
    writer = YoloWriter(
        Path('/home/forge/uavf_2024/imaging_training_2024/yolo_2'),
        PredictionTask.DETECTION,
        tuple(reader_1.classes)
    )
    reader_1_class_dict = reader_1.classes
    reader_2_class_dict = reader_2.classes

    def find_class_index(old_class_name, new_class_name, value):
        class_name = old_class_name[value]
        for key, val in new_class_name.items():
            if class_name == val:
                return key
    
    changed_data_1 = []
    for original_data in reader_1.read():
        new_yolo_label_locations = []
        for label in original_data.labels:
            new_yolo_label_locations.append(YoloLabel(label.location, find_class_index(reader_1_class_dict, reader_2_class_dict, label.classname)))
        changed_data_1.append((YoloImageData(
            "1_" + original_data.img_id, original_data.task, original_data.image, new_yolo_label_locations)))
    
    changed_data_2 = (
    (YoloImageData(
        "2_" + original_data.img_id, original_data.task, original_data.image, original_data.labels
    )) for original_data in reader_1.read())


    writer.write(tqdm(chain(changed_data_1, changed_data_2), desc="Processing data"))
