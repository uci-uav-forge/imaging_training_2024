from functools import reduce
from pathlib import Path
from typing import Iterable
from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.yolo_io import YoloWriter, YoloReader
from yolo_to_yolo.yolo_io_types import PredictionTask
from tqdm import tqdm

# Functions in this module are broken down so we can reuse them selectively.


def merge_readers(readers: Iterable[YoloReader]) -> Iterable[YoloImageData]:
    """
    Sequentially yield images from multiple readers, adding a prefix to `img_id`.
    """
    for prefix, reader in enumerate(readers):
        for data in reader.read():
            yield YoloImageData(
                img_id=f"{prefix}_{data.img_id}",
                task=data.task,
                image=data.image,
                labels=data.labels
            )


def merge_reader_classes(readers: Iterable[YoloReader]) -> set[str]:
    """
    Get all unique classes from all readers.
    
    Since a set is used, the order of classes is not preserved.
    """
    def reducer(acc: set[str], reader: YoloReader) -> set[str]:
        return acc.union(set(reader.classes))
    
    return reduce(reducer, readers, set())


def merge_datasets(
    readers: Iterable[YoloReader],
    output_dir: Path, 
    prediction_task: PredictionTask = PredictionTask.DETECTION
) -> None:
    """
    Merge YOLO-formatted datasets from readers.
    Readers are injected as a dependency to enable the use of different readers.
    
    Does not preserve the original order of classes.
    """    
    classes = merge_reader_classes(readers)
    
    writer = YoloWriter(output_dir, prediction_task, classes)
    
    writer.write(
        tqdm(
            merge_readers(readers),
            desc="Processing data",
            unit="images"
        )
    )


if __name__ == "__main__":
    dataset_paths = [
        Path('/home/minh/Desktop/uavf_2024/imaging_training_2024/data/godot_4000_all_labels/data.yaml'),
        Path('/home/minh/Desktop/uavf_2024/imaging_training_2024/data/isaac_yolo_7671_all/data.yaml'),
    ]
    
    datasets = map(lambda path: YoloReader(path, PredictionTask.DETECTION), dataset_paths)
    
    output_dir = Path('/home/minh/Desktop/uavf_2024/imaging_training_2024/data/isaac_godot_merged/')
    
    merge_datasets(datasets, output_dir)
