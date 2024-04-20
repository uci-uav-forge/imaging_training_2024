from itertools import repeat
from pathlib import Path
from typing import Iterable, Generator
import multiprocessing

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo_to_yolo.data_types import YoloImageData, YoloLabel, YoloBbox, Point, YoloOutline
from yolo_to_yolo.yolo_io_types import PredictionTask, DatasetDescriptor, YoloSubsetDirs, Task


class YoloReader:
    """
    Reader for YOLO training data.

    Example:
        reader = YoloReader("YOLO_DATASET/data.yaml")
        for yolo_image_data, task in reader:
            ...
    """
    def __init__(
        self,
        yaml_path: Path,
        prediction_task: PredictionTask,
        num_workers: int = multiprocessing.cpu_count()
    ) -> None:
        self.prediction_task = prediction_task
        self.num_workers = num_workers

        self.yaml_path = yaml_path
        self.parent_dir, self.train_dirs, self.val_dirs, self.test_dirs, self.classes = \
            DatasetDescriptor.from_yaml(self.yaml_path)

        self._check_paths()

    def _check_paths(self):
        """
        Check whether the stored directories exist. Throws if something is missing.

        Must be called after initialization.
        """
        if not Path(self.parent_dir).is_dir():
            raise NotADirectoryError()

        for task_dir in (self.train_dirs, self.val_dirs, self.test_dirs):
            for subdir in task_dir:
                if not subdir.is_dir():
                    raise NotADirectoryError(f"{subdir} is not a directory")

    def get_data(
        self,
        tasks: tuple[Task, ...] = (Task.TRAIN, Task.VAL, Task.TEST),
        img_file_pattern: str = "*.png"
    ) -> Generator[tuple[YoloImageData, Task], None, None]:
        """
        Read the dataset with concurrency. Yields tuples of `(YoloImageData, Task)`.
        """
        pool: multiprocessing.Pool = multiprocessing.Pool(self.num_workers)
        outputs: list[Iterable[tuple[YoloImageData, Task]]] = []

        for task in tasks:
            images_dir, labels_dir = self._get_image_and_labels_dirs(task)
            paths: Iterable[Path] = images_dir.glob(img_file_pattern)

            output: Iterable[tuple[YoloImageData, Task]] = pool.imap_unordered(
                self._worker_task,
                zip(paths, repeat(task)),
                chunksize=4
            )

            outputs.append(output)

        pool.close()

        for output_iterable in outputs:
            yield from output_iterable

        pool.join()

    def _worker_task(self, path_and_task: tuple[Path, Task]) -> tuple[YoloImageData, Task]:
        """
        Worker task for reading image and labels files.

        Takes a tuple, so it can be used in `imap_unordered`.
        """
        img_path, task = path_and_task

        image = np.array(Image.open(img_path))

        img_id = self._get_id_from_filename(img_path)
        _, labels_dir = self._get_image_and_labels_dirs(task)
        labels = list(self._get_labels_from_id(img_id, labels_dir))

        data_obj = YoloImageData(image, labels)

        return data_obj, task

    @staticmethod
    def _get_id_from_filename(filename: Path) -> str:
        return filename.stem

    def _get_labels_from_id(self, img_id: str, labels_dir: Path) -> Iterable[YoloLabel]:
        labels_path = labels_dir / f'{img_id}.txt'

        with open(labels_path, 'r') as f:
            for line in f.readlines():
                yield self._parse_label_line(line)

    def _parse_label_line(self, label_line: str) -> YoloLabel:
        """
        Parse one line of YOLO's labels file, e.g., '0 0.1'
        """
        split = label_line.strip().split()

        if self.prediction_task == PredictionTask.DETECTION and len(split) != 5:
            raise ValueError(f"Label line for detection should have 5 fields, got '{label_line}'")

        if self.prediction_task == PredictionTask.SEGMENTATION and (len(split) - 1) % 2:
            raise ValueError(f"Got odd number of points in label line: {label_line}")

        classname = self.classes[int(split[0])]

        if self.prediction_task == PredictionTask.DETECTION:
            location_data = YoloBbox(*map(float, split[1:]))
        elif self.prediction_task == PredictionTask.SEGMENTATION:
            location_data = YoloOutline([Point(float(x), float(y)) for x, y in batched(split[1:], 2)])
        else:
            raise NotImplementedError(
                f"Only DETECTION and SEGMENTATION prediction tasks are supported, not {self.prediction_task}"
            )

        return YoloLabel(location_data, classname)

    def _get_image_and_labels_dirs(self, task: Task) -> YoloSubsetDirs:
        match task:
            case Task.TRAIN:
                return self.train_dirs
            case Task.VAL:
                return self.val_dirs
            case Task.TEST:
                return self.test_dirs
            case _:
                raise ValueError(f"Task {task} is invalid")


class YoloWriter:
    ...


def batched(iterable: Iterable, n: int) -> Iterable[tuple]:
    if n < 1:
        raise ValueError("n must be greater than or equal to 1")

    current = []
    for count, item in enumerate(iterable):
        current.append(item)
        if (count + 1) % n == 0:
            yield tuple(current)
            current = []

    if current:
        yield tuple(current)


if __name__ == "__main__":
    reader = YoloReader(
        Path('/home/minh/Desktop/imaging_training_2024/data/YOLO_DATASET/DATASETv1/data.yaml'),
        PredictionTask.DETECTION,
    )

    for i, (data, task) in enumerate(tqdm(reader.get_data())):
        if i % 100 == 0:
            print(i, data.labels, task)


