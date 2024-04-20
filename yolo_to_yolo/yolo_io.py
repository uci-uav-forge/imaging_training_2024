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
        num_workers: int = int(multiprocessing.cpu_count()) - 1
    ) -> None:
        self.prediction_task = prediction_task
        self.num_workers = num_workers

        self.yaml_path = yaml_path

        self.descriptor = DatasetDescriptor.from_yaml(self.yaml_path)
        self.descriptor.check_dirs_exist()

    @property
    def parent_dir(self) -> Path:
        return self.descriptor.parent_dir

    @property
    def train_dirs(self) -> YoloSubsetDirs:
        return self.descriptor.train_dirs

    @property
    def val_dirs(self) -> YoloSubsetDirs:
        return self.descriptor.val_dirs

    @property
    def test_dirs(self) -> YoloSubsetDirs:
        return self.descriptor.test_dirs

    @property
    def classes(self) -> tuple[str, ...]:
        return self.descriptor.classes

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
            images_dir, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
            paths: Iterable[Path] = images_dir.glob(img_file_pattern)

            output: Iterable[tuple[YoloImageData, Task]] = pool.imap_unordered(
                self._worker_task,
                zip(paths, repeat(task)),
                chunksize=8
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
        _, labels_dir = self.descriptor.get_image_and_labels_dirs(task)
        labels = list(self._get_labels_from_id(img_id, labels_dir))

        data_obj = YoloImageData(img_id, image, labels)

        return data_obj, task

    def _get_labels_from_id(self, img_id: str, labels_dir: Path) -> Iterable[YoloLabel]:
        labels_path = labels_dir / f'{img_id}.txt'

        with open(labels_path, 'r') as f:
            for line in f.readlines():
                yield self._parse_label_line(line)

    @staticmethod
    def _get_id_from_filename(filename: Path) -> str:
        return filename.stem

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


class YoloWriter:
    def __init__(
        self,
        out_dir: Path,
        prediction_task: PredictionTask,
        classes: tuple[str, ...],
        num_workers: int = int(multiprocessing.cpu_count()) - 1
    ) -> None:
        self.out_dir = out_dir
        self.prediction_task = prediction_task
        self.num_workers = num_workers

        self.descriptor = DatasetDescriptor.from_parent_dir(self.out_dir, classes)
        self.descriptor.create_dirs()

    def write_data(
        self,
        data: Iterable[tuple[YoloImageData, Task]]
    ) -> None:
        pool: multiprocessing.Pool = multiprocessing.Pool(self.num_workers)
        pool.imap_unordered(self._worker_task, data, chunksize=8)

        pool.close()
        pool.join()

        # Write it after everything's done as an indicator that the dataset is complete.
        self._write_dataset_yaml()

    def _worker_task(self, data_and_task: tuple[YoloImageData, Task]) -> None:
        data, task = data_and_task
        img_id, image, labels = data

        images_dir, labels_dir = self.descriptor.get_image_and_labels_dirs(task)

        img_path = images_dir / f'{img_id}.png'
        labels_path = labels_dir / f'{img_id}.txt'

        Image.fromarray(image).save(img_path)

        with open(labels_path, 'w') as f:
            for label in labels:
                f.write(self._format_label(label))
                f.write('\n')

    @staticmethod
    def _format_label(label: YoloLabel) -> str:
        if isinstance(label.location, YoloBbox):
            return f"{label.classname} {label.location.x} {label.location.y} {label.location.w} {label.location.h}"

        if isinstance(label.location, YoloOutline):
            return f"{label.classname} {' '.join(f'{point.x} {point.y}' for point in label.location.points)}"

        raise ValueError(f"Unknown location annotation type: {label.location}")

    def _write_dataset_yaml(self):
        yaml_path = self.out_dir / "data.yaml"

        contents: list[str] = [
            f"path: {self.out_dir}",
            "train: ../train/images",
            "val: ../valid/images",
            "test: ../test/images",
            "",
            f"nc: {len(self.descriptor.classes)}",
            f"names: {list(self.descriptor.classes)}",
        ]

        with yaml_path.open('w') as f:
            f.write("\n".join(contents))


def batched(iterable: Iterable, n: int) -> Iterable[tuple]:
    """
    Implementation of `itertools.batched` because I couldn't import it, despite using python 3.11.
    """
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

    writer = YoloWriter(
        Path('/home/minh/Desktop/imaging_training_2024/data/YOLO_DATASET/DATASETv1_processed/'),
        PredictionTask.DETECTION,
        reader.classes
    )

    writer.write_data(tqdm(reader.get_data(), desc="Processing data"))