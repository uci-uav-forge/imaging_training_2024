from pathlib import Path
from typing import NamedTuple, Any, Iterable
from enum import Enum

import yaml


class PredictionTask(Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class Task(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class YoloSubsetDirs(NamedTuple):
    images: Path
    labels: Path

    @staticmethod
    def from_task_dir(task_dir: Path) -> 'YoloSubsetDirs':
        if task_dir.stem == "images":
            images = task_dir
            labels = task_dir.parent / "labels"
        else:
            images = task_dir / "images"
            labels = task_dir / "labels"

        return YoloSubsetDirs(images, labels)


class DatasetDescriptor(NamedTuple):
    parent_dir: Path
    train_dirs: YoloSubsetDirs
    val_dirs: YoloSubsetDirs
    test_dirs: YoloSubsetDirs
    classes: tuple[str, ...]

    def check_dirs_exist(self):
        if not Path(self.parent_dir).is_dir():
            raise NotADirectoryError(f"{self.parent_dir} is not a directory")

        for task_dir in (self.train_dirs, self.val_dirs, self.test_dirs):
            for sub_dir in task_dir:
                if not sub_dir.is_dir():
                    raise NotADirectoryError(f"{sub_dir} is not a directory")

    def create_dirs(self):
        if self.parent_dir.is_dir() and not any(self.parent_dir.iterdir()):
            raise IsADirectoryError(f"{self.parent_dir} exists and is not empty")

        for task_dir in (self.train_dirs, self.val_dirs, self.test_dirs):
            for sub_dir in task_dir:
                sub_dir.mkdir(parents=True, exist_ok=True)

    def get_image_and_labels_dirs(self, task: Task) -> YoloSubsetDirs:
        match task:
            case Task.TRAIN:
                return self.train_dirs
            case Task.VAL:
                return self.val_dirs
            case Task.TEST:
                return self.test_dirs
            case _:
                raise ValueError(f"Task {task} is invalid")

    @staticmethod
    def from_parent_dir(parent_dir: Path, classes: Iterable[str] = ()) -> 'DatasetDescriptor':
        train_dir = YoloSubsetDirs.from_task_dir(parent_dir / 'train')
        val_dir = YoloSubsetDirs.from_task_dir(parent_dir / 'valid')
        test_dir = YoloSubsetDirs.from_task_dir(parent_dir / 'test')

        return DatasetDescriptor(parent_dir, train_dir, val_dir, test_dir, tuple(classes))

    @staticmethod
    def from_yaml(yaml_path: Path) -> 'DatasetDescriptor':
        if not yaml_path.is_file():
            raise FileNotFoundError(f"{yaml_path} is not a file")

        with yaml_path.open() as f:
            data: dict[str, Any] = yaml.safe_load(f)

        parent = Path(data['path']).resolve()

        # TODO: These should be resolved from the yaml directly,
        #  but I couldn't tell what the logic was
        train_dir = parent / 'train'
        val_dir = parent / 'valid'
        test_dir = parent / 'test'

        num_classes = data['nc']
        classnames = data['names']
        if num_classes != len(data['names']):
            raise ValueError(
                f"Number of classes does not match between `nc` ({num_classes}) and `names` ({classnames})"
            )

        return DatasetDescriptor(
            parent,
            YoloSubsetDirs.from_task_dir(train_dir),
            YoloSubsetDirs.from_task_dir(val_dir),
            YoloSubsetDirs.from_task_dir(test_dir),
            tuple(classnames)
        )
