from pathlib import Path
from typing import NamedTuple, Any, Iterable, Generator
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
            classnames
        )


class ClassnameMap:
    """
    Bidirectional mapping between classnames and class ids.

    Class IDs are 0-indexed and counted upwards.

    Removal of a class decreases the class id of all classes with a higher id.
    """

    def __init__(self):
        self._classname_to_id: dict[str, int] = {}
        self._id_to_classname: dict[int, str] = {}

    @staticmethod
    def from_classnames(classnames: Iterable[str]) -> 'ClassnameMap':
        """
        Creates a ClassnameMap from an iterable of classnames.
        """
        classname_map = ClassnameMap()

        for classname in classnames:
            classname_map.add_class(classname)

        return classname_map

    def add_class(self, classname: str) -> int:
        """
        Adds a class to the mapping and returns the class id if it is new.
        Otherwise, returns the existing class id.
        """
        if classname in self._classname_to_id:
            return self._classname_to_id[classname]

        class_id = len(self._classname_to_id)
        self._classname_to_id[classname] = class_id
        self._id_to_classname[class_id] = classname

        return class_id

    def remove_class(self, classname: str):
        """
        Removes a class from the mapping.

        Raises a KeyError if the class is not in the mapping.
        """
        class_id = self._classname_to_id.pop(classname)
        del self._id_to_classname[class_id]

        for id, name in self._id_to_classname.items():
            if id > class_id:
                self._classname_to_id[name] -= 1

    def get_class_id(self, classname: str) -> int:
        """
        Gets the class id for the given classname.

        Raises a KeyError if the class is not in the mapping.
        """
        return self._classname_to_id[classname]

    def get_classname(self, class_id: int) -> str:
        """
        Gets the classname for the given class id.

        Raises a KeyError if the class id is not in the mapping.
        """
        return self._id_to_classname[class_id]

    def classnames(self) -> Generator[str, None, None]:
        """
        Yields the classnames in the mapping in order of class id.
        """
        for id in range(len(self)):
            yield self._id_to_classname[id]

    def ids(self) -> Generator[int, None, None]:
        """
        Yields the class ids in the mapping in order of class id.
        """
        yield from range(len(self))

    def __len__(self):
        return len(self._classname_to_id)

    def __iter__(self):
        """
        Yields the classnames in the mapping in order of class id.
        """
        return self.classnames()

    def __contains__(self, classname: str):
        """
        Whether the name is in the mapping.
        """
        return classname in self._classname_to_id

