import shutil
from pathlib import Path
from typing import Generator, NamedTuple
from PIL import Image


class Tile(NamedTuple):
    image: Image.Image
    x_start: int
    y_start: int
    x_end: int
    y_end: int


class TileData(NamedTuple):
    id: str
    image: Image.Image
    labels: str


class Bbox(NamedTuple):
    y: float
    x: float
    height: float
    width: float


class OutputLocations(NamedTuple):
    base_dir: Path
    train_img: Path
    train_label: Path
    valid_img: Path
    valid_label: Path
    test_img: Path
    test_label: Path

    @staticmethod
    def create_from_base_path(base: Path, create_new: bool) -> "OutputLocations":
        """
        Finds the next available version directory within the base directory and creates the necessary subdirectories.

        If create_new is True, it will create a new version directory if one already exists.

        If create_new is False, it will overwrite the highest contiguous version directory.
        """
        version_dir = OutputLocations._create_version_dir(base, create_new)

        train_dir = version_dir / 'train'
        valid_dir = version_dir / 'valid'
        test_dir = version_dir / 'test'

        train_img = train_dir / 'images'
        train_label = train_dir / 'labels'
        valid_img = valid_dir / 'images'
        valid_label = valid_dir / 'labels'
        test_img = test_dir / 'images'
        test_label = test_dir / 'labels'

        for directory in [train_img, train_label, valid_img, valid_label, test_img, test_label]:
            directory.mkdir(parents=True)

        return OutputLocations(
            base_dir=version_dir,
            train_img=train_img,
            train_label=train_label,
            valid_img=valid_img,
            valid_label=valid_label,
            test_img=test_img,
            test_label=test_label
        )

    @staticmethod
    def _create_version_dir(base: Path, create_new: bool) -> Path:
        """
        Find a dataset dir within the base dir following the pattern [base]/DATASETv[version] for the new dataset.

        If create_new is True, it will create a new version directory if one already exists.

        If create_new is False, it will overwrite the highest contiguous version directory.
        """
        version = 1

        while (base / f'DATASETv{version}').exists():
            version += 1

        if not create_new and version > 1:
            version -= 1

        target = base / f'DATASETv{version}'

        if target.exists():
            print(f'Warning: Target directory {target} already exists. Overwriting.')
            data = input('Enter "yes" to confirm deletion...')
            if data == 'yes':
                shutil.rmtree(target, ignore_errors=True)
            else:
                print('Aborting...')
                exit()

        print(f'Creating target directory {target}')
        target.mkdir(parents=True)

        return target


class ClassnameMap:
    """
    Bidirectional mapping between classnames and class ids.

    Class IDs are 0-indexed and counted upwards.

    Removal of a class decreases the class id of all classes with a higher id.
    """

    def __init__(self):
        self.classname_to_id: dict[str, int] = {}
        self.id_to_classname: dict[int, str] = {}

    def add_class(self, classname: str) -> int:
        """
        Adds a class to the mapping and returns the class id.
        """
        if classname in self.classname_to_id:
            return self.classname_to_id[classname]

        class_id = len(self.classname_to_id)
        self.classname_to_id[classname] = class_id
        self.id_to_classname[class_id] = classname

        return class_id

    def remove_class(self, classname: str):
        """
        Removes a class from the mapping.

        Raises a KeyError if the class is not in the mapping.
        """
        class_id = self.classname_to_id.pop(classname)
        del self.id_to_classname[class_id]

        for id, name in self.id_to_classname.items():
            if id > class_id:
                self.classname_to_id[name] -= 1

    def get_class_id(self, classname: str) -> int:
        """
        Gets the class id for the given classname.

        Raises a KeyError if the class is not in the mapping.
        """
        return self.classname_to_id[classname]

    def get_classname(self, class_id: int) -> str:
        """
        Gets the classname for the given class id.

        Raises a KeyError if the class id is not in the mapping.
        """
        return self.id_to_classname[class_id]

    def classnames(self) -> Generator[str, None, None]:
        """
        Yields the classnames in the mapping in order of class id.
        """
        for id in range(len(self)):
            yield self.id_to_classname[id]

    def ids(self) -> Generator[int, None, None]:
        """
        Yields the class ids in the mapping in order of class id.
        """
        yield from range(len(self))

    def __len__(self):
        return len(self.classname_to_id)

    def __iter__(self):
        """
        Yields the classnames in the mapping in order of class id.
        """
        return self.classnames()

    def __contains__(self, classname: str):
        """
        Whether the name is in the mapping.
        """
        return classname in self.classname_to_id
