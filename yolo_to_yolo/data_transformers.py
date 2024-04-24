from abc import abstractmethod, ABC
from typing import Iterable

from .data_types import YoloImageData, YoloLabel


class YoloDataTransformer(ABC):
    """
    Parent class for YOLO data transformations, e.g., augmentations, tiling, classname-mapping, etc.

    Subclasses only need to implement the __call__ method.
    """
    @abstractmethod
    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        """
        Transforms one annotated, YOLO-formatted image into a sequence of them.

        The one-to-many signature is to support sub-tiling or offline augmentation steps.
        """
        ...

    def flat_apply(self, input_data: Iterable[YoloImageData]) -> Iterable[YoloImageData]:
        """
        Applies the transformation to each annotated, YOLO-formatted image, outputting a
        possibly-longer sequence of them.
        """
        for datum in input_data:
            yield from self(datum)


class ShapeTargetsOnly(YoloDataTransformer):
    """
    Filters out non-shape labels and changes shape labels to 'target'
    """

    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        new_labels: list[YoloLabel] = []
        for label in input_data.labels:
            classname = label.classname.lower()
            if self.include(classname):
                new_label = YoloLabel(label.location, 'target')
                new_labels.append(new_label)

        yield YoloImageData(input_data.img_id, input_data.task, input_data.image, new_labels)

    @staticmethod
    def include(classname: str) -> bool:
        return classname != 'background' and len(classname) > 1

