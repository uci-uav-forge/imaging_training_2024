from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.data_transformers import ShapeTargetsOnly, YoloDataTransformer
from yolo_to_yolo.yolo_io import YoloReader, YoloWriter
from yolo_to_yolo.yolo_io_types import PredictionTask


class YoloDataPipeline:
    """
    Class to compose YOLO dataset pipelines.
    """

    def __init__(
        self,
        pipeline: list[YoloDataTransformer],
        classes: list[str] | None = None,
        prediction_task: PredictionTask = PredictionTask.DETECTION
    ):
        """
        NOTE: If the input classes are not the same as the output classes,
        it must be specified because it cannot be inferred.

        Parameters:
            pipeline: List of transformers to apply in sequence.
            classes: List of class names in the output dataset.
                If None, the classes are assumed to be the same as the input dataset.
            prediction_task: The prediction task for the output dataset.
        """
        self.pipeline = pipeline
        self.classes = classes
        self.prediction_task = prediction_task

    def apply(self, input_data: Iterable[YoloImageData]) -> Iterable[YoloImageData]:
        """
        Lazily flat-maps each annotated, YOLO-formatted image into a sequence of them
        and yields the flat results.
        """
        for datum in input_data:
            pipe: Iterable[YoloImageData] = [datum]
            for transformer in self.pipeline:
                pipe = transformer.flat_apply(pipe)

            yield from pipe

    def apply_to_dir(self, input_dir: Path, output_dir: Path) -> None:
        """
        Applies the transformations to an entire directory of YOLO-formatted data.
        """
        reader = YoloReader(input_dir / "data.yaml", self.prediction_task)
        classes = reader.classes if self.classes is None else self.classes
        writer = YoloWriter(output_dir, self.prediction_task, classes)

        data_in = reader.read()
        transformed = tqdm(self.apply(data_in))
        writer.write(transformed)


class DummyTransformer(YoloDataTransformer):
    """
    A dummy transformer that does nothing.
    """

    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        yield input_data


if __name__ == "__main__":
    """
    Example usage for the pipeline using the dummy transformer.
    This should replicate the input directory to the output directory.
    """
    pipeline = YoloDataPipeline(
        pipeline=[DummyTransformer()],
        # classes=["target"],
    )

    pipeline.apply_to_dir(
        Path("/home/minh/Desktop/imaging_training_2024/data/YOLO_DATASET/DATASETv1"),
        Path("/home/minh/Desktop/imaging_training_2024/data/YOLO_DATASET/DATASETv1_processed")
    )
