from pathlib import Path

from .data_transformers import BBoxToCropTransformer
from .yolo_data_pipeline import YoloDataPipeline


def crop_targets(
    input_dir: Path,
    output_dir: Path,
) -> None:
    pipeline = YoloDataPipeline([BBoxToCropTransformer(min_padding=5, min_size=(64, 64))])
    pipeline.apply_to_dir(input_dir, output_dir)


def copy_dataset(
    input_dir: Path,
    output_dir: Path
) -> None:
    """
    Copy a dataset from one directory to another using an empty pipeline for debugging.
    """
    pipeline = YoloDataPipeline([])
    pipeline.apply_to_dir(input_dir, output_dir)

if __name__ == '__main__':
    datasets_dir = Path("/home/minh/data")
    
    crop_targets(
        Path(datasets_dir / "godot_4000_all_labels"),
        Path(datasets_dir / "godot_4000_cropped_64")
    )

    # copy_dataset(
    #     Path(datasets_dir / "godot_4000_all_labels"),
    #     Path(datasets_dir / "godot_4000_copy")
    # )
