from pathlib import Path

from .data_transformers import BBoxToCropTransformer
from .yolo_data_pipeline import YoloDataPipeline


def crop_targets(
    input_dir: Path,
    output_dir: Path
) -> None:
    pipeline = YoloDataPipeline([BBoxToCropTransformer()])
    pipeline.apply_to_dir(input_dir, output_dir)


if __name__ == '__main__':
    datasets_dir = Path("/home/minh/Desktop/uavf_2024/imaging_training_2024/data")
    
    crop_targets(
        Path(datasets_dir / "crop_bug_min_reproduce"),
        Path(datasets_dir / "isaac_godot_processed")
    )
