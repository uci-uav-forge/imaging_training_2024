from pathlib import Path

from .data_transformers import BBoxToCropTransformer
from .yolo_data_pipeline import YoloDataPipeline


def crop_targets(
    input_dir: Path,
    output_dir: Path
) -> None:
    pipeline = YoloDataPipeline([BBoxToCropTransformer(min_padding=5)])
    pipeline.apply_to_dir(input_dir, output_dir)


if __name__ == '__main__':
    datasets_dir = Path("/home/minh/Desktop/uavf_2024/imaging_training_2024/data")
    
    crop_targets(
        Path(datasets_dir / "godot_4000_all_labels"),
        Path(datasets_dir / "godot_4000_cropped")
    )
