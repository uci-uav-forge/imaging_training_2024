from pathlib import Path

DATA_DIR = (Path('data'))
IMAGES_DIR = DATA_DIR / 'train'

SEMANTIC_MASKS_DIR = DATA_DIR / 'semantic'
SEMANTIC_LEGEND_DIR = DATA_DIR / 'semantic_segmentation_labels'

BOXES_DIR = DATA_DIR / 'bbox_labels'
BOXES_LEGEND_DIR = DATA_DIR / 'bbox_class_labels'