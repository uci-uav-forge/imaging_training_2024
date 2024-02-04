from . import config

import numpy as np
import json
import cv2

from pathlib import Path
from enum import Enum


class ImageOverlay(Enum):
    """
    Enum for the image overlays.
    """
    NONE = 0
    SEMANTIC_MASK = 1
    BOUNDING_BOXES = 2
    BOUNDING_BOXES_WITH_LABELS = 3

class Dataset():
    """
    Interface to the dataset. Currently only supports basic functionality to validate the dataset itself.
    
    Should be converted to be a Torch Dataset for use with a DataLoader in training.
    """
    def __init__(
        self,
        data_dir: Path|str = config.DATA_DIR,
        images_dir: Path|str = config.IMAGES_DIR,
        semantic_masks_dir: Path|str = config.SEMANTIC_MASKS_DIR,
        semantic_legend_dir: Path|str = config.SEMANTIC_LEGEND_DIR,
        boxes_dir: Path|str = config.BOXES_DIR,
        boxes_legend_dir: Path|str = config.BOXES_LEGEND_DIR
    ):
        """
        Defaults are defined in settings.py
        """
        self.data_dir: Path = Path(data_dir)
        self.images_dir: Path = Path(images_dir)
        self.semantic_masks_dir: Path = Path(semantic_masks_dir)
        self.semantic_legend_dir: Path = Path(semantic_legend_dir)
        self.boxes_dir: Path = Path(boxes_dir)
        self.boxes_legend_dir: Path = Path(boxes_legend_dir)
        
    def get_image_path(self, image_id: str, extension: str = 'png') -> Path:
        return self.images_dir / (f"rgb_{image_id}.{extension}")
    
    def get_image(self, image_id: str, extension: str = 'png', overlay = ImageOverlay.NONE) -> np.ndarray:
        """
        Returns the image as a numpy array in default open-cv (BGR) format.
        """
        img_path = self.get_image_path(image_id, extension)
        img = cv2.imread(str(img_path))

        match overlay:
            case ImageOverlay.SEMANTIC_MASK:
                mask = self.get_semantic_mask(image_id)
                img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        
            case ImageOverlay.BOUNDING_BOXES:
                boxes = self.get_boxes(image_id)
                for box in boxes:
                    img = cv2.rectangle(img, (box[1], box[2]), (box[3], box[4]), (0, 0, 255), 2)

            case ImageOverlay.BOUNDING_BOXES_WITH_LABELS:
                boxes = self.get_boxes(image_id)
                boxes_legend = self.get_boxes_legend(image_id)
                
                for box, label in zip(boxes, boxes_legend):
                    label = boxes_legend[label]

                    img = cv2.rectangle(img, (box[1], box[2]), (box[3], box[4]), (0, 0, 255), 2)
                    img = cv2.putText(img, label, (box[1], box[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            case _: ...
            
        return img

    def get_semantic_mask_path(self, image_id: str, extension: str = 'png') -> Path:
        return self.semantic_masks_dir / f"semantic_segmentation_{image_id}.{extension}"
    
    def get_semantic_mask(self, image_id: str, extension: str = 'png') -> np.ndarray:
        mask_path = self.get_semantic_mask_path(image_id, extension)
        return cv2.imread(str(mask_path))
    
    def get_semantic_legend_path(self, id: str, extension: str = 'json') -> Path:
        return self.semantic_legend_dir / f"semantic_segmentation_labels_{id}.{extension}"
    
    def get_semantic_legend(self, id: str, extension: str = 'json') -> dict:
        legend_path = self.get_semantic_legend_path(id, extension)
        with open(legend_path, 'r') as f:
            return json.load(f)
        
    def get_boxes_path(self, image_id: str, extension: str = 'npy') -> Path:
        return self.boxes_dir / f"bounding_box_2d_tight_{image_id}.{extension}"
    
    def get_boxes(self, image_id: str, extension: str = 'npy') -> np.ndarray:
        """
        The boxes are in the format: `(id, x1, y1, x2, y2, rotation)`. 
        It seems that the rotation is always `-1` and can be ignored.
        
        The associated class of the box can be found in the legend given by `get_boxes_legend(image_id)`.
        """
        boxes_path = self.get_boxes_path(image_id, extension)
        return np.load(boxes_path)
    
    def get_boxes_legend_path(self, image_id: str, extension: str = 'json') -> Path:
        return self.boxes_legend_dir / f"bounding_box_2d_tight_labels_{image_id}.{extension}"
    
    def get_boxes_legend(self, image_id: str, extension: str = 'json') -> dict[int, str]:
        """
        Returns a mapping of the box id to the class name for a particular image.
        It's not clear whether the mapping is consistent across images.
        """
        boxes_legend_path = self.get_boxes_legend_path(image_id, extension)
        
        with open(boxes_legend_path, 'r') as f:
            raw: dict[str, dict[str, str]] = json.load(f)

        return {int(key): value["class"] for key, value in raw.items()}

if __name__ == "__main__":
    """
    Example
    """
    import matplotlib.pyplot as plt
    
    dataset = Dataset()
    id = "0007"
    
    img = dataset.get_image(id)[:, :, ::-1] # BGR to RGB
    img_with_mask = dataset.get_image(id, overlay=ImageOverlay.SEMANTIC_MASK)[:, :, ::-1]
    img_with_boxes = dataset.get_image(id, overlay=ImageOverlay.BOUNDING_BOXES_WITH_LABELS)[:, :, ::-1]
    
    # Display
    figs, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img)
    axs[0].set_title("Image")
    
    axs[1].imshow(img_with_mask)
    axs[1].set_title("Image with semantic mask")
    
    axs[2].imshow(img_with_boxes)
    axs[2].set_title("Image with bounding boxes")
    
    plt.show()
    #figs.savefig("example.png", dpi=600, bbox_inches='tight')
    