"""
For drawing bounding boxes on images in YOLO format

I'm making a different file to draw the bounding boxes on the images, separate from `draw_bbox.py`
because I think it'd be quicker than understanding that code.
"""
import cv2

from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np


class BBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float
    class_name: str
    

class YOLOImage(NamedTuple):
    file_path: Path
    bboxes: list[BBox]


def remove_chars(remove: set[str], string: str) -> str:
    """
    Remove characters from a string.
    """
    return ''.join(char for char in string if char not in remove)


def get_labels_mapping(data_yaml_path: Path):
    """
    Get the YOLO labels from the data.yaml file.
    """
    chars_to_remove: set[str] = set(('[', ']', '\n', "'", '"'))
    for line in data_yaml_path.open('r'):
        if 'names' not in line: 
            continue
        
        names = remove_chars(chars_to_remove, line.split(':')[-1].strip()).split(', ')
        print("classes: ", names)
        return names
        
    raise ValueError(f'Could not find labels in {data_yaml_path}.')


def get_labels_from_file(labels_mapping: list[str], file_path: Path) -> Iterable[BBox]:
    """
    Get the labels from a YOLO file.
    """
    allowed_chars = set('0123456789. \n')
    for line in file_path.open('r'):
        # Skip lines with invalid characters
        # TODO: Check if these characters are actually invalid
        if not all(c in allowed_chars for c in line):
            print(f"Skipping line: {line}")
            continue
        
        class_id, x, y, w, h = line.strip(' \n').split(' ')
        yield BBox(float(x), float(y), float(w), float(h), labels_mapping[int(class_id)])

def get_yolo_image(labels_mapping: list[str], dataset_subdir: Path, name: str) -> YOLOImage:
    """
    Get the YOLO image from the dataset directory.
    
    Parameters:
        dataset_subdir: The subdirectory of the dataset, e.g., 'train' or 'val'
        name: The name of the image, e.g., '0095_1' (image 95, tile 1)
    """
    image_path = dataset_subdir / 'images' / f'{name}.png'
    label_path = dataset_subdir / 'labels' / f'{name}.txt'
    
    bboxes = list(get_labels_from_file(labels_mapping, label_path))
    return YOLOImage(image_path, bboxes)

def draw_bbox(image: np.ndarray, bbox: BBox, color = (0, 0, 255)):
    """
    Draw a bounding box on an image. Edits the image in-place.
    
    From: https://stackoverflow.com/a/64097592
    """
    img_width, img_height = image.shape[:2]
    x, y, w, h, class_name = bbox
    
    l = int((x - w / 2) * img_width)
    r = int((x + w / 2) * img_width)
    t = int((y - h / 2) * img_height)
    b = int((y + h / 2) * img_height)
    
    if l < 0:
        l = 0
    if r > img_width - 1:
        r = img_width - 1
    if t < 0:
        t = 0
    if b > img_height - 1:
        b = img_height - 1

    cv2.rectangle(image, (l, t), (r, b), color, 1)
    cv2.putText(image, class_name, (l, t - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_bboxes(yolo_image: YOLOImage, show = True):
    """
    Draw the bounding boxes on the image.
    """
    image = cv2.imread(str(yolo_image.file_path))
    print(f"Found {len(yolo_image.bboxes)} bounding boxes")
    
    for bbox in yolo_image.bboxes:
        draw_bbox(image, bbox)

    if show:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image

def visualize(data_yaml_path: Path, data_sudir: Path, name: str):
    mapping = get_labels_mapping(data_yaml_path)
    yolo_image = get_yolo_image(mapping, data_sudir, name)
    draw_bboxes(yolo_image)
    
if __name__ == '__main__':
    base = Path('C:\\code\\imaging_training_2024\\data\\YOLO_DATASET\\DATASETv1')
    data_yaml_path = base / 'data.yaml'
    data_sudir = base / 'test'
    name = '0095_1'
    
    visualize(data_yaml_path, data_sudir, name)
    