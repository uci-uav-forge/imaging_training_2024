from abc import abstractmethod, ABC
from typing import Iterable

from uavf_2024.imaging.imaging_types import Character, Color, Shape

from .data_types import YoloImageData, YoloLabel, YoloBbox, YoloOutline

import cv2


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

class BBoxToCropTransformer(YoloDataTransformer):
    """
    Converts bounding box labels to cropped images

    Args:
    min_size: Minimum size of the cropped image (width, height)
    min_padding: Padding to add around the bounding box
    min_char_overlap: Minimum overlap between shape and character bounding boxes to consider them a match (0 to 1)
        (This is used to filter out char bboxes that don't overlap enough with the shape bboxes. Default is 0, meaning it just has to overlap a little bit to be considered a match)
        NOTE: SUPER sensitive when I tried it on test data, even 0.01 was too high for some of the bboxes 
    """
    def __init__(self, min_size: tuple[int, int] = (0, 0), min_padding: int = 0, min_char_overlap: float = 0):
        self.min_size = min_size
        self.min_padding = min_padding
        self.min_char_overlap = min_char_overlap

    @staticmethod
    def _extract_label_categories(labels: Iterable[YoloLabel]) -> tuple[list[YoloLabel], list[YoloLabel], list[YoloLabel]]:
        """
        Returns three lists of labels: shapes, characters, and colors
        """
        shapes = []
        characters = []
        colors = []
        
        for label in labels:
            name = label.classname.upper()
            
            if Shape.from_str(name) is not None:
                shapes.append(label)
            elif Character.from_str(name) is not None:
                characters.append(label)
            elif Color.from_str(name.replace("SHAPE:", ""), name) or Color.from_str(name.replace("CHAR:", ""), name) is not None:
                colors.append(label)
        
        return shapes, characters, colors

    def __call__(self, input_data: YoloImageData) -> Iterable[YoloImageData]:
        img_height, img_width = input_data.image.shape[:2]
        
        shape_labels, char_labels, color_labels = BBoxToCropTransformer._extract_label_categories(input_data.labels)

        for idx, shape_label in enumerate(shape_labels):
            
            if isinstance(shape_label.location, YoloBbox):
                shape_bbox = shape_label.location
                best_char_label = None
                best_overlap = self.min_char_overlap

                # Find best match for character bbox (basically the one that overlaps the most with the shape bbox)
                for char_label in char_labels:
                    if isinstance(char_label.location, YoloBbox):
                        char_bbox = char_label.location
                        overlap = BBoxToCropTransformer._calculate_iou(shape_bbox, char_bbox)

                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_char_label = char_label

                if best_char_label is None:
                    print(f'WARNING: No character bbox found for object of class "{shape_label.classname}", skipping!')
                    continue

                x1, y1, x2, y2 = BBoxToCropTransformer._get_combined_bbox(shape_bbox, best_char_label, img_width, img_height)

                # print(f"Combined BBox for {shape_label.classname}: ({x1}, {y1}) to ({x2}, {y2})")

                # Apply minimum size and padding
                width = max(x2 - x1, self.min_size[0])
                height = max(y2 - y1, self.min_size[1])
                x1 = max(x1 - self.min_padding, 0)
                y1 = max(y1 - self.min_padding, 0)
                x2 = min(x1 + width + self.min_padding * 2, img_width)
                y2 = min(y1 + height + self.min_padding * 2, img_height)

                # Apply crop
                cropped_image = input_data.image[y1:y2, x1:x2]

                # print(f"Cropped Image Shape for {shape_label.classname}: {cropped_image.shape}")

                # Create new labels for the cropped image
                new_labels = [YoloLabel(location=YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0), classname=shape_label.classname), YoloLabel(location=YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0), classname=best_char_label.classname)]

                shape_color_label = None
                char_color_label = None

                # Find color labels
                for color_label in color_labels:
                    if (not shape_color_label) and color_label.location == shape_label.location and color_label.classname.upper().startswith("SHAPE:"):
                        shape_color_label = YoloLabel(YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0), classname=color_label.classname)
                        new_labels.append(shape_color_label)
                    elif (not char_color_label) and color_label.location == best_char_label.location and color_label.classname.upper().startswith("CHAR:"):
                        char_color_label = YoloLabel(YoloBbox(x=0.5, y=0.5, w=1.0, h=1.0), classname=color_label.classname)
                        new_labels.append(char_color_label)
                
                if not shape_color_label:
                    print(f"WARNING: No color label found for shape of class {shape_label.classname}, skipping!")
                if not char_color_label:
                    print(f"WARNING: No color label found for character of class {best_char_label.classname}, skipping!")
                if not shape_color_label or not char_color_label:
                    continue

                # Skip if the image is missing any of the categories
                if not __class__._has_all_categories(input_data.img_id, (shape_label.classname, best_char_label.classname, shape_color_label.classname, char_color_label.classname)):
                    continue
                
                # Make new image data to yield
                new_img_data = YoloImageData(
                    img_id=f"{input_data.img_id}_{idx}_{shape_label.classname}",
                    task=input_data.task,
                    image=cropped_image,
                    labels=new_labels
                )
                # show the cropped image in window 
                # cv2.imshow("cropped image", cropped_image)
                # cv2.waitKey(0)
                yield new_img_data

    @staticmethod
    def _has_all_categories(img_id: str, classnames: Iterable[str]) -> bool:
        has_shape = False
        has_shape_color = False
        has_character = False
        has_character_color = False
        
        for name in classnames:
            name = name.upper()
            if Shape.from_str(name) is not None:
                has_shape = True
            elif Color.from_str(name.replace("SHAPE:", "")) is not None:
                has_shape_color = True
            elif Character.from_str(name) is not None:
                has_character = True
            elif Color.from_str(name.replace("CHAR:", "")) is not None:
                has_character_color = True
                
        if not all([has_shape, has_shape_color, has_character, has_character_color]):
            missing_categories = [
                category for category, has in
                zip(
                    ["shape", "shape color", "character", "character color"], 
                    [has_shape, has_shape_color, has_character, has_character_color]
                ) if not has
            ]
            print(f"{img_id} is missing {', '.join(missing_categories)}.")
            return False
        
        return True
    
    @staticmethod
    def _calculate_iou(bbox1: YoloBbox, bbox2: YoloBbox) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes (this is basically an "overlap score")
        """
        x1_min, y1_min = bbox1.x - bbox1.w / 2, bbox1.y - bbox1.h / 2
        x1_max, y1_max = bbox1.x + bbox1.w / 2, bbox1.y + bbox1.h / 2
        x2_min, y2_min = bbox2.x - bbox2.w / 2, bbox2.y - bbox2.h / 2
        x2_max, y2_max = bbox2.x + bbox2.w / 2, bbox2.y + bbox2.h / 2

        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)

        if intersect_x_min < intersect_x_max and intersect_y_min < intersect_y_max:
            intersection_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
            bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
            bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
            iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
            return iou
        return 0.0

    @staticmethod
    def _get_combined_bbox(shape_bbox: YoloBbox, char_label: YoloLabel, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """
        Get the smallest bbox that contains both the shape and character bboxes.
        """
        x_center, y_center, bbox_width, bbox_height = shape_bbox
        x1 = int((x_center - bbox_width / 2) * img_width)
        y1 = int((y_center - bbox_height / 2) * img_height)
        x2 = int((x_center + bbox_width / 2) * img_width)
        y2 = int((y_center + bbox_height / 2) * img_height)

        if char_label:
            char_location = char_label.location
            if isinstance(char_location, YoloBbox):
                char_x_center, char_y_center, char_bbox_width, char_bbox_height = char_location
                char_x1 = int((char_x_center - char_bbox_width / 2) * img_width)
                char_y1 = int((char_y_center - char_bbox_height / 2) * img_height)
                char_x2 = int((char_x_center + char_bbox_width / 2) * img_width)
                char_y2 = int((char_y_center + char_bbox_height / 2) * img_height)
            elif isinstance(char_location, YoloOutline):
                # Get the smallest bbox that contains all the points in the outline
                points = char_location.points
                char_x1 = int(min(point.x for point in points) * img_width)
                char_y1 = int(min(point.y for point in points) * img_height)
                char_x2 = int(max(point.x for point in points) * img_width)
                char_y2 = int(max(point.y for point in points) * img_height)
            else:
                raise TypeError("char_label.location is supposed to be a YoloBbox or YoloOutline but wasn't")

            x1 = min(x1, char_x1)
            y1 = min(y1, char_y1)
            x2 = max(x2, char_x2)
            y2 = max(y2, char_y2)

        return x1 + 1, y1 + 1, x2, y2

    
    @staticmethod
    def _adjust_bbox(bbox: YoloBbox, crop_x1: int, crop_y1: int, crop_width: int, crop_height: int, img_width: int, img_height: int) -> YoloBbox:
        """
        Adjust the bounding box coordinates to be relative to the cropped image.
        (currently unused but we might need it later, it's called in a piece of commented code in __call__)
        """
        x_center, y_center, bbox_width, bbox_height = bbox
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = bbox_width * img_width
        abs_height = bbox_height * img_height

        new_x_center = (abs_x_center - crop_x1) / crop_width
        new_y_center = (abs_y_center - crop_y1) / crop_height
        new_width = abs_width / crop_width
        new_height = abs_height / crop_height

        return YoloBbox(new_x_center, new_y_center, new_width, new_height)
