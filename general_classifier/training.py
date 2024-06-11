from pathlib import Path
from typing import Callable, Iterable, NamedTuple

from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import lightning as L
import numpy as np
import cv2

from .default_settings import BATCH_SIZE, DATA_YAML, EPOCHS, LOGS_PATH, DEBUG
from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.yolo_io import YoloReader
from yolo_to_yolo.yolo_io_types import PredictionTask, Task
from uavf_2024.imaging.general_classifier.resnet import resnet18
from uavf_2024.imaging.imaging_types import Character, Color, Shape


class ClassificationLabel(NamedTuple):
    shape: Shape | None = None
    shape_color: Color | None = None
    character: Character | None = None
    character_color: Color | None = None
    

class ClassificationLosses(NamedTuple):
    shape: torch.Tensor
    shape_color: torch.Tensor | None
    character: torch.Tensor | None
    character_color: torch.Tensor | None


class TrainingBatch(NamedTuple):
    images: torch.Tensor
    labels: list[ClassificationLabel]


class ResnetOutputTensors(NamedTuple):
    shape: torch.Tensor
    shape_color: torch.Tensor
    character: torch.Tensor
    character_color: torch.Tensor


class GeneralClassifierDataset(Dataset):
    """
    Dataset class for the general classifier. One each should be created for the training, validation, and test sets.
    
    Uses yolo_io.YoloReader to read the data.
    """
    RESIZE_METHOD = cv2.INTER_CUBIC
    
    def __init__(
        self, 
        yaml_path: Path, 
        task: Task, 
        transformation: Callable[[YoloImageData], YoloImageData | Iterable[YoloImageData]] = lambda x: x,
        output_size: tuple[int, int] | None = (224, 224),
    ):
        """
        Args:
            yaml_path: Path to the YAML file containing the dataset information.
            task: The task to read the data for.
            transformation: A function that takes a YoloImageData object and returns a transformed version of it.
                If it's an Iterable, only the first element will be used. This is to support the current Augmentation classes.
        """
        self.task = task
        
        # We use detection-formatted data because we assume that the whole image
        # is all the classes within it.
        self.yolo_reader = YoloReader(yaml_path, PredictionTask.DETECTION)
        
        # We need to store this so that we can get the length of the dataset and index it.
        self.image_paths = list(self.yolo_reader.get_image_paths(task))
        
        self.transformation = transformation
        
        self.output_size = output_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image_data = self.yolo_reader.read_single(self.task, image_path)
        
        transformed = self.transformation(image_data)
        
        # If the transformation returns an iterable, we only use the first element.
        if not isinstance(transformed, YoloImageData):
            transformed = next(iter(transformed))
            
        if DEBUG:
            self._print_missing_categories(transformed.img_id, (label.classname for label in transformed.labels))
                
        if self.output_size is None:
            return transformed
        
        resized = self.resize(transformed.image)
        return YoloImageData(
            img_id=transformed.img_id,
            task=transformed.task,
            image=resized,
            labels=transformed.labels
        )
        
    @staticmethod
    def _print_missing_categories(img_id: str, classnames: Iterable[str]):
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
    
    def resize(self, image: np.ndarray):
        if not self.output_size:
            return image
        
        return cv2.resize(image, self.output_size)
            

class GeneralClassifierDataloader(DataLoader):
    """
    Lightweight specification of a DataLoader for the general classifier, overriding the collate_fn
    to take a list of YoloImageData objects and return a TrainingBatch object.
    """
    def __init__(
        self, 
        dataset: GeneralClassifierDataset, 
        batch_size: int = 32, 
        num_workers: int = 2, 
        shuffle: bool = True
    ):
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=GeneralClassifierDataloader.collate
        )
    
    @staticmethod
    def collate(batch: list[YoloImageData]) -> TrainingBatch:
        """
        Custom collate function for the general classifier.
        
        Args:
            batch: A list of YoloImageData objects.
        
        Returns:
            A TrainingBatch object containing the images and labels.
        """
        # Stack the images and transpose them to CHW
        images = torch.stack([torch.Tensor(image.image) for image in batch]).transpose(1, 3)
        labels = [
            __class__.names_to_classification_label(
                label.classname for label in image.labels
            ) for image in batch
        ]
        
        return TrainingBatch(images, labels)

    @staticmethod
    def names_to_classification_label(classnames: Iterable[str]) -> ClassificationLabel:
        """
        Converts a list of classnames to a ClassificationLabel.
        
        Args:
            classnames: An iterable of classnames.
        
        Returns:
            A ClassificationLabel object.
        """
        shape = None
        shape_color = None
        character = None
        character_color = None
        
        for classname in map(str.upper, classnames):
            if classname in Shape.__members__:
                shape = Shape[classname]
            elif classname in Color.__members__:
                shape_color = Color[classname]
            elif classname in Character.__members__:
                character = Character(classname)
            elif classname in Color.__members__:
                character_color = Color[classname]
        
        return ClassificationLabel(shape, shape_color, character, character_color)


class GeneralClassifierLightningModule(L.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        yaml_path: Path,
        batch_size: int = 32,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
    ):
        super().__init__()
        self.model = model
        
        self.train_dataset = GeneralClassifierDataset(yaml_path, Task.TRAIN)
        self.val_dataset = GeneralClassifierDataset(yaml_path, Task.VAL)
        self.test_dataset = GeneralClassifierDataset(yaml_path, Task.TEST)

        self.batch_size = batch_size
        self.loss_function = loss_function
        
        print(f"Initialized GeneralClassifierLightningModule on device {self.device}.")

    def forward(self, x: torch.Tensor) -> ResnetOutputTensors:
        return self.model(x)
    
    @staticmethod
    def _all_equals_value(tensor: torch.Tensor, value: float) -> bool:
        return bool(torch.all(tensor == value).item())
    
    def step(self, batch: TrainingBatch, batch_idx: int | None = None):
        image, labels = batch
        
        shape_y, shape_color_y, character_y, character_color_y = self._labels_to_y(labels)
        
        shape_color_missing = __class__._all_equals_value(shape_y, -1)
        character_missing = __class__._all_equals_value(character_y, -1)
        character_color_missing = __class__._all_equals_value(character_color_y, -1)
        
        predictions: ResnetOutputTensors = self.model(image)
        shape_pred, shape_color_pred, character_pred, character_color_pred = predictions

        shape_loss = self.loss_function(shape_pred, shape_y)
        shape_color_loss = self.loss_function(shape_color_pred, shape_color_y) if not shape_color_missing else None
        character_loss = self.loss_function(character_pred, character_y) if not character_missing else None
        character_color_loss = self.loss_function(character_color_pred, character_color_y) if not character_color_missing else None
        
        return ClassificationLosses(shape_loss, shape_color_loss, character_loss, character_color_loss)

    def training_step(self, batch: TrainingBatch, batch_idx: int | None = None):
        return self.step(batch, batch_idx)
    
    def backward(self, losses: ClassificationLosses):
        """
        Custom back-propagation for multi-head loss conditional on present/missing labels.
        
        Losses not computed (for missing labels) must be None to be skipped.
        """
        for loss_value in losses:
            if loss_value is not None:
                loss_value.backward()
                
    def validation_step(self, batch: TrainingBatch, batch_idx: int | None = None):
        return self.step(batch, batch_idx)
        
    @staticmethod
    def _labels_to_y(labels: list[ClassificationLabel]) -> ResnetOutputTensors:
        """
        Creates a list of four one-hot-encoded tensors from a list of ClassificationLabels.
        
        If an annotation category is None for any label, the corresponding tensor will be all -1.
        
        TODO: Refactor this functionality to the Dataset.
        """
        exclude_shape = False
        exclude_shape_color = False
        exclude_character = False
        exclude_character_color = False
        
        shape_indices: list[int] = []
        shape_color_indices: list[int] = []
        character_indices: list[int] = []
        character_color_indices: list[int] = []
        
        for label in labels:
            if not exclude_shape:
                if label.shape is None:
                    exclude_shape = True
                else:
                    shape_indices.append(label.shape.value)
                    
            if not exclude_shape_color:
                if label.shape_color is None:
                    exclude_shape_color = True
                else:
                    shape_color_indices.append(label.shape_color.value)
                    
            if not exclude_character:
                if label.character is None:
                    exclude_character = True
                else:
                    character_indices.append(label.character.index)
                    
            if not exclude_character_color:
                if label.character_color is None:
                    exclude_character_color = True
                else:
                    character_color_indices.append(label.character_color.value)
            
        shape_y = F.one_hot(torch.tensor(shape_indices), len(Shape)).to(torch.float16) if not exclude_shape else torch.ones(len(labels), len(Shape)) * -1
        shape_color_y = F.one_hot(torch.tensor(shape_color_indices), len(Color)).to(torch.float16) if not exclude_shape_color else torch.ones(len(labels), len(Color)) * -1
        character_y = F.one_hot(torch.tensor(character_indices), Character.count()).to(torch.float16) if not exclude_character else torch.ones(len(labels), Character.count()) * -1
        character_color_y = F.one_hot(torch.tensor(character_color_indices), len(Color)).to(torch.float16) if not exclude_character_color else torch.ones(len(labels), len(Color)) * -1
        
        # Convert to float16 and move to CUDA device.
        return ResnetOutputTensors(*map(
            lambda tensor: tensor.to(torch.float16).to("cuda:0"), 
            [shape_y, shape_color_y, character_y, character_color_y]
        ))
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _make_dataloader(self, dataset: GeneralClassifierDataset, num_workers: int = 2, shuffle: bool = True):
        return GeneralClassifierDataloader(dataset, self.batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, 6)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, 2, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, 2, shuffle=False)


def train(
    model: nn.Module = resnet18([len(Shape), len(Color), Character.count(), len(Color)]),
    data_yaml: Path = Path(DATA_YAML),
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    logs_path: Path = Path(LOGS_PATH)
):
    """
    Extracted to a function for potential use in CLI.
    Generally, settings should be changed in `default_settings.py`.
    """
    module = GeneralClassifierLightningModule(model, data_yaml, batch_size)
    
    logger = TensorBoardLogger(logs_path, name="general_classifier")
    trainer = L.Trainer(max_epochs=epochs, logger=logger)
    
    trainer.fit(module)


if __name__ == '__main__':
    train()
