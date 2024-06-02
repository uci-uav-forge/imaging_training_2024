from pathlib import Path
from typing import Callable, Iterable, NamedTuple

from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import lightning as L

from .default_settings import BATCH_SIZE, DATA_YAML, EPOCHS, LOGS_PATH
from imaging_training_2024.yolo_to_yolo.data_types import YoloImageData
from imaging_training_2024.yolo_to_yolo.yolo_io import YoloReader
from imaging_training_2024.yolo_to_yolo.yolo_io_types import PredictionTask, Task
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


resnet_output_t = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class GeneralClassifierDataset(Dataset):
    """
    Dataset class for the general classifier. One each should be created for the training, validation, and test sets.
    
    Uses yolo_io.YoloReader to read the data.
    """
    def __init__(
        self, 
        yaml_path: Path, 
        task: Task, 
        transformation: Callable[[YoloImageData], YoloImageData | Iterable[YoloImageData]] = lambda x: x
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image_data = self.yolo_reader.read_single(image_path, self.task)
        
        transformed_image_data = self.transformation(image_data)
        
        # If the transformation returns an iterable, we only use the first element.
        if not isinstance(transformed_image_data, YoloImageData):
            transformed_image_data = next(iter(transformed_image_data))
        
        return transformed_image_data


class GeneralClassifierDataloader(DataLoader):
    """
    Lightweight specification of a DataLoader for the general classifier, overriding the collate_fn
    to take a list of YoloImageData objects and return a TrainingBatch object.
    """
    def __init__(self, dataset: GeneralClassifierDataset, batch_size: int = 32, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=GeneralClassifierDataloader.collate)
    
    @staticmethod
    def collate(batch: list[YoloImageData]) -> TrainingBatch:
        """
        Custom collate function for the general classifier.
        
        Args:
            batch: A list of YoloImageData objects.
        
        Returns:
            A TrainingBatch object containing the images and labels.
        """
        images = torch.stack([torch.Tensor(image.image) for image in batch])
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
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    ):
        super().__init__()
        self.model = model
        
        self.train_dataset = GeneralClassifierDataset(yaml_path, Task.TRAIN)
        self.val_dataset = GeneralClassifierDataset(yaml_path, Task.VAL)
        self.test_dataset = GeneralClassifierDataset(yaml_path, Task.TEST)

        self.batch_size = batch_size
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor) -> resnet_output_t:
        return self.model(x)

    def training_step(self, batch: TrainingBatch, batch_idx: int | None = None):
        image, labels = batch
        
        shape_y, shape_color_y, character_y, character_color_y = self._labels_to_y(labels)
        
        shape_pred, shape_color_pred, character_pred, character_color_pred = self.model(image)
        
        shape_color_missing = not torch.count_nonzero(shape_y)
        character_missing = not torch.count_nonzero(character_y)
        character_color_missing = not torch.count_nonzero(character_color_y)

        shape_loss = self.loss_function(shape_pred, shape_y)
        shape_color_loss = self.loss_function(shape_color_pred, shape_color_y) if not shape_color_missing else None
        character_loss = self.loss_function(character_pred, character_y) if not character_missing else None
        character_color_loss = self.loss_function(character_color_pred, character_color_y) if not character_color_missing else None
        
        return ClassificationLosses(shape_loss, shape_color_loss, character_loss, character_color_loss)
    
    def backward(self, losses: ClassificationLosses):
        """
        Custom back-propagation for multi-head loss conditional on present/missing labels.
        
        Losses not computed (for missing labels) must be None to be skipped.
        """
        for loss_value in losses:
            if loss_value is not None:
                loss_value.backward()
        
    @staticmethod
    def _labels_to_y(labels: list[ClassificationLabel]) -> resnet_output_t:
        """
        Creates a list of four one-hot-encoded tensors from a list of ClassificationLabels.
        
        If an annotation category is None for any label, the corresponding tensor will be all zeros.
        """
        if any(label.shape is None for label in labels):
            shape_one_hot = torch.zeros(len(Shape))
        else:
            # label.shape is not None for all labels, as verified above
            shape_indices = torch.Tensor([label.shape.value for label in labels if label.shape is not None])
            shape_one_hot = F.one_hot(shape_indices, num_classes=len(Shape))
        
        if any(label.shape_color is None for label in labels):
            shape_color_one_hot = torch.zeros(len(Color))
        else:
            # label.shape_color is not None for all labels, as verified above
            shape_color_indices = torch.Tensor([label.shape_color.value for label in labels if label.shape_color is not None])
            shape_color_one_hot = F.one_hot(shape_color_indices, num_classes=len(Color))
        
        if any(label.character is None for label in labels):
            character_one_hot = torch.zeros(Character.count())
        else:
            # label.character is not None for all labels, as verified above
            character_indices = torch.Tensor([label.character.value for label in labels if label.character is not None])
            character_one_hot = F.one_hot(character_indices, num_classes=Character.count())
        
        if any(label.character_color is None for label in labels):
            character_color_one_hot = torch.zeros(len(Color))
        else:
            # label.character_color is not None for all labels, as verified above
            character_color_indices = torch.Tensor([label.character_color.value for label in labels if label.character_color is not None])
            character_color_one_hot = F.one_hot(character_color_indices, num_classes=len(Color))
            
        return (shape_one_hot, shape_color_one_hot, character_one_hot, character_color_one_hot)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_dataloader(self):
        return GeneralClassifierDataloader(self.train_dataset, self.batch_size)

    def val_dataloader(self):
        return GeneralClassifierDataloader(self.val_dataset, self.batch_size)

    def test_dataloader(self):
        return GeneralClassifierDataloader(self.test_dataset, self.batch_size)


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
