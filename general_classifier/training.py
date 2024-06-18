from itertools import repeat
from pathlib import Path
from typing import Callable, Generic, Iterable, NamedTuple, TypeVar
from logging import warning

from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassPrecisionRecallCurve, MulticlassStatScores
import torch
import lightning as L
import numpy as np
import cv2

from yolo_to_yolo.data_types import YoloImageData
from yolo_to_yolo.yolo_io import YoloReader
from yolo_to_yolo.yolo_io_types import PredictionTask, Task
from uavf_2024.imaging.general_classifier.resnet import ResNet, resnet18
from uavf_2024.imaging.imaging_types import Character, Color, Shape

from .optimizers import CustomOptimizer, ResnetOptimizers
from .default_settings import BATCH_SIZE, DATA_YAML, EPOCHS, LOGS_PATH, DEBUG


class ClassificationLabel(NamedTuple):
    shape: Shape | None = None
    shape_color: Color | None = None
    character: Character | None = None
    character_color: Color | None = None


class ClassificationMetrics(NamedTuple):
    shape: float | None
    shape_color: float | None
    character: float | None
    character_color: float | None
    
    def average(self) -> float:
        total: float = 0
        count: int = 0
        
        for val in self:
            if val is None:
                continue
            total += val
            count += 1
            
        return total / count


class ClassificationLosses(NamedTuple):
    shape: torch.Tensor
    shape_color: torch.Tensor | None
    character: torch.Tensor | None
    character_color: torch.Tensor | None
    
    def get_total(self) -> torch.Tensor | None:
        loss = sum(loss for loss in self if loss is not None)
        
        if not isinstance(loss, torch.Tensor):
            print("total_loss is not a tensor. This likely means all loss values were empty")
            return None
        
        return loss


class TrainingBatch(NamedTuple):
    images: torch.Tensor
    labels: list[ClassificationLabel]
    ids: list[str] | None = None # IDs of all of the images in for debugging purposes


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
                
        if self.output_size is None:
            return transformed
        
        resized = self.resize(transformed.image)
        return YoloImageData(
            img_id=transformed.img_id,
            task=transformed.task,
            image=resized,
            labels=transformed.labels
        )
    
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
        
        ids = None
        
        if DEBUG:
            ids = [image.img_id for image in batch]
            for img_id, label in zip(ids, labels):
                __class__.warn_missing_labels(label, img_id)
        
        return TrainingBatch(images, labels, ids)
    
    @staticmethod
    def warn_missing_labels(label: ClassificationLabel, img_id: str) -> None:
        """
        Warns if any of the labels are missing.
        """
        for field, val in zip(label._fields, label):
            if val is None:
                warning(f"Missing {field} for image {img_id}.")     
        

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
                continue
            elif classname in Character.__members__:
                character = Character(classname)
                continue
            
            if Color.is_shape_color(classname):
                shape_color = Color.from_str(classname)
                continue
            elif Color.is_char_color(classname):
                character_color = Color.from_str(classname)
                continue
            
            warning(f"Unknown classname: {classname}.")
        
        return ClassificationLabel(shape, shape_color, character, character_color)

# TODO: Implement custom optimizer step
# Read: https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#use-multiple-optimizers-like-gans
ModelT = TypeVar("ModelT", bound=nn.Module)
class GeneralClassifierLightningModule(L.LightningModule, Generic[ModelT]):    
    def __init__(
        self, 
        model: ModelT,
        yaml_path: Path,
        make_optimizer: Callable[[ModelT], CustomOptimizer],
        batch_size: int = 64,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
    ):
        super().__init__()
        self._device = device
        
        self.model = model.to(device)
        
        self.train_dataset = GeneralClassifierDataset(yaml_path, Task.TRAIN)
        self.val_dataset = GeneralClassifierDataset(yaml_path, Task.VAL)
        self.test_dataset = GeneralClassifierDataset(yaml_path, Task.TEST)

        self.batch_size = batch_size
        self.loss_function = loss_function
        
        # Turn off automatic optimization to allow for custom back-propagation.
        self.automatic_optimization = False
        
        self.optimizer = make_optimizer(model)
        
        # Metrics calculators
        self.accuracy_metrics = self._make_classification_metrics(MulticlassAccuracy)
        self.precision_metrics = self._make_classification_metrics(MulticlassPrecision)
        self.recall_metrics = self._make_classification_metrics(MulticlassRecall)
        self.f1_metrics = self._make_classification_metrics(MulticlassF1Score)
        
        print(f"Initialized GeneralClassifierLightningModule on device {self.device}.")
        
    def _make_classification_metrics(self, statistic: type[MulticlassStatScores]):
        return [
            statistic(num_classes).to(self.device)
            for num_classes in [len(Shape), len(Color), Character.count(), len(Color)]
        ]

    def forward(self, x: torch.Tensor) -> ResnetOutputTensors:
        return self.model(x)
    
    @staticmethod
    def _all_equals_value(tensor: torch.Tensor, value: float) -> bool:
        return bool(torch.all(tensor == value).item())
    
    def step(self, batch: TrainingBatch, batch_idx: int | None = None):
        if DEBUG:
            ids = repeat("") if batch.ids is None else batch.ids
            for id, labels in zip(ids, batch.labels):
                for label, field in zip(labels, ClassificationLabel._fields):
                    if label is None:
                        warning(f"Missing {field} for image {id}.")
        
        images, labels, ids = batch
        predictions: ResnetOutputTensors = self.model(images)
        
        return self._compute_losses(predictions, labels, ids)
    
    def _apply_evaluation_conditional(
        self,
        preds: ResnetOutputTensors, 
        labels: list[ClassificationLabel],
        functions: Iterable[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> list[torch.Tensor | None]:
        """
        Applies conditional logic to determine whether to compute metrics based on the presence of labels.
        
        Returns a list of metrics
        """
        results: list[torch.Tensor | None] = []
        
        all_y = __class__._labels_to_indicies(labels)
        
        for func, pred, y in zip(functions, preds, all_y):
            if y is None:
                results.append(None)
                continue
            
            y_tensor = torch.tensor(y, dtype=torch.int8).to(self.device)
            result = func(pred, y_tensor)
            results.append(result)

        return results

    def _compute_losses(
        self, 
        preds: ResnetOutputTensors, 
        labels: list[ClassificationLabel], 
        ids: list[str] | None = None # Image IDs for debugging purposes
    ) -> ClassificationLosses:
        results: list[torch.Tensor | None] = []
        
        all_y = map(lambda t: t.to(self.device), __class__._labels_to_y_distribution(labels))
        
        for field_name, pred, y in zip(ClassificationLosses._fields, preds, all_y):
            is_missing = __class__._all_equals_value(y, -1)
            if is_missing:
                if DEBUG:
                    warning(f"Skipping loss calculation for missing {field_name}{f' ({ids})' if ids is not None else ''}.")
                results.append(None)
                continue
            
            result = self.loss_function(pred, y)
            results.append(result)

        return ClassificationLosses(*results) # type: ignore
    
    def training_step(self, batch: TrainingBatch, batch_idx: int | None = None):
        self.optimizer.zero_grad()
        losses = self.step(batch, batch_idx)
        total_loss = self.backward(losses)
        self.optimizer.step()
        
        if total_loss is not None:
            self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
    
    def backward(self, losses: ClassificationLosses) -> torch.Tensor | None:
        """
        Custom back-propagation for multi-head loss conditional on present/missing labels.
        
        Losses not computed (for missing labels) must be None to be skipped.
        """
        total_loss = losses.get_total()
        
        if not isinstance(total_loss, torch.Tensor):
            return None
        
        total_loss.backward()    
        
        return total_loss    
                
    def validation_step(self, batch: TrainingBatch, batch_idx: int | None = None):
        losses = self.step(batch, batch_idx)
        total_loss = losses.get_total()
        
        if total_loss is not None:
            self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        metrics = self._compute_metrics(self.model(batch.images), batch.labels)
        self._log_classification_metrics(metrics)
        return total_loss
        
    def _log_classification_metrics(self, metrics_dict: dict[str, ClassificationMetrics]):
        for metric_name, metrics in metrics_dict.items():
            for category, value in zip(ClassificationMetrics._fields, metrics):
                if value is not None:
                    self.log(f"{category} {metric_name}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                else:
                    warning(f"{category} {metric_name} is None. This likely means the category is missing from the labels.")
        
        # Aggregate each metric over all the categories
        for metric_name in metrics_dict.keys():
            self.log(f"average {metric_name}", metrics_dict[metric_name].average(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def _compute_metrics(
        self,
        preds: ResnetOutputTensors,
        labels: list[ClassificationLabel]
    ) -> dict[str, ClassificationMetrics]:
        """
        Given a batch of predictions and labels, computes the metrics for each category.
        
        Returns a dictionary of metrics.
        """
        accuracies = self._apply_evaluation_conditional(preds, labels, self.accuracy_metrics)
        precisions = self._apply_evaluation_conditional(preds, labels, self.precision_metrics)
        recalls = self._apply_evaluation_conditional(preds, labels, self.recall_metrics)
        f1s = self._apply_evaluation_conditional(preds, labels, self.f1_metrics)
        # precision_recalls = __class__._apply_evaluation_conditional(preds, labels, self.precision_recall_metrics)
        
        return {
            "accuracy": ClassificationMetrics(*(tensor.item() if tensor is not None else None for tensor in accuracies)),
            "precision": ClassificationMetrics(*(tensor.item() if tensor is not None else None for tensor in precisions)),
            "recall": ClassificationMetrics(*(tensor.item() if tensor is not None else None for tensor in recalls)),
            "f1": ClassificationMetrics(*(tensor.item() if tensor is not None else None for tensor in f1s)),
        }
    
    @staticmethod
    def _labels_to_indicies(
        labels: list[ClassificationLabel]
    ) -> tuple[list[int] | None, list[int] | None, list[int] | None, list[int] | None]:
        """
        Converts a list of ClassificationLabels to a list of indices.
        
        If an annotation category is None for any label, that category's indices will be None.
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
                    
        return (
            shape_indices if not exclude_shape else None,
            shape_color_indices if not exclude_shape_color else None,
            character_indices if not exclude_character else None,
            character_color_indices if not exclude_character_color else None
        )
        
    @staticmethod
    def _labels_to_y_distribution(labels: list[ClassificationLabel]) -> ResnetOutputTensors:
        """
        Creates a list of four one-hot-encoded tensors from a list of ClassificationLabels.
        
        If an annotation category is None for any label, the corresponding tensor will be all -1.
        
        TODO: Refactor this functionality to the Dataset.
        """
        shape_indices, shape_color_indices, character_indices, character_color_indices = __class__._labels_to_indicies(labels)
            
        shape_y = F.one_hot(torch.tensor(shape_indices), len(Shape)).to(torch.float16) if shape_indices is not None else torch.ones(len(labels), len(Shape)) * -1
        shape_color_y = F.one_hot(torch.tensor(shape_color_indices), len(Color)).to(torch.float16) if shape_color_indices else torch.ones(len(labels), len(Color)) * -1
        character_y = F.one_hot(torch.tensor(character_indices), Character.count()).to(torch.float16) if character_indices is not None else torch.ones(len(labels), Character.count()) * -1
        character_color_y = F.one_hot(torch.tensor(character_color_indices), len(Color)).to(torch.float16) if character_color_indices is not None else torch.ones(len(labels), len(Color)) * -1
        
        # Convert to float16 and move to CUDA device.
        return ResnetOutputTensors(*map(
            lambda tensor: tensor.to(torch.float16), 
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


def train_resnet(
    model: ResNet = resnet18([len(Shape), len(Color), Character.count(), len(Color)]),
    data_yaml: Path = Path(DATA_YAML),
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    logs_path: Path = Path(LOGS_PATH)
):
    """
    Extracted to a function for potential use in CLI.
    Generally, settings should be changed in `default_settings.py`.
    """
    module = GeneralClassifierLightningModule(model, data_yaml, ResnetOptimizers, batch_size, device=torch.device("cuda:0"))
    
    logger = TensorBoardLogger(logs_path, name="general_classifier")
    print("Initalized logger. Logging to", logger.log_dir)
    print(f"Use `tensorboard --logdir={logger.log_dir}` to view logs.")
    
    trainer = L.Trainer(max_epochs=epochs, logger=logger, default_root_dir=logs_path)
    
    trainer.fit(module)


def test_dataset():
    dataset = GeneralClassifierDataset(Path(DATA_YAML), Task.VAL)
    dataloader = GeneralClassifierDataloader(dataset, 1, shuffle=False)
    
    for data in dataset:
        print("Read", data.img_id)
        if data.img_id == "2950_0_circle":
            label = GeneralClassifierDataloader.names_to_classification_label(label.classname for label in data.labels)
            print(label)
            break
        

if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")
    train_resnet()

    # test_dataset()
