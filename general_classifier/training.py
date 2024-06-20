from itertools import repeat
import os
from pathlib import Path
from typing import Callable, Generic, Iterable, Literal, NamedTuple, TypeVar
from logging import warning

from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassPrecisionRecallCurve, MulticlassStatScores
from torchvision.transforms.functional import to_pil_image
import torch
import torchvision

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import PIL.Image
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
    
    def __str__(self) -> str:
        shape = Shape._member_names_[self.shape.value] if self.shape is not None else "No Shape"
        shape_color = Color._member_names_[self.shape_color.value] if self.shape_color is not None else "No Shape Color"
        character = str(self.character) if self.character is not None else "No Character"
        character_color = Color._member_names_[self.character_color.value] if self.character_color is not None else "No character Color"
        
        return " | ".join([shape, shape_color, character, character_color])


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
    
    def iter_labels(self) -> Iterable[ClassificationLabel]:
        """
        Yields ClassificationLabels from the output tensors.
        """
        for shape, shape_color, character, character_color in zip(*self):
            shape = Shape(int(shape.argmax().item()))
            shape_color = Color(int(shape_color.argmax().item()))
            character = Character.from_index(int(character.argmax().item()))
            character_color = Color(int(character_color.argmax().item()))
            yield ClassificationLabel(shape, shape_color, character, character_color)


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

# Read: https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#use-multiple-optimizers-like-gans
ModelT = TypeVar("ModelT", bound=nn.Module)
class GeneralClassifierLightningModule(LightningModule, Generic[ModelT]):    
    def __init__(
        self, 
        model: ModelT,
        yaml_path: Path,
        make_optimizer: Callable[[ModelT], CustomOptimizer],
        batch_size: int = 64,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
    ):
        """
        PyTorch Lightning module to train any all-classes model.
        
        Args:
            model: The model to train.
            yaml_path: Path to the YAML file containing the dataset information.
            make_optimizer: A function that takes the model and returns an optimizer.
            batch_size: The batch size to use for training.
            device: The device to train on.
            loss_function: The loss function to use.
        """
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

    def save_weights(self, path: Path):
        if not (path.suffix == ".pt" or path.suffix == ".pth"):
            raise ValueError("Weights path must have a .pt or .pth extension")
        torch.save(self.model.state_dict(), path)
    
    @staticmethod
    def _all_equals_value(tensor: torch.Tensor, value: float) -> bool:
        return bool(torch.all(tensor == value).item())
    
    def forward(self, x: torch.Tensor) -> ResnetOutputTensors:
        return ResnetOutputTensors(*self.model(x))

    def step(self, batch: TrainingBatch, batch_idx: int | None = None):
        """
        Performs a forward pass, computes losses, and returns the losses and predictions.
        """
        if DEBUG:
            ids = repeat("") if batch.ids is None else batch.ids
            for id, labels in zip(ids, batch.labels):
                for label, field in zip(labels, ClassificationLabel._fields):
                    if label is None:
                        warning(f"Missing {field} for image {id}.")
        
        images, labels, ids = batch
        predictions: ResnetOutputTensors = self.forward(images)
        
        return self._compute_losses(predictions, labels, ids), predictions
    
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
        losses, _ = self.step(batch, batch_idx)
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
        losses, predictions = self.step(batch, batch_idx)
        total_loss = losses.get_total()
        
        self._save_samples(batch, predictions)
        
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
    
    def _save_samples(self, batch: TrainingBatch, preds: ResnetOutputTensors, grid_shape: tuple[int, int] = (4, 4)):
        """
        Make a grid of samples and save it to the log directory.
        
        By default, a 4x4 grid is created.
        """
        log_dir = self.logger.log_dir if self.logger is not None else None
        if log_dir is None:
            warning("Log directory is None. Skipping sample saving.")
            return
        log_dir = Path(log_dir)
        
        images, ys, _ = batch
        y_hats: Iterable[ClassificationLabel] = preds.iter_labels()
        
        grid_contents = [
            __class__._make_labeled_image(img, y, y_hat) 
            for img, y, y_hat, _ in zip(images, ys, y_hats, range(grid_shape[0] * grid_shape[1]))
        ]
        
        grid = torchvision.utils.make_grid(grid_contents, nrow=grid_shape[0])
        image: PIL.Image.Image = to_pil_image(grid)
        
        sample_dir = log_dir / "val_samples"
        sample_dir.mkdir(exist_ok=True, parents=True)
        image.save(sample_dir / f"epoch_{self.current_epoch}.png")
        
    @staticmethod
    def _make_labeled_image(img: torch.Tensor, y: ClassificationLabel, pred: ClassificationLabel) -> torch.Tensor:
        """
        Draws the labels and predictions on an image.
        """
        # Convert to CV2 format
        cv2_img: np.ndarray = img.detach().cpu().numpy().transpose(1, 2, 0)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        
        # Add labels
        cv2.putText(cv2_img, f"truth: {y}", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 2)
        cv2.putText(cv2_img, f"pred: {pred}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)
        
        # Convert back to PyTorch format
        return torch.Tensor(cv2_img).transpose(0, 2).transpose(1, 2)
    
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
        return self._make_dataloader(self.val_dataset, 2)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, 2)


class SaveBestWeightsCallback(Callback):
    def __init__(self, monitor="val_loss", mode: Literal["min","max"]="min"):        
        self.monitor = monitor
        self.mode = mode
        self.best_metric: float = float("inf" if mode == "min" else "-inf")
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: GeneralClassifierLightningModule):      
        metric: float = float(torch.mean(trainer.callback_metrics[self.monitor]).item())
        
        log_dir = trainer.log_dir
        if log_dir is None:
            warning("Log directory is None. Skipping weight saving.")
            return
        log_dir = Path(log_dir) / "weights"
        log_dir.mkdir(exist_ok=True, parents=True)
        
        if self._is_improvement(metric):
            self.best_metric = metric
            save_path = log_dir  / f"best_{trainer.current_epoch}.pt"
            pl_module.save_weights(save_path)
            
            print(f"Saved best weights to {save_path}")
            
    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

def train_resnet(
    model: ResNet = resnet18([len(Shape), len(Color), Character.count(), len(Color)]),
    weights_path: Path | None = None,
    data_yaml: Path = Path(DATA_YAML),
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    logs_path: Path = Path(LOGS_PATH),
    early_stop_metric: str = "average f1"
):
    """
    Extracted to a function for potential use in CLI.
    Generally, settings should be changed in `default_settings.py`.
    """
    terminal_width = os.get_terminal_size()[0]
    print("Training ResNet model".center(terminal_width, "-"))
    
    if weights_path is not None:
        print("Loading weights from", weights_path)
        model.load_state_dict(torch.load(weights_path))
    
    module = GeneralClassifierLightningModule(model, data_yaml, ResnetOptimizers, batch_size, device=torch.device("cuda:0"))
    
    logger = TensorBoardLogger(logs_path, name="general_classifier")
    print("Initalized logger. Logging to", logger.log_dir)
    print(f"Use `tensorboard --logdir={logger.log_dir}` to view logs.")
    
    # Callbacks
    early_stopper = EarlyStopping(monitor=early_stop_metric, mode="max", patience=epochs//2)
    best_weights_saver = SaveBestWeightsCallback(monitor=early_stop_metric, mode="max")
    
    trainer = Trainer(
        precision='16-mixed', 
        max_epochs=epochs, 
        callbacks=[early_stopper, best_weights_saver], 
        logger=logger, 
        default_root_dir=logs_path
    )
    
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
