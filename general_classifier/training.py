from typing import Callable, NamedTuple

from torch.nn import functional as F
import torch.nn as nn
import torch

from uavf_2024.imaging.imaging_types import CHARACTERS, Character, Color, Shape


class ClassificationLabel(NamedTuple):
    shape: Shape
    shape_color: Color | None
    character: Character | None
    character_color: Color | None
    

class ClassificationLosses(NamedTuple):
    shape: torch.Tensor
    shape_color: torch.Tensor | None
    character: torch.Tensor | None
    character_color: torch.Tensor | None


class TrainingBatch(NamedTuple):
    images: torch.Tensor
    labels: list[ClassificationLabel]


resnet_output_t = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class GeneralClassifierTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    ):
        self.model = model
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor) -> resnet_output_t:
        return self.model(x)

    def training_step(self, batch: TrainingBatch):
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
        
    @staticmethod
    def _labels_to_y(labels: list[ClassificationLabel]) -> resnet_output_t:
        """
        Creates a list of four one-hot-encoded tensors from a list of ClassificationLabels.
        
        If an annotation category is None for any label, the corresponding tensor will be all zeros.
        """
        shape_indices = torch.Tensor([label.shape.value for label in labels])
        shape_one_hot = F.one_hot(shape_indices, num_classes=len(Shape))
        
        if any(label.shape_color is None for label in labels):
            shape_color_one_hot = torch.zeros(len(Color))
        else:
            # label.shape_color is not None for all labels, as verified above
            shape_color_indices = torch.Tensor([label.shape_color.value for label in labels if label.shape_color is not None])
            shape_color_one_hot = F.one_hot(shape_color_indices, num_classes=len(Color))
        
        if any(label.character is None for label in labels):
            character_one_hot = torch.zeros(len(Character))
        else:
            # label.character is not None for all labels, as verified above
            character_indices = torch.Tensor([label.character.value for label in labels if label.character is not None])
            character_one_hot = F.one_hot(character_indices, num_classes=len(Character))
        
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
        ...

    def val_dataloader(self):
        ...

    def test_dataloader(self):
        ...
