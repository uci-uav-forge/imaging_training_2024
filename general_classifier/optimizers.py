from abc import ABC
from typing import Protocol
from torch.optim import Adam
from torch import nn

from uavf_2024.imaging.general_classifier.resnet import ResNet


# Same as an interface in other languages
class CustomOptimizer(Protocol):
    def __init__(self, model: nn.Module):
        ...
        
    def step(self) -> None:
        raise NotImplementedError()
    
    def zero_grad(self) -> None:
        raise NotImplementedError()


class ResnetOptimizers(CustomOptimizer):
    """
    TODO: Parameterize optimizer selection and parameters.
    TODO: Include scheduling
    """
    def __init__(
        self, 
        model: ResNet,
        initial_lr: float = 0.001,
    ):
        self.backbone_optimizer = Adam(model.backbone.parameters(), lr=initial_lr)
        self.head_optimizers = [
            Adam(head.parameters(), lr=initial_lr) for head in model.heads
        ]
        
    def step(self):
        self.backbone_optimizer.step()
        for head_optimizer in self.head_optimizers:
            head_optimizer.step()
            
    def zero_grad(self):
        self.backbone_optimizer.zero_grad()
        for head_optimizer in self.head_optimizers:
            head_optimizer.zero_grad()
            
