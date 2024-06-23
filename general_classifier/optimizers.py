from abc import ABC
from itertools import chain

from typing import Generic, TypeVar
from torch.optim import Adam
from torch import nn, optim

from pytorch_lightning.loggers import Logger
from lightning_fabric.loggers import Logger

from uavf_2024.imaging.general_classifier.resnet import ResNet


ModelT = TypeVar("ModelT", bound=nn.Module)
class CustomOptimizer(ABC, Generic[ModelT]):
    def __init__(self, model: ModelT, lr: float = 0.01, logger: Logger | None = None):
        ...
        
    def step(self) -> None:
        raise NotImplementedError()
    
    def step_epoch(self) -> None:
        pass
    
    def zero_grad(self) -> None:
        raise NotImplementedError()


class ResnetOptimizers(CustomOptimizer):
    """
    Abstraction of Adam optimizer and Cosine Annealing learning rate scheduler
    for the multi-headed ResNet model.
    
    TODO: Parameterize optimizer selection and parameters.
    """
    def __init__(
        self, 
        model: ResNet,
        lr: float = 0.01,
        logger: Logger | None = None
    ):
        print(f"Initializing ResnetOptimizers with lr={lr}")
        
        self.backbone_optimizer = Adam(model.backbone.parameters(), lr=lr)
        self.head_optimizers = [
            Adam(head.parameters(), lr=lr) for head in model.heads
        ]
        
        self.schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
            for optimizer in chain(self.head_optimizers, [self.backbone_optimizer])
        ]
        
        self.logger = logger
        
    def step(self):
        self.backbone_optimizer.step()
        for head_optimizer in self.head_optimizers:
            head_optimizer.step()
            
    def step_epoch(self):
        """
        Steps the scheduler, which should happen at the end of each epoch.
        """
        lrs: dict[str, float] = {}
        
        print("Updating learning rates...")
        for index, scheduler in enumerate(self.schedulers):
            scheduler.step()
            
            if self.logger is not None:
                # Type annotation is incorrect per https://github.com/pytorch/pytorch/issues/100804
                last_lr: float = scheduler.get_last_lr() # type: ignore
                lrs[f"CosLR_{index}"] = last_lr
                
        if self.logger is not None:
            self.logger.log_hyperparams(lrs)
            
    def zero_grad(self):
        self.backbone_optimizer.zero_grad()
        for head_optimizer in self.head_optimizers:
            head_optimizer.zero_grad()
            