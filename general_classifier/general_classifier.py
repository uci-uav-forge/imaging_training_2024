import torch
from torch import nn


class GeneralClassifier(nn.Module):
    """
    Implementation of Resnet-34 for classification of color, shape, and character from cropped bbox.
    """
    def __init__(self, model_path: str | None = None, device=torch.device("cuda:0")):
        """
        TODO: Design and implement model architecture.
        """
        super().__init__()
        self.device = device

        # TODO: Implement model loading
        self.model_path = model_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented

