"""
General implementation of Resnet.
Adapted from: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb
"""
from typing import Sequence

import torch
from torch import nn


class BasicBlock(nn.Module):
    EXPANSION = 1

    @staticmethod
    def make_conv3x3(in_planes: int, out_planes: int, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def __init__(self, in_planes: int, out_planes: int, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicBlock.make_conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BasicBlock.make_conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    IN_CHANNELS = 3

    def __init__(self, block, layers: Sequence[int], num_classes: int, dry_run=False):
        """
        TODO: Support for multi-head classification. It's currently single-head because
            there is only one FC layer, so multiple heads would do the same thing.
            This might be especially useful if we start with a Resnet-18 backbone instead.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dry_run = dry_run

        self.submodules = nn.ModuleDict({
            "conv1": nn.Conv2d(
                self.IN_CHANNELS, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            ),
            "bn1": nn.BatchNorm2d(64),
            "relu": nn.ReLU(inplace=True),
            "maxpool": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            "layer1": self._make_layer(block, 64, layers[0]),
            "layer2": self._make_layer(block, 128, layers[1], stride=2),
            "layer3": self._make_layer(block, 256, layers[2], stride=2),
            "layer4": self._make_layer(block, 512, layers[3], stride=2),
            "avgpool": nn.AvgPool2d(7, stride=1),
            # NOTE: Currently single-headed
            "flatten": nn.Flatten(),
            "fc": nn.Linear(512 * block.EXPANSION, num_classes),
        })

    def init_weights(self):
        """
        Initializes weights, depending on the layer type.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.EXPANSION:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.EXPANSION,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.EXPANSION),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.EXPANSION
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the logits and the probabilities.
        """
        if self.dry_run:
            print("Input shape: ", x.shape)

        for module_name, module in self.submodules.items():
            x = module(x)
            if self.dry_run:
                print(f"Shape after {module_name}: ", x.shape)

        probabilities = nn.functional.softmax(x, dim=1)
        return x, probabilities


def make_resnet34(num_classes: int, dry_run=False):
    """Constructs a ResNet-34 model."""
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        dry_run=dry_run
    )


if __name__ == "__main__":
    model = make_resnet34(10, dry_run=True)
    # print(model)
    model(torch.rand(1, 3, 224, 224))
