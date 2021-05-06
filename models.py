import torch
from torch import nn
from torchvision import models


class MobileNetV2(nn.Module):
    """
    MobileNetV2.

    official repo.
    https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
    """

    def __init__(self, num_sections):
        super().__init__()

        self.model = models.mobilenet_v2(
            pretrained=False,  # random initialization
            num_classes=num_sections,  # 1000 (default) -> 3 (00, 01, and 02)
            width_mult=0.5,  # expand ratio
        )

        # dropout is removed and output is unnormalized logits.
        self.model.classifier = nn.Linear(self.model.last_channel, num_sections)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        "Forward propagation through MobileNetV2."

        # duplication; 1ch -> 3ch
        # shape: (512, 1, 64, 128) into (512, 3, 64, 128)
        dup = torch.cat((inputs, inputs, inputs), dim=1)
        output = self.model(dup)

        return output

    def get_loss(self, inputs, labels):
        "Calculate loss function through MobileNetV2."

        output = self.forward(inputs)
        loss = self.criterion(output, labels)

        return loss