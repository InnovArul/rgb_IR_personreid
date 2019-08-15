import os, sys, math
import numpy as np
import torch.nn as nn, torch.nn.functional as F
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        print("creating model " + self.__class__.__name__)
        self.resnet50 = models.resnet50(pretrained=pretrained)

        # modify the initial conv layer
        conv1_layer = self.resnet50.conv1
        conv1_weights = conv1_layer.weight.data.clone()

        self.resnet50.conv1 = nn.Conv2d(
            2,
            out_channels=conv1_layer.out_channels,
            kernel_size=conv1_layer.kernel_size,
            padding=conv1_layer.padding,
            stride=conv1_layer.stride,
            bias=not (conv1_layer is None),
        )

        mean_conv1_weights = conv1_weights.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        self.resnet50.conv1.weight.data.copy_(mean_conv1_weights)

        self.feature_dim = 2048
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        features = x.view(x.size(0), -1)
        classifier_output = self.fc(features)

        return features, classifier_output

