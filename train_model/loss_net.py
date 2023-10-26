"""
    Code from: https://github.com/gordicaleksa/pytorch-neural-style-transfer-johnson/blob/master/models/definitions/perceptual_loss_net.py
"""

from collections import namedtuple
from torchvision import models
import torch.nn as nn

class LossNet(nn.Module):
    def __init__(self, requires_grad = False, show_progress = False):
        super().__init__()

        self.style_features = ["3", "8", "15", "22"]

        self.model = models.vgg16(pretrained=True).features.eval()
        self.names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), self.model[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.model[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.model[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.model[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", self.names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)

        return out