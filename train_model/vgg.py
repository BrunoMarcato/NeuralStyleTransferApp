import torch
import torch.nn as nn
import torchvision

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # Load VGG Skeleton, Pretrained Weights
        vgg16_features = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_FEATURES')
        self.features = vgg16_features.features

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if (name=='22'):
                    break

        return features