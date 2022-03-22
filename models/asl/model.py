import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import sys 
sys.path.append("..") 
sys.path.append("../..") 

from global_utils import get_backbone

class MSAA(nn.Module):
    def __init__(self):
        super(MSAA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=9, padding=4, bias=False)
    def forward(self, image_features, semantic_features):
        features = torch.cat((image_features, semantic_features.expand(*semantic_features.shape[:2], *image_features.shape[2:])), 1)    # broadcast along height and width dimension
        transpose_features = features.view(*features.shape[:2], -1).transpose(1, 2)
        avg_pooled_features = self.avg_pool(transpose_features)
        max_pooled_features = self.max_pool(transpose_features)
        pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
        pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *image_features.shape[2:])
        weights1 = torch.sigmoid(self.conv1(pooled_features))
        weights2 = torch.sigmoid(self.conv2(pooled_features))
        weights3 = torch.sigmoid(self.conv3(pooled_features))
        weights4 = torch.sigmoid(self.conv4(pooled_features))
        weights = weights1 + weights2 + weights3 + weights4
        weights = torch.sigmoid(weights)
        return image_features * weights

class CAA(nn.Module):
    def __init__(self, in_channels, semantic_size):
        super(CAA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels + semantic_size, in_channels//4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, kernel_size=1, bias=False),
        )
    def forward(self, image_features, semantic_features):
        avg_pooled_image_features = self.avg_pool(image_features)
        max_pooled_image_features = self.max_pool(image_features)

        # concat in channel dimension
        avg_pooled_features = torch.cat((avg_pooled_image_features, semantic_features), 1)
        max_pooled_features = torch.cat((max_pooled_image_features, semantic_features), 1)
        avg_pool_weights = self.fc(avg_pooled_features)
        max_pool_weights = self.fc(max_pooled_features)
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)
        return image_features * weights

class ASL(nn.Module):
    def __init__(self, backbone, semantic_size, out_channels):
        super(ASL, self).__init__()
        self.encoder = get_backbone(backbone)
        self.CAA = CAA(out_channels,semantic_size)
        self.MSAA = MSAA()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.VAG = nn.Conv2d(out_channels, semantic_size, kernel_size=1, bias=False)
    def forward(self, inputs, semantics=None, Support=False):

        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))

        # attention after the last conv
        if Support:    # attributes-guided
            attr_label = semantics.float().view(-1, semantics.shape[2])
            semantics = semantics.float().view(-1, semantics.shape[2], 1, 1)
            b, n, _, _ = semantics.shape
            pooled_features = self.avgpool(embeddings)
            generated_attr = self.VAG(pooled_features)
            embeddings = self.CAA(embeddings, semantics)
            embeddings = self.MSAA(embeddings, semantics)
            return embeddings.view(*inputs.shape[:2], -1), generated_attr.view(b, n), attr_label

        else:    # self-guided
            attr_label = semantics.float().view(-1, semantics.shape[2])
            semantics = semantics.float().view(-1, semantics.shape[2], 1, 1)
            b, n, _, _ = semantics.shape

            pooled_features = self.avgpool(embeddings)
            generated_attr = self.VAG(pooled_features)
            embeddings = self.CAA(embeddings, generated_attr)
            embeddings = self.MSAA(embeddings, generated_attr)
            return embeddings.view(*inputs.shape[:2], -1), generated_attr.view(b, n), attr_label
