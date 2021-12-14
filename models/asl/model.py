import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys 
sys.path.append("..") 
sys.path.append("../..") 

from global_utils import get_backbone

class ECABlock(nn.Module):
    def __init__(self, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
    def forward(self, image_features):
        avg_pooled_feature = self.avg_pool(image_features)
        max_pooled_feature = self.max_pool(image_features)

        avg_pool_weights = self.fc(avg_pooled_feature.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_pool_weights = self.fc(max_pooled_feature.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)

        return image_features * weights, weights

class ESABlock(nn.Module):
    def __init__(self, k_size=3):
        super(ESABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    def forward(self, image_features):
        transpose_features = image_features.view(*image_features.shape[:2], -1).transpose(1, 2)
        avg_pooled_features = self.avg_pool(transpose_features)
        max_pooled_features = self.max_pool(transpose_features)
        pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
        pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *image_features.shape[2:])
        pooled_features = self.fc1(pooled_features)
        weights = self.fc2(pooled_features.view(*pooled_features.shape[:2], -1))
        weights = weights.view(-1, 1, *image_features.shape[2:])
        return image_features * weights, weights

class Attribute_Attention(nn.Module):
    def __init__(self, in_channels, semantic_size):
        super(Attribute_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(in_channels + semantic_size, in_channels, kernel_size=1, bias=False)

    def forward(self, image_features, semantic_features):
        avg_pooled_image_features = self.avg_pool(image_features)
        max_pooled_image_features = self.max_pool(image_features)

        # concat in channel dimension
        avg_pooled_features = torch.cat((avg_pooled_image_features, semantic_features), 1)
        max_pooled_features = torch.cat((max_pooled_image_features, semantic_features), 1)
        avg_pool_weights = self.fc(avg_pooled_features)
        max_pool_weights = self.fc(max_pooled_features)
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)

        return  image_features * weights

class ASL(nn.Module):
    def __init__(self, backbone, semantic_size, out_channels):
        super(ASL, self).__init__()
        self.encoder = get_backbone(backbone)
        self.eca_block = ECABlock()
        self.esa_block = ESABlock()
        self.attr_attention = Attribute_Attention(out_channels, semantic_size)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.attr_generator = nn.Conv2d(out_channels, semantic_size, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.ones(1)*0.5)
    def forward(self, inputs, semantics=None, Support=False):

        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))

        # attention after the last conv
        if Support:    # attributes-guided
            attr_label = semantics.float().view(-1, semantics.shape[2])
            semantics = semantics.float().view(-1, semantics.shape[2], 1, 1)
            b, n, _, _ = semantics.shape
            pooled_features = self.avgpool(embeddings)
            generated_attr = self.attr_generator(pooled_features)
            attr_attention =  self.attr_attention(embeddings, semantics)
            eca_embeddings, eca_weights = self.eca_block(embeddings)
            esa_embeddings, esa_weights = self.esa_block(eca_embeddings)
            embeddings = self.gamma * attr_attention + (1-self.gamma) * esa_embeddings

            return embeddings.view(*inputs.shape[:2], -1), generated_attr.view(b, n), attr_label

        else:    # self-guided
            attr_label = semantics.float().view(-1, semantics.shape[2])
            semantics = semantics.float().view(-1, semantics.shape[2], 1, 1)
            b, n, _, _ = semantics.shape

            pooled_features = self.avgpool(embeddings)
            generated_attr = self.attr_generator(pooled_features)

            attr_attention = self.attr_attention(embeddings, generated_attr)
            eca_embeddings, eca_weights = self.eca_block(embeddings)
            esa_embeddings, esa_weights = self.esa_block(eca_embeddings)
            embeddings = self.gamma * attr_attention + (1-self.gamma) * esa_embeddings

            return embeddings.view(*inputs.shape[:2], -1), generated_attr.view(b, n), attr_label
