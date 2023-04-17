# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.ops import FeaturePyramidNetwork
from typing import Dict, List

from .resnet import resnet50
from .fpn import FeaturePyramidNetwork

# from .position_encoding import build_position_encoding
# from COTR.utils import debug_utils, constants

DEFAULT_PRECISION = 'float32'
MAX_SIZE = 256
VALID_NN_OVERLAPPING_THRESH = 0.1


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, return_layers, train_backbone=True, use_fpn=False, fpn_dim=128):
        super().__init__()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.layer_to_channels = layer_to_channels = {
            'conv1': 64,
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048, 
        }
        self.layer_to_stride = layer_to_stride = {
            'conv1': 2,
            'layer1': 4,
            'layer2': 8,
            'layer3': 16,
            'layer4': 32, 
        }
        self.return_layers = return_layers
        backbone = resnet50(
            replace_stride_with_dilation=[False, False, False],
            pretrained=False, 
            norm_layer=FrozenBatchNorm2d
            # norm_layer=nn.BatchNorm2d
            # norm_layer=partial(nn.InstanceNorm2d, track_running_stats=True)
        )
        for name, parameter in backbone.named_parameters():
            if (not train_backbone) or ('layer2' not in name and 'layer3' not in name and 'layer4' not in name):
                parameter.requires_grad_(False)
                print(f'freeze {name}')
        self.body = IntermediateLayerGetter(backbone, return_layers={l:l for l in return_layers})
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=[layer_to_channels[l] for l in return_layers if 'layer' in l], 
                # in_channels_list=[layer_to_channels[l] for l in return_layers], 
                out_channels=fpn_dim,
                # norm_layer=nn.BatchNorm2d
                norm_layer=nn.InstanceNorm2d
            )
            self.layer_to_channels.update({l:fpn_dim for l in return_layers if 'layer' in l})
            # self.layer_to_channels.update({l:fpn_dim for l in return_layers})

    def forward(self, x):
        x = self.normalize(x)
        y = self.body(x)
        if self.use_fpn:
            fpn_input = OrderedDict()
            for l in self.return_layers:
                if 'layer' in l:  # conv1 is not passed to FPN
                    fpn_input[l] = y[l]
                # fpn_input[l] = y[l]
            fpn_out = self.fpn(fpn_input)
            y.update(fpn_out)
        return y


# class Backbone(nn.Module):
#     """ResNet backbone with frozen BatchNorm."""

#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool,
#                  layer='layer3',
#                  num_channels=1024):
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=False, 
#             norm_layer=FrozenBatchNorm2d)
#         super().__init__(backbone, train_backbone, num_channels, return_interm_layers, layer)


def build_backbone(
        model_path='models/COTR/default/checkpoint.pth.tar', 
        return_layers=['layer1', 'layer2', 'layer3', 'layer4'], 
        train_backbone=True,
        use_fpn=False,
        fpn_dim=128):
    ckpt = torch.load(model_path)
    state_dict = {k.replace('backbone.0.', ''):v for k,v in ckpt['model_state_dict'].items() if 'backbone' in k}
    # position_embedding = build_position_encoding(args)
    # train_backbone = False

    backbone = Backbone(return_layers=return_layers, train_backbone=train_backbone, use_fpn=use_fpn, fpn_dim=fpn_dim)
    # backbone.layer_to_stride = layer_to_stride
    # backbone.layer_to_channels = layer_to_channels
    backbone.load_state_dict(state_dict, strict=False)
    return backbone

if __name__ == '__main__':
    model = build_backbone(return_layers=['conv1', 'layer1', 'layer2', 'layer3'], use_fpn=True)
    print(model)
    x = model(torch.randn(1,3,32,32))
    for k,v in x.items():
        print(k, v.shape)
