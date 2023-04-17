"""
Author: jenningsliu
Date: 2022-03-09 18:02:30
LastEditors: jenningsliu
LastEditTime: 2022-03-09 18:30:32
FilePath: /nerf-loc/models/COTR/position_encoding.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i+1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2**i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat([
            torch.sin(i * math.pi * inputs) for i in self.bases
        ] + [torch.cos(i * math.pi * inputs) for i in self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, sine_type='lin_sine'):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.sine = NerfPositionalEncoding(num_pos_feats//2, sine_type)

    @torch.no_grad()
    def forward(self, x):
        """
        Args: 
            x: B,H,W
        Returns: 
            B,H,W,C
        """        
        mask = torch.ones_like(x)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = (y_embed-0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
        pos = torch.stack([x_embed, y_embed], dim=-1)
        return self.sine(pos)



def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('lin_sine', 'exp_sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, sine_type=args.position_embedding)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
