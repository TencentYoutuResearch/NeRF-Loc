"""
Author: jenningsliu
Date: 2022-04-14 21:10:00
LastEditors: jenningsliu
LastEditTime: 2022-08-18 12:36:13
FilePath: /nerf-loc/models/appearance_embedding.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .COTR.backbone2d import build_backbone as build_cotr_backbone

class AppearanceEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.appearance_emb_dim

    def forward(self, imgs, x):
        """
        Args: 
            imgs: b,3,h,w
            x: {'conv1': [b,c,h,w]}
        Returns: 
        """        
        xs = []
        b,c = x['conv1'].shape[:2] # 64
        for i in range(b):
            std,mean = torch.std_mean(x['conv1'][i].view(c,-1), dim=1)
            xs.append(torch.cat([mean,std])) # 128
        x = torch.stack(xs)
        assert x.shape[-1] == self.dim
        return x

class AppearanceAdaptLayer(nn.Module):
    def __init__(self, args, input_dim, is_rgb=False):
        super().__init__()
        self.input_dim = input_dim
        self.appearance_emb_dim = args.appearance_emb_dim
        self.is_rgb = is_rgb
        self.mlp = nn.Sequential(
            nn.Linear(self.appearance_emb_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, input_dim*2)
        )

    def forward(self, x, embedding, target_embedding):
        """
        Args: 
            x: B,H,W,C
            embedding: B,appearance_emb_dim
            target_embedding: 1,appearance_emb_dim
        Returns: 
            B,H,W,C
        """        
        embedding_diff = target_embedding - embedding
        code = self.mlp(embedding_diff)
        a, b = torch.split(code, [self.input_dim, self.input_dim], dim=-1)
        y = a[:,None,None,:] * x + b[:,None,None,:]
        if self.is_rgb:
            y = torch.clip(y, min=0., max=1.)
        return y

if __name__ == '__main__':
    from collections import namedtuple
    Config = namedtuple('Config', ['appearance_emb_dim'])
    args = Config(128)
    net = AppearanceEmbeddingNetwork(args)
    x = net(torch.randn([1,3,512,512]))
    print(x.shape, x)
