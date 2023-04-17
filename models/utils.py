"""
Author: jenningsliu
Date: 2022-03-20 13:05:04
LastEditors: jenningsliu
LastEditTime: 2022-06-21 17:35:12
FilePath: /nerf-loc/models/utils.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch

def camera_project(p3d, K):
    # K0 = torch.eye(3).to(K.device)
    # K0[1,1] = -1
    # K0[2,2] = -1
    # p3d = torch.mm(K0, p3d.t())
    uvz = torch.mm(K, p3d.t())
    z = uvz[2]
    u = uvz[0] / z
    v = uvz[1] / z
    return u,v,z
