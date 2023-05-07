"""
Author: jenningsliu
Date: 2022-04-27 10:02:03
LastEditors: jenningsliu
LastEditTime: 2022-04-27 10:39:51
FilePath: /nerf-loc/utils/transform/__init__.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import numpy as np
from .rotation_conversions import euler_angles_to_matrix

def get_pose_perturb(translation_noise, rotation_noise):
    pose_perturb = torch.eye(4)
    radians = rotation_noise * (2*torch.rand(3)-1) * np.pi / 180
    pose_perturb[:3,:3] = euler_angles_to_matrix(radians, 'XYZ')
    pose_perturb[:3,3] = (2*torch.rand(3)-1) * translation_noise
    return pose_perturb
