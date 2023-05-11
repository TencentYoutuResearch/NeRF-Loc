"""
Author: jenningsliu
Date: 2022-03-02 13:56:51
LastEditors: jenningsliu
LastEditTime: 2022-08-10 20:47:41
FilePath: /nerf-loc/models/pointnerf/losses.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# transform to inverse depth coordinate
def to_inverse_normalized_depth(depth, near, far):
    near_inv, far_inv = -1/near, -1/far # rfn,1
    depth = torch.clamp(depth, min=1e-5)
    depth = -1 / depth
    depth = (depth - near_inv) / (far_inv - near_inv)
    depth = torch.clamp(depth, min=0, max=1.0)
    return depth

class RenderingLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
    """
    def __init__(self, coef=1, lambda_u=0.01, use_depth=False, depth_std=0.05):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        self.use_depth = use_depth
        self.depth_std = depth_std

    def forward(self, inputs, targets):
        if 'mask' not in targets:
            mask = torch.ones_like(targets['rgb'][:,0]).bool()
        else:
            mask = targets['mask'].bool()
        rgb = inputs['rgb'][mask]
        depth = inputs['depth'][mask]
        rgb_target = targets['rgb'][mask]

        if 'beta' in inputs:
            beta = inputs['beta'][mask]
            rgb_loss = ((rgb-rgb_target)**2/(2*beta.unsqueeze(1)**2)).mean()
            beta_loss = 3 + torch.log(beta).mean() # +3 to make it positive
            loss = self.coef * (rgb_loss + beta_loss)
        else:
            rgb_loss = ((rgb-rgb_target)**2).mean()
            loss = self.coef * rgb_loss

        if self.use_depth and 'depth' in targets:
            target_depth = targets['depth'][mask]

            # weights = inputs['weights']

            # valid_depth_mask = target_depth > 0
            # depth_mask = (target_depth > 0) & ((depth-target_depth).abs()>self.depth_std)
            depth_mask = target_depth > 0
            near, far = targets['depth_range']
            target_depth = to_inverse_normalized_depth(target_depth, near, far)
            depth = to_inverse_normalized_depth(depth, near, far)

            # uncertainty = inputs['depth_uncertainty'][mask] + 1.
            # depth_loss = torch.log(uncertainty) + ((depth-target_depth)**2)/uncertainty
            # depth_loss = 0.003 * (depth_loss * depth_mask).sum() / (1e-8 + depth_mask.sum())

            depth_loss = (depth-target_depth)**2
            depth_loss = (depth_loss * depth_mask).sum() / (1e-8 + depth_mask.sum())
            # print('rgb_loss: ', rgb_loss, 'depth_loss: ', depth_loss)
            loss += (self.coef * depth_loss)


            if 'depth_coarse' in inputs:
                depth_coarse = inputs['depth_coarse'][mask]
                depth_coarse = to_inverse_normalized_depth(depth_coarse, near, far)
                depth_loss_c = (depth_coarse-target_depth)**2
                depth_loss_c = (depth_loss_c * depth_mask).sum() / (1e-8 + depth_mask.sum())
                # print('rgb_loss: ', rgb_loss, 'depth_loss: ', depth_loss, 'depth_loss_c: ', depth_loss_c)
                loss += (self.coef * depth_loss_c)

        if 'feat' in inputs and 'feat' in targets:
            feat_loss = 0.1 * ((inputs['feat'][mask] - targets['feat'][mask])**2).mean()
            loss += (self.coef * feat_loss)

        return loss
