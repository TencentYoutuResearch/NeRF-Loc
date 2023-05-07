"""
Author: jenningsliu
Date: 2022-06-30 10:51:58
LastEditors: jenningsliu
LastEditTime: 2022-08-11 12:24:10
FilePath: /nerf-loc/models/pointnerf/pointnerf_3d_model.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import copy
import math
import torch
import torch.nn as nn

from .model import ConditionalNeRF

class ConditionalNeRFSimple(ConditionalNeRF):
    def __init__(self, args, activation_func=nn.LeakyReLU(inplace=True)):
        super().__init__(args, activation_func)

        self.out_fc = nn.Linear(3+args.backbone2d_fpn_dim, args.model_3d_hidden_dim)
        self.proj_layer_3d_coarse = nn.Linear(args.model_3d_hidden_dim, args.matcher_hidden_dim)
        self.proj_layer_3d_fine = nn.Linear(args.model_3d_hidden_dim, args.matcher_hidden_dim)

    def query(self, data, xyz, support_featmaps, support_neural_points, 
            direction=None, K=8, embed_a=None, target_proj_mat=None):
        """
        Args: 
            data:
            xyz: target points in world frame (N,3)
            support_featmaps: feature maps of support views (V,C,h,w)
            support_neural_points:
                xyz: (M,3)
                feature: (M,C)
                confidence: (M,1)
            direction: (N,4) viewing direction and distance of sample points. in world frame
            embed_a: (1,C_a)
            target_proj_mat: (3,4)
        Returns: 
            feature: K-nearest support features (N,K,C1)
            weights: K-nearest support weights (N,K)
            feature_agg: aggregated support feature (N,C1)
        """                
        intrinsics = data['topk_Ks']
        intrinsics_hom = torch.eye(4, device=intrinsics.device).expand(intrinsics.shape[0], 4, 4).clone()
        intrinsics_hom[:,:3,:3] = intrinsics
        rgb, feat, mask = \
            self.projector.compute(xyz, intrinsics_hom, data['topk_poses'], data['topk_images'], support_featmaps)
        weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        multiview_feature = torch.cat([rgb, feat], dim=-1)
        multiview_visibility = mask.float()
        feature_agg = self.out_fc((multiview_feature * weight).sum(dim=1))
        return {
            'feature_agg': feature_agg,
            'multiview_feature': multiview_feature,
            'multiview_visibility': multiview_visibility
        }

    def query_coarse(self, data, points=None, embed_a=None):        
        if self.support_neural_points is None:
            self.build_support_neural_points(data)

        if points is None:
            pts3d, pts3d_ndc, _ = self.sample_points_3d()
        else:
            pts3d = points
            w2c_ref = data['topk_poses'][0].inverse()
            pts3d_ndc = (torch.matmul(w2c_ref[:3,:3], points.T) + w2c_ref[:3,3:]).T

        query_dict = self.query(
            data, pts3d, 
            support_featmaps=data['feat_coarse_src'].permute(0,3,1,2),
            support_neural_points=self.support_neural_points['coarse'],
            K=8,
            embed_a=embed_a
        )
        desc_3d = self.proj_layer_3d_coarse(query_dict['feature_agg'])

        return desc_3d, pts3d, pts3d_ndc

    def query_fine(self, data, points, embed_a=None):
        if self.support_neural_points is None:
            self.build_support_neural_points(data)

        query_dict = self.query(
            data, points, 
            support_featmaps=data['feat_fine_src'].permute(0,3,1,2),
            support_neural_points=self.support_neural_points['fine'],
            K=1,
            embed_a=embed_a
        )

        desc_3d = self.proj_layer_3d_fine(query_dict['feature_agg'])

        return desc_3d, None, None
